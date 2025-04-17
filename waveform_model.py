import contextlib
import math
import pickle
from typing import Optional, Union, Tuple, Dict
import torch
from neuralop import TFNO1d
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from torch.nn import functional as nn_func
from pytorch_lightning.utilities import grad_norm
from torch.optim import Optimizer
from config import Config
from layers import FourierFeature, PulseLength, LKA1d, LKATranspose1d, WindowGenerate
import numpy as np

from utils import normalize, get_pslr, _xavier_init

EXP_1 = 0.36787944117144233
EPS = 1e-12


class FlatModule(LightningModule):
    """
    Base Module for different encoder models and generators. This adds
    parameters to flatten the model to use in a loss landscape, should it be desired.
    """

    def __init__(self, config: Config = None):
        super(FlatModule, self).__init__()
        self.config = config

    def get_flat_params(self):
        """Get flattened and concatenated params of the model."""
        return torch.cat([torch.flatten(p) for _, p in self._get_params().items()])

    def _get_params(self):
        return {name: param.data for name, param in self.named_parameters()}

    def init_from_flat_params(self, flat_params):
        """Set all model parameters from the flattened form."""
        assert isinstance(flat_params, torch.Tensor), "Argument to init_from_flat_params() must be torch.Tensor"
        state_dict = self._unflatten_to_state_dict(flat_params, self._get_param_shapes())
        for name, params in self.state_dict().items():
            if name not in state_dict:
                state_dict[name] = params
        self.load_state_dict(state_dict, strict=True)

    def _unflatten_to_state_dict(self, flat_w, shapes):
        state_dict = {}
        counter = 0
        for shape in shapes:
            name, tsize, tnum = shape
            param = flat_w[counter: counter + tnum].reshape(tsize)
            state_dict[name] = torch.nn.Parameter(param)
            counter += tnum
        assert counter == len(flat_w), "counter must reach the end of weight vector"
        return state_dict

    def _get_param_shapes(self):
        return [
            (name, param.shape, param.numel())
            for name, param in self.named_parameters()
        ]


class GeneratorModel(FlatModule):
    def __init__(self,
                 config: Config,
                 **kwargs,
                 ) -> None:
        super(GeneratorModel, self).__init__(config)
        self.n_ants = config.n_ants
        self.fft_len = config.fft_len
        self.target_latent_size = config.target_latent_size
        self.fs = config.fs
        self.clutter_target_channels = config.clutter_target_channels
        self.flowthrough_channels = config.flowthrough_channels
        self.wave_decoder_channels = config.wave_decoder_channels
        self.waveform_channels = config.waveform_channels
        self.target_channels = config.target_channels
        self.automatic_optimization = False
        self.n_fourier_modes = config.n_fourier_modes
        self.bandwidth = config.bandwidth
        self.baseband_fc = (config.fc % self.fs) - self.fs

        '''TRANSFORMER'''
        # self.predict_lstm = nn.LSTM(self.target_latent_size, self.target_latent_size, self.lstm_layers, batch_first=True)
        self.predict_decoder = nn.Transformer(self.target_latent_size, num_decoder_layers=7, num_encoder_layers=7, nhead=8,
                                              batch_first=True, activation=nn.SiLU())
        # self.predict_decoder.apply(init_weights)

        '''CLUTTER AND TARGET COMBINATION LAYERS'''
        self.clutter_encoder = nn.Sequential(
            nn.Linear(self.fft_len, self.target_latent_size),
            nn.SiLU(),
        )
        self.clutter_squash = nn.Sequential(
            nn.Linear(2, 1),
            nn.SiLU(),
        )

        self.clutter_target_init = nn.Sequential(
            TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=2, out_channels=self.flowthrough_channels,
                   hidden_channels=self.clutter_target_channels, non_linearity=nn.SiLU()),
            LKA1d(self.clutter_target_channels, kernel_sizes=(255, 129), dilation=12, activation='silu'),
        )

        '''SKIP LAYERS'''
        self.waveform_layers = nn.ModuleList()
        self.target_layers = nn.ModuleList()
        for _ in range(config.n_skip_layers):
            self.waveform_layers.append(nn.Sequential(
            TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=self.flowthrough_channels + 2,
                   out_channels=self.waveform_channels, non_linearity=nn.SiLU(),
                   hidden_channels=self.waveform_channels),
            TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=self.waveform_channels,
                   out_channels=self.flowthrough_channels, non_linearity=nn.SiLU(), hidden_channels=self.waveform_channels),
            ))
            self.target_layers.append(nn.Sequential(
                TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=self.flowthrough_channels + 1,
                       out_channels=self.target_channels, non_linearity=nn.SiLU(),
                       hidden_channels=self.target_channels),
                TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=self.target_channels, non_linearity=nn.SiLU(),
                       out_channels=self.flowthrough_channels, hidden_channels=self.target_channels),
            ))

        self.wave_decoder = nn.Sequential(
            TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=self.flowthrough_channels + 2,
                   out_channels=self.wave_decoder_channels, non_linearity=nn.SiLU(),
                   hidden_channels=self.wave_decoder_channels),
            TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=self.wave_decoder_channels, non_linearity=nn.SiLU(),
                   out_channels=self.n_ants * 2, hidden_channels=self.wave_decoder_channels),
            nn.Linear(self.target_latent_size, self.fft_len),
        )

        ''' PULSE LENGTH INFORMATION '''
        self.pinfo = nn.Sequential(
            FourierFeature(100., 50),
            nn.Linear(100, self.target_latent_size),
            nn.SiLU(),
        )

        self.bandwidth_info = nn.Sequential(
            FourierFeature(.5, 50),
            nn.Linear(100, self.target_latent_size),
            nn.SiLU(),
        )

        self.plength = PulseLength()
        self.bw_generate = WindowGenerate(self.fft_len, self.n_ants)

        _xavier_init(self.wave_decoder)
        _xavier_init(self.predict_decoder)
        _xavier_init(self.clutter_target_init)
        _xavier_init(self.pinfo)
        _xavier_init(self.bandwidth_info)


        '''self.example_input_array = (torch.zeros((1, 32, 2, self.fft_len)),
                                      torch.zeros((1, self.target_latent_size)),
                                      torch.tensor([[1250]]), torch.tensor([[.4]]))'''

    def forward(self, clutter, target, pulse_length, bandwidth) -> torch.tensor:

        # Run clutter through the encoder
        x = self.clutter_encoder(clutter)
        x = self.clutter_squash(x.swapaxes(-2, -1))
        x = x.squeeze(-1)

        # Predict the next clutter step using transformer
        x = self.predict_decoder(x[:, :-1], x[:, 1:])[:, -1, ...].unsqueeze(1)

        # Combine clutter prediction with target information
        x = self.clutter_target_init(torch.cat([x, target.unsqueeze(1)], dim=1))

        bw_info = self.bandwidth_info(bandwidth.float().view(-1, 1)).unsqueeze(1)
        pl_info = self.pinfo(pulse_length.float().view(-1, 1)).unsqueeze(1)

        # Run through LKA layers to create a waveform according to spec
        for wl, tl in zip(self.waveform_layers, self.target_layers):
            x = wl(torch.cat([x, bw_info, pl_info], dim=1))
            x = tl(torch.cat([x, target.unsqueeze(1)], dim=1))
        x = self.wave_decoder(torch.cat([x, bw_info, pl_info], dim=1))
        x = self.plength(x, pulse_length) * self.bw_generate(bandwidth)
        x = x.view(-1, self.n_ants, 2, self.fft_len)#  * bump
        return x

    def full_forward(self, clutter_array, target_array: Tensor | np.ndarray, pulse_length: int, bandwidth: float) -> torch.tensor:
        """
        This function is meant to simplify using the waveforms. It takes a set of frequency domain pulses and outputs a set of waveforms.
        :param bandwidth: Normalized bandwidth. Should be between 0 and 1.
        :param clutter_array: N x fft_len array of pulse data.
        :param target_array: 1 x target_latent_size array of target embedding vector.
        :param pulse_length: int giving the length of the expected pulse in TAC.
        :return: n_ant x fft_len array of waveforms.
        """
        # Make everything into tensors and place on device
        clut_norm = normalize(clutter_array)
        clut_norm = (clut_norm - clut_norm.mean()) / clut_norm.std()
        clut = (torch.stack([torch.tensor(clut_norm.real, dtype=torch.float32, device=self.device),
                             torch.tensor(clut_norm.imag, dtype=torch.float32, device=self.device)])).swapaxes(0, 1)
        pl = torch.tensor([[pulse_length]], device=self.device)
        bw = torch.tensor([bandwidth], device=self.device)

        # Now that everything is formatted, get the waveform
        if isinstance(target_array, np.ndarray):
            tt = torch.tensor(target_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            tt = target_array.to(self.device).unsqueeze(0)
        unformatted_waveform = self.getWaveform(nn_output=self.forward(clut.unsqueeze(0), tt, pl, bw)).cpu().data.numpy()

        # Return it in complex form without the leading dimension
        return unformatted_waveform[0]

    def loss_function(self, *args, **kwargs) -> dict:

        # Get clutter spectrum into complex form and normalize to unit energy
        clutter_spectrum = torch.fft.fftshift(torch.complex(args[1][:, -1, 0, :], args[1][:, -1, 1, :]), dim=1)
        clutter_spectrum = clutter_spectrum / torch.sqrt(torch.sum(clutter_spectrum * torch.conj(clutter_spectrum),
                                                                   dim=1))[:, None]

        # Get target spectrum into complex form and normalize to unit energy
        target_spectrum = torch.fft.fftshift(torch.complex(args[2][:, 0, :], args[2][:, 1, :]), dim=1)
        target_spectrum = target_spectrum / torch.sqrt(torch.sum(target_spectrum * torch.conj(target_spectrum),
                                                                 dim=1))[:, None]

        # Get waveform into complex form and normalize it to unit energy
        gen_waveform = self.getWaveform(nn_output=args[0])
        mfiltered = gen_waveform * gen_waveform.conj()

        # Target and clutter power functions
        targ_ac = torch.sum(
            torch.abs(torch.fft.ifft(target_spectrum.unsqueeze(1) * mfiltered, dim=2)), dim=1)
        clut_ac = torch.sum(
            torch.abs(torch.fft.ifft(clutter_spectrum.unsqueeze(1) * mfiltered, dim=2)), dim=1)
        target_loss = torch.nanmean(clut_ac / (EPS + targ_ac))
        # target_loss = torch.nanmean(nn_func.cosine_similarity(targ_ac, clut_ac)**2)

        # Sidelobe loss functions
        slf = nn_func.threshold(torch.abs(torch.fft.ifft(mfiltered, dim=2)), 1e-9, 1e-9)
        sidelobe_func = 10 * torch.log(slf / 10)
        theoretical_mainlobe_width = torch.max(torch.ceil(1 / (2 * args[5]))).int()
        pslrs = torch.max(sidelobe_func, dim=-1)[0] - torch.mean(sidelobe_func[..., theoretical_mainlobe_width:-theoretical_mainlobe_width], dim=-1)
        # pslrs = get_pslr(sidelobe_func.squeeze(1))
        sidelobe_loss = 1. / (EPS + torch.nanmean(pslrs))
        # sidelobe_loss = torch.tensor(0., device=self.device)

        # Bandwidth loss

        # Orthogonality
        if self.n_ants > 1:
            crossfiltered = gen_waveform * torch.flip(gen_waveform.conj(), dims=(1,))
            cross_sidelobe = torch.abs(torch.fft.ifft(crossfiltered, dim=2))
            cross_sidelobe[cross_sidelobe == 0] = 1e-9
            cross_sidelobe = 10 * torch.log(cross_sidelobe / 10)
            ortho_loss = torch.nanmean(torch.abs(sidelobe_func.sum(dim=1)[:, 0]) /
                                       (EPS + torch.abs(cross_sidelobe.sum(dim=1)[:, 0])))**2

            # Apply hinge loss
            ortho_loss = torch.abs(ortho_loss - .3)
        else:
            ortho_loss = torch.tensor(0., device=self.device)

        # Make sure the losses are positive (they should always be)
        sidelobe_loss = torch.abs(sidelobe_loss)
        target_loss = torch.abs(target_loss)

        return {'target_loss': target_loss,
                'sidelobe_loss': sidelobe_loss, 'ortho_loss': ortho_loss}

    # def example_input_array(self) -> Optional[Union[Tensor, Tuple, Dict]]:
    #     return self.example_input_array

    def getWaveform(self, cc: Tensor = None, tc: Tensor = None, pulse_length=None,
                    bandwidth: [Tensor, float] = 400e6, nn_output: tuple[Tensor] = None) -> Tensor:
        """
        Given a clutter and target spectrum, produces a waveform FFT.
        :param custom_fft_sz: If scale is True, outputs waveforms of custom_fft_sz
        :param bandwidth: Desired bandwidth of waveform.
        :param scale: If True, scales the output FFT so that it is at least pulse_length long on IFFT.
        :param pulse_length: Length of pulse in samples.
        :param use_window: if True, applies a window to the finished waveform. Set to False for training.
        :param nn_output: Optional. If the waveform data is already created, use this to avoid putting in cc and tc.
        :param cc: Tensor of clutter spectrum. Same as input to model.
        :param tc: Tensor of target spectrum. Same as input to model.
        :return: Tensor of waveform FFTs, of size (batch_sz, n_ants, fft_sz).
        """
        if pulse_length is None:
            pulse_length = [1]

        # Get the STFT either from the clutter, target, and pulse length or directly from the neural net
        dual_waveform = self.forward(cc, tc, pulse_length, bandwidth) if nn_output is None else nn_output
        gen_waveform = torch.zeros((dual_waveform.shape[0], self.n_ants, self.fft_len), dtype=torch.complex64,
                                   device=self.device)
        for n in range(self.n_ants):
            complex_wave = torch.fft.fftshift(torch.complex(dual_waveform[:, n, 0, :],
                                                            dual_waveform[:, n, 1, :]), dim=1)
            gen_waveform[:, n, :] = (complex_wave / torch.sqrt(torch.sum(complex_wave *
                                                                         torch.conj(complex_wave),
                                                                         dim=1))[:, None])  # Unit energy calculation
        return gen_waveform

    def training_step(self, batch, batch_idx):
        train_loss = self.train_val_get(batch, batch_idx)
        self.manual_backward(train_loss['loss'])
        opt = self.optimizers()
        opt.step()
        opt.zero_grad()
        # self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        '''if self.global_step % self.config.accumulation_steps == 0:
            opt = self.optimizers()
            opt.step()
            opt.zero_grad()'''
        self.log_dict(train_loss, sync_dist=True,
                      prog_bar=True, rank_zero_only=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.log_dict(self.train_val_get(batch, batch_idx), sync_dist=True, prog_bar=True,
                      rank_zero_only=True)

    def on_validation_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        else:
            sch.step()
        if self.trainer.is_global_zero and not self.config.is_tuning and self.config.loss_landscape:
            self.optim_path.append(self.get_flat_params())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.config.lr,
                                      weight_decay=self.config.weight_decay,
                                      betas=self.config.betas,
                                      eps=1e-7)
        if self.config.scheduler_gamma is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config.scheduler_gamma)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx):
        clutter_spec, target_spec, target_enc, pulse_length, bandwidth = batch

        results = self.forward(clutter_spec, target_enc, pulse_length, bandwidth)
        train_loss = self.loss_function(results, clutter_spec, target_spec, target_enc, pulse_length, bandwidth)

        train_loss['loss'] = torch.sqrt(torch.abs(
            train_loss['target_loss'] * (1 + train_loss['sidelobe_loss'] + train_loss['ortho_loss'])))# + train_loss['bandwidth_loss']
        return train_loss

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if self.global_step % self.config.log_epoch == 0:
            # norms = grad_norm(self, norm_type=2)  # Compute 2-norm for each layer
            norms = {**grad_norm(self.clutter_encoder, norm_type=2),
                     **grad_norm(self.clutter_squash, norm_type=2),
                     **grad_norm(self.clutter_target_init, norm_type=2),
                     **grad_norm(self.wave_decoder, norm_type=2)}
            self.log_dict(norms)
