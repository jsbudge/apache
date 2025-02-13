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
from layers import FourierFeature, PulseLength, LKA1d, LKATranspose1d
import numpy as np

from utils import normalize, get_pslr


def init_weights(m):
    with contextlib.suppress(ValueError):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_normal_(m.weight)
        # sourcery skip: merge-nested-ifs
        if hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.fill_(.01)
                
def _xavier_init(model):
    """
    Performs the Xavier weight initialization.
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)
            # nn.init.he_(module.weight)


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

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        norms = grad_norm(self, norm_type=2)  # Compute 2-norm for each layer
        self.log_dict(norms)


class GeneratorModel(FlatModule):
    def __init__(self,
                 config: Config,
                 embedding: LightningModule | Config,
                 **kwargs,
                 ) -> None:
        super(GeneratorModel, self).__init__(config)
        self.n_ants = config.n_ants
        self.fft_len = config.fft_len
        self.clutter_latent_size = config.clutter_latent_size
        self.target_latent_size = config.target_latent_size
        self.fs = config.fs
        self.encoder_start_channel_sz = config.encoder_start_channel_sz
        self.encoder_layers = config.encoder_layers
        self.embedding_concatenation_channels = config.embedding_concatenation_channels
        self.clutter_target_channels = config.clutter_target_channels
        self.flowthrough_channels = config.flowthrough_channels
        self.decoder_layers = config.decoder_layers
        self.exp_to_ant_channels = config.exp_to_ant_channels
        self.automatic_optimization = False
        self.n_fourier_modes = config.n_fourier_modes

        self.embedding = embedding
        self.embedding.eval()
        for param in self.embedding.parameters():
            param.requires_grad = False

        '''CLUTTER ENCODING LAYERS'''
        # Calculate out layer sizes
        encoder_l = [nn.Sequential(
            nn.Conv1d(2, self.encoder_start_channel_sz, 1, 1, 0),
            nn.SiLU(),
        )]
        for n in range(self.encoder_layers):
            encoder_l.append(nn.Sequential(
                nn.Conv1d(self.encoder_start_channel_sz * 2**n, self.encoder_start_channel_sz * 2**(n + 1), 4, 2, 1),
                nn.SiLU(),
                LKA1d(self.encoder_start_channel_sz * 2**(n + 1), kernel_sizes=(513, 95), dilation=12),
                nn.LayerNorm(self.fft_len // 2**(n + 1)),
            ))
        encoder_l.append(nn.Conv1d(self.encoder_start_channel_sz * 2**self.encoder_layers, 1, 1, 1, 0))
        self.clutter_encoder = nn.Sequential(*encoder_l)
        self.clutter_encoder_final = nn.Sequential(
            nn.Linear(self.fft_len // 2**(n + 1), self.clutter_latent_size),
            nn.SiLU(),
        )

        '''TRANSFORMER'''
        self.predict_decoder = nn.Transformer(self.clutter_latent_size, num_decoder_layers=7, num_encoder_layers=7, nhead=8,
                                              batch_first=True, activation='gelu')
        # self.predict_decoder.apply(init_weights)

        '''EMBEDDING CONCATENATION LAYERS'''
        self.target_embedding_combine = nn.Sequential(
            TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=2, out_channels=self.embedding_concatenation_channels,
                   hidden_channels=self.embedding_concatenation_channels),
            LKA1d(self.embedding_concatenation_channels, kernel_sizes=(513, 513), dilation=12),
            nn.LayerNorm(self.target_latent_size),
            TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=self.embedding_concatenation_channels,
                   out_channels=self.flowthrough_channels, hidden_channels=self.embedding_concatenation_channels),
            nn.Linear(self.target_latent_size, self.clutter_latent_size),
            nn.SiLU(),
        )

        '''CLUTTER AND TARGET COMBINATION LAYERS'''
        self.clutter_target_combine = nn.Sequential(
            TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=self.flowthrough_channels, out_channels=self.clutter_target_channels,
                   hidden_channels=self.clutter_target_channels),
            LKA1d(self.clutter_target_channels, kernel_sizes=(255, 255), dilation=12),
            nn.LayerNorm(self.clutter_latent_size),
            LKA1d(self.clutter_target_channels, kernel_sizes=(255, 255), dilation=6),
            nn.LayerNorm(self.clutter_latent_size),
            TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=self.clutter_target_channels,
                   out_channels=self.flowthrough_channels, hidden_channels=self.clutter_target_channels),
        )
        '''WAVEFORM CREATION LAYERS'''
        self.expand_to_ants = nn.Sequential(
            TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=self.flowthrough_channels, out_channels=self.exp_to_ant_channels,
                   hidden_channels=self.exp_to_ant_channels),
            LKA1d(self.exp_to_ant_channels, kernel_sizes=(513, 513), dilation=12),
            nn.LayerNorm(self.clutter_latent_size),
            LKA1d(self.exp_to_ant_channels, kernel_sizes=(513, 513), dilation=12),
            nn.LayerNorm(self.clutter_latent_size),
            TFNO1d(n_modes_height=self.n_fourier_modes, in_channels=self.exp_to_ant_channels,
                   out_channels=self.flowthrough_channels, hidden_channels=self.exp_to_ant_channels),
        )
        # self.expand_to_ants.apply(init_weights)

        '''DECODER LAYERS'''
        self.clutter_decoder_start = nn.Sequential(
            nn.Linear(self.clutter_latent_size, self.fft_len // 2 ** self.decoder_layers),
            nn.SiLU(),
        )
        decode_l = [nn.Sequential(
            nn.Conv1d(self.flowthrough_channels, self.encoder_start_channel_sz * 2**self.decoder_layers, 1, 1, 0),
            nn.SiLU(),
        )]
        for n in range(self.decoder_layers, 0, -1):
            decode_l.append(nn.Sequential(
                nn.ConvTranspose1d(self.encoder_start_channel_sz * 2**n, self.encoder_start_channel_sz * 2**(n - 1), 4, 2, 1),
                nn.SiLU(),
                LKATranspose1d(self.encoder_start_channel_sz * 2**(n - 1), kernel_sizes=(513, 513), dilation=12),
                nn.LayerNorm(self.fft_len // 2**(n - 1)),
            ))
        decode_l.append(nn.Conv1d(self.encoder_start_channel_sz * 2**(n-1), self.n_ants * 2, 1, 1, 0))
        self.wave_decoder = nn.Sequential(*decode_l)

        self.window_threshold = nn.Threshold(1e-9, 0.)

        ''' PULSE LENGTH INFORMATION '''
        self.pinfo = nn.Sequential(
            FourierFeature(100., 50),
            nn.Linear(100, self.clutter_latent_size),
            nn.SiLU(),
        )

        self.plength = PulseLength()

        _xavier_init(self.wave_decoder)
        _xavier_init(self.predict_decoder)
        _xavier_init(self.clutter_encoder)
        _xavier_init(self.expand_to_ants)
        _xavier_init(self.clutter_target_combine)
        _xavier_init(self.target_embedding_combine)


        self.example_input_array = ([[torch.zeros((1, 32, 2, self.fft_len)),
                                      torch.zeros((1, self.target_latent_size)),
                                      torch.tensor([[1250]])]])

    def forward(self, inp: list) -> torch.tensor:
        clutter, target, pulse_length = inp
        # bw_info = self.fourier(torch.cat([pulse_length.float().view(-1, 1), bandwidth.view(-1, 1) / self.fs],
        #                                  dim=1))
        # Run clutter through the encoder
        x = torch.cat([self.clutter_encoder(clutter[:, n, ...]) for n in range(clutter.shape[1])], dim=1)
        x = self.clutter_encoder_final(x)

        # Run clutter through target embedding
        y = self.embedding(clutter[:, -1, ...]).unsqueeze(1)

        # Predict the next clutter step using transformer
        x = self.predict_decoder(x[:, :-1], x[:, 1:])[:, -1, ...].unsqueeze(1)

        # Add selected target into embedding and do some deep learning stuff
        y = self.target_embedding_combine(torch.cat([y, target.unsqueeze(1)], dim=1))

        # Combine clutter prediction with target information
        x = self.clutter_target_combine(x + y)

        # Run through LKA layers to create a waveform according to spec
        x = self.expand_to_ants(x * self.pinfo(pulse_length.float().view(-1, 1)).unsqueeze(1))
        # final_win = self.window(self.bw_integrate(x).squeeze(1), bandwidth.view(-1, 1) / self.fs).repeat_interleave(
        #     2, dim=1)
        x = self.wave_decoder(self.clutter_decoder_start(x)) * self.window_threshold(clutter[:, -1, ...])
        x = self.plength(x, pulse_length)
        x = x.view(-1, self.n_ants, 2, self.fft_len)
        return x

    def full_forward(self, clutter_array, target_array: Tensor | np.ndarray, pulse_length: int) -> torch.tensor:
        """
        This function is meant to simplify using the waveforms. It takes a set of frequency domain pulses and outputs a set of waveforms.
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

        # Now that everything is formatted, get the waveform
        if isinstance(target_array, np.ndarray):
            tt = torch.tensor(target_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            tt = target_array.to(self.device).unsqueeze(0)
        unformatted_waveform = self.getWaveform(nn_output=self.forward([clut.unsqueeze(0), tt, pl])).cpu().data.numpy()

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
        crossfiltered = gen_waveform * torch.flip(gen_waveform.conj(), dims=(1,))

        # Target and clutter power functions
        targ_ac = torch.sum(
            torch.abs(torch.fft.ifft(target_spectrum.unsqueeze(1) * mfiltered, dim=2)), dim=1)
        clut_ac = torch.sum(
            torch.abs(torch.fft.ifft(clutter_spectrum.unsqueeze(1) * mfiltered, dim=2)), dim=1)
        ratio = clut_ac / (1e-12 + targ_ac) * torch.fft.fftshift(torch.signal.windows.gaussian(self.fft_len, std=self.fft_len / 8., device=self.device))
        target_loss = ratio.nanmean()

        # Sidelobe loss functions
        slf = torch.abs(torch.fft.ifft(mfiltered, dim=2))
        slf[slf == 0] = 1e-9
        sidelobe_func = 10 * torch.log(slf / 10)
        pslrs = get_pslr(sidelobe_func.squeeze(1))
        sidelobe_loss = 1. / (1e-12 + torch.nanmean(pslrs))

        # Orthogonality
        if self.n_ants > 1:
            cross_sidelobe = torch.abs(torch.fft.ifft(crossfiltered, dim=2))
            cross_sidelobe[cross_sidelobe == 0] = 1e-9
            cross_sidelobe = 10 * torch.log(cross_sidelobe / 10)
            ortho_loss = torch.nanmean(torch.abs(sidelobe_func.sum(dim=1)[:, 0]) /
                                       (1e-12 + torch.abs(cross_sidelobe.sum(dim=1)[:, 0])))**2

            # Apply hinge loss
            ortho_loss = torch.abs(ortho_loss - .3)
        else:
            ortho_loss = torch.tensor(0., device=self.device)

        # Make sure the losses are positive (they should always be)
        sidelobe_loss = torch.abs(sidelobe_loss)
        target_loss = torch.abs(target_loss)

        return {'target_loss': target_loss,
                'sidelobe_loss': sidelobe_loss, 'ortho_loss': ortho_loss}

    def example_input_array(self) -> Optional[Union[Tensor, Tuple, Dict]]:
        return self.example_input_array

    def getWaveform(self, cc: Tensor = None, tc: Tensor = None, pulse_length=None,
                    bandwidth: [Tensor, float] = 400e6, nn_output: tuple[Tensor] = None,
                    use_window: bool = False, scale: bool = False, custom_fft_sz: int = 4096) -> Tensor:
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
        dual_waveform = self.forward([cc, tc, pulse_length]) if nn_output is None else nn_output
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
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss['loss'])
        # self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()
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
        clutter_spec, target_spec, target_enc, pulse_length = batch

        results = self.forward([clutter_spec, target_enc, pulse_length])
        train_loss = self.loss_function(results, clutter_spec, target_spec, target_enc, pulse_length)

        train_loss['loss'] = torch.sqrt(torch.abs(
            train_loss['sidelobe_loss'] * (1 + train_loss['target_loss'] + train_loss['ortho_loss'])))
        return train_loss
