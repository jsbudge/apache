import contextlib
import pickle
from typing import Optional, Union, Tuple, Dict
import torch
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from torch.nn import functional as nn_func
from torchvision import transforms
from layers import FourierFeature, WindowConvolution, Block1d, WindowGenerate


def init_weights(m):
    with contextlib.suppress(ValueError):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_normal_(m.weight)
        # sourcery skip: merge-nested-ifs
        if hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.fill_(.01)


class FlatModule(LightningModule):

    def __init__(self):
        super(FlatModule, self).__init__()

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
                 fft_sz: int,
                 clutter_latent_size: int,
                 target_latent_size: int,
                 n_ants: int,
                 decoder: LightningModule,
                 fs: float = 2e9,
                 channel_sz: int = 64,
                 **kwargs,
                 ) -> None:
        super(GeneratorModel, self).__init__()

        self.n_ants = n_ants
        self.fft_sz = fft_sz
        self.clutter_latent_size = clutter_latent_size
        self.target_latent_size = target_latent_size
        self.fs = fs
        self.decoder = decoder
        self.decoder.eval()
        self.decoder.requires_grad = False

        self.transformer = nn.Transformer(clutter_latent_size, num_decoder_layers=7, num_encoder_layers=7,
                                          activation='gelu', batch_first=True)
        self.transformer.apply(init_weights)

        self.expand_to_ants = nn.Sequential(
            nn.Conv1d(1, channel_sz, 1, 1, 0),
            nn.GELU(),
            Block1d(channel_sz),
            Block1d(channel_sz),
            Block1d(channel_sz),
            Block1d(channel_sz),
            nn.Conv1d(channel_sz, self.n_ants, 1, 1, 0),
            nn.GELU(),
            nn.Linear(clutter_latent_size, clutter_latent_size),
            nn.Tanh(),
        )
        self.expand_to_ants.apply(init_weights)

        '''self.fourier = nn.Sequential(
            FourierFeature(2, 24),
            nn.Linear(96, clutter_latent_size),
            nn.GELU(),
            nn.Linear(clutter_latent_size, clutter_latent_size),
            nn.GELU(),
        )
        self.fourier.apply(init_weights)'''

        self.win_shift = WindowGenerate(fs, fft_sz, n_ants)

        self.bw_integrate = nn.Sequential(
            nn.Linear(clutter_latent_size, clutter_latent_size),
            nn.GELU(),
            nn.Linear(clutter_latent_size, 1),
            nn.Tanh(),
        )
        self.bw_integrate.apply(init_weights)

        self.clutter_mod = nn.Sequential(
            nn.Linear(clutter_latent_size, clutter_latent_size),
            nn.GELU(),
        )
        self.clutter_mod.apply(init_weights)

        self.target_mod = nn.Sequential(
            nn.Linear(target_latent_size, target_latent_size),
            nn.GELU(),
        )
        self.target_mod.apply(init_weights)

        self.example_input_array = ([[torch.zeros((1, 32, clutter_latent_size)), torch.zeros((1, target_latent_size)),
                                      torch.tensor([[1250]]), torch.tensor([[400e6]])]])

    def forward(self, inp: list) -> torch.tensor:
        clutter, target, pulse_length, bandwidth = inp
        # bw_win = self.fourier(torch.cat([pulse_length.float().view(-1, 1), bandwidth / self.fs],
        #                                 dim=1)).view(-1, 1, self.clutter_latent_size)
        mod_clut = self.clutter_mod(clutter)
        mod_targ = self.target_mod(target).unsqueeze(1)
        x = self.transformer(mod_clut, mod_targ)
        x = self.expand_to_ants(x)
        bw_win = self.win_shift(self.bw_integrate(x), bandwidth)
        decoded = [self.decoder.decode(x[:, n, :]) * bw_win for n in range(self.n_ants)]
        x = torch.cat([d.unsqueeze(1) for d in decoded], dim=1)
        return x, bandwidth

    def full_forward(self, clutter_array, target_array, pulse_length: int, bandwidth: float) -> torch.tensor:
        # Make everything into tensors and place on device
        clutter = self.decoder.encode(torch.tensor(clutter_array, dtype=torch.float32, device=self.device).unsqueeze(0))
        target = self.decoder.encode(torch.tensor(target_array, dtype=torch.float32, device=self.device).unsqueeze(0))
        pl = torch.tensor([[pulse_length]], device=self.device)
        bw = torch.tensor([[bandwidth]], device=self.device)
        return self.getWaveform(nn_output=self.forward([clutter, target, pl, bw])).cpu().data.numpy()

    def loss_function(self, *args, **kwargs) -> dict:

        # Get clutter spectrum into complex form and normalize to unit energy
        clutter_spectrum = torch.fft.fftshift(torch.complex(args[1][:, 0, :], args[1][:, 1, :]), dim=1)
        clutter_spectrum = clutter_spectrum / torch.sqrt(torch.sum(clutter_spectrum * torch.conj(clutter_spectrum),
                                                                   dim=1))[:, None]

        # Get target spectrum into complex form and normalize to unit energy
        target_spectrum = torch.fft.fftshift(torch.complex(args[2][:, 0, :], args[2][:, 1, :]), dim=1)
        target_spectrum = target_spectrum / torch.sqrt(torch.sum(target_spectrum * torch.conj(target_spectrum),
                                                                 dim=1))[:, None]

        # Get waveform into complex form and normalize it to unit energy
        gen_waveform = self.getWaveform(nn_output=args[0])
        gconj = gen_waveform.conj()

        # Target and clutter power functions
        '''targ_power = torch.abs(target_spectrum.unsqueeze(1) - gen_waveform)**2
        clut_power = torch.abs(clutter_spectrum.unsqueeze(1) - gen_waveform)**2
        ratio = 1 + targ_power / (1e-6 + clut_power)
        target_loss = torch.nanmean(torch.log(torch.nanmean(ratio, dim=2)))'''
        bw_bin = int(args[0][1][0] / self.fs * self.fft_sz / 2)
        targ_ac = torch.fft.fftshift(torch.abs(torch.fft.ifft(target_spectrum.unsqueeze(1) * gen_waveform * gconj, dim=2)), dim=2)
        clut_ac = torch.fft.fftshift(torch.abs(torch.fft.ifft(clutter_spectrum.unsqueeze(1) * gen_waveform * gconj, dim=2)), dim=2)
        ratio = clut_ac[:, :, self.fft_sz // 2 - bw_bin:self.fft_sz // 2 + bw_bin] / (1e-6 + targ_ac[:, :, self.fft_sz // 2 - bw_bin:self.fft_sz // 2 + bw_bin])
        target_loss = 1. - torch.nanmean(ratio.mean(dim=2)[0])

        # Sidelobe loss functions
        slf = torch.abs(torch.fft.ifft(gen_waveform * gconj, dim=2))
        slf[slf == 0] = 1e-9
        sidelobe_func = 10 * torch.log(slf / 10)
        slf = nn_func.max_pool2d_with_indices(sidelobe_func,
                                              (1, 65), (1, 1), padding=(0, 32))[0].unique(dim=2).detach()[:, :, 1]
        # Get the ISLR for this waveform
        sidelobe_loss = torch.nanmean(torch.max(sidelobe_func, dim=-1)[0] / slf)

        # Orthogonality
        ortho_loss = 1. - (torch.nanmean(torch.abs(gen_waveform * torch.flip(gconj, dims=(1,)))) /
                      torch.max(torch.abs(gen_waveform * torch.flip(gconj, dims=(1,)))))

        # Bandwidth Losses
        '''bwf = torch.cumsum(torch.abs(gen_waveform * gconj), dim=2)
        bwf = bwf / bwf[:, :, -1][:, :, None]
        bwf = torch.nansum(bwf > .5, dim=2) / self.fft_sz * self.fs
        bandwidth_loss = torch.nanmean(torch.sqrt(torch.abs(args[0][1] - bwf) / args[0][1]))'''

        # Apply hinge loss to sidelobes
        sidelobe_loss = torch.abs(sidelobe_loss - .1)
        ortho_loss = torch.abs(ortho_loss - .1)
        target_loss = torch.abs(target_loss)

        return {'target_loss': target_loss,
                'sidelobe_loss': sidelobe_loss, 'ortho_loss': ortho_loss}

    def save(self, fpath, model_name='current'):
        torch.save(self.state_dict(), f'{fpath}/{model_name}_wave_model.state')
        with open(f'{fpath}/{model_name}_model_params.pic', 'wb') as f:
            pickle.dump({'fft_sz': self.fft_sz,
                         'clutter_latent_size': self.clutter_latent_size,
                         'target_latent_size': self.target_latent_size, 'n_ants': self.n_ants,
                         'state_file': f'{fpath}/{model_name}_wave_model.state'}, f)

    def getWindow(self, params):
        # print(params)
        bin_bw = int(params[0] / self.fs * self.fft_sz)
        bin_bw += 0 if bin_bw % 2 == 0 else 1
        win_func = torch.zeros(self.fft_sz, device=self.device)
        # win_func[-bin_bw:] = torch.windows.hann(bin_bw, device=self.device)
        win_func[:bin_bw // 2] = torch.windows.hann(bin_bw, device=self.device)[-bin_bw // 2:]
        win_func[-bin_bw // 2:] = torch.windows.hann(bin_bw, device=self.device)[:bin_bw // 2]
        return win_func

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
        bandwidth = bandwidth if nn_output is None else nn_output[1]
        dual_waveform, _ = self.forward([cc, tc, pulse_length, bandwidth]) if nn_output is None else nn_output
        gen_waveform = torch.zeros((dual_waveform.shape[0], self.n_ants, self.fft_sz), dtype=torch.complex64,
                                   device=self.device)
        for n in range(self.n_ants):
            complex_wave = torch.fft.fftshift(torch.complex(dual_waveform[:, n, 0, :],
                                                            dual_waveform[:, n, 1, :]), dim=1)
            gen_waveform[:, n, :] = (complex_wave / torch.sqrt(torch.sum(complex_wave *
                                                                         torch.conj(complex_wave),
                                                                         dim=1))[:, None])  # Unit energy calculation
        return gen_waveform


class RCSModel(FlatModule):
    def __init__(self,
                 ) -> None:
        super(RCSModel, self).__init__()

        nchan = 32

        self.optical_stack = nn.Sequential(
            nn.Conv2d(3, nchan, 129, 1, 64),
            nn.LeakyReLU(),
            nn.Conv2d(nchan, nchan, 65, 1, 32),
            nn.LeakyReLU(),
            nn.Conv2d(nchan, nchan, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(nchan, nchan, 1, 1, 0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(nchan),
        )

        self.pose_stack = nn.Sequential(
            nn.Linear(7, 64),
            nn.LeakyReLU(),
        )

        self.pose_inflate = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 4, 2, 1),
            nn.ConvTranspose2d(1, 1, 4, 2, 1),
            nn.ConvTranspose2d(1, 1, 4, 2, 1),
            nn.ConvTranspose2d(1, 1, 4, 2, 1),
            nn.ConvTranspose2d(1, 1, 4, 2, 1),
            nn.LeakyReLU(),
        )

        self.comb_stack = nn.Sequential(
            nn.Conv2d(nchan + 1, nchan, 3, 1, 1),
            nn.Tanh(),
            nn.Conv2d(nchan, nchan, 3, 1, 1),
            nn.Tanh(),
            nn.Conv2d(nchan, nchan, 3, 1, 1),
            nn.Tanh(),
            nn.Conv2d(nchan, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

        self.loss = nn.MSELoss()

    def forward(self, opt_data: torch.tensor, pose_data: torch.tensor) -> torch.tensor:
        w = self.pose_stack(pose_data)
        w = self.pose_inflate(w.view(-1, 1, 8, 8))
        x = torch.concat((self.optical_stack(opt_data), w), dim=1)
        return self.comb_stack(x).swapaxes(2, 3)

    def loss_function(self, *args, **kwargs) -> dict:
        return {'loss': self.loss(args[0], args[1])}
