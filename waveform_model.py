import contextlib
import pickle
from typing import Optional, Union, Tuple, Dict
import torch
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from torch.nn import functional as nn_func
from layers import FourierFeature, WindowGenerate, PulseLength, LKA1d


def init_weights(m):
    with contextlib.suppress(ValueError):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_normal_(m.weight)
        # sourcery skip: merge-nested-ifs
        if hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.fill_(.01)


class FlatModule(LightningModule):
    """
    Base Module for different encoder models and generators. This adds
    parameters to flatten the model to use in a loss landscape, should it be desired.
    """

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
        self.channel_sz = channel_sz
        de_ch_sz = channel_sz // 8

        self.predict_decoder = nn.Transformer(clutter_latent_size, num_decoder_layers=7, num_encoder_layers=7, nhead=6,
                                              batch_first=True, activation='gelu')
        self.predict_decoder.apply(init_weights)

        self.target_compressor = nn.Sequential(
            nn.Linear(target_latent_size, clutter_latent_size),
            nn.GELU(),
            nn.Linear(clutter_latent_size, clutter_latent_size),
            nn.Tanh(),
        )

        self.expand_to_ants = nn.Sequential(
            nn.Linear(clutter_latent_size, clutter_latent_size),
            nn.GELU(),
            nn.Conv1d(1, channel_sz, 1, 1, 0),
            nn.GELU(),
            LKA1d(channel_sz, kernel_sizes=(155, 95), dilation=12),
            LKA1d(channel_sz, kernel_sizes=(125, 125), dilation=6),
            LKA1d(channel_sz, kernel_sizes=(95, 155), dilation=3),
            nn.LayerNorm(clutter_latent_size),
            LKA1d(channel_sz, kernel_sizes=(155, 95), dilation=12),
            LKA1d(channel_sz, kernel_sizes=(125, 125), dilation=6),
            LKA1d(channel_sz, kernel_sizes=(95, 155), dilation=3),
            nn.LayerNorm(clutter_latent_size),
            nn.Conv1d(channel_sz, self.n_ants, 1, 1, 0),
            nn.GELU(),
            nn.Linear(clutter_latent_size, clutter_latent_size),
            nn.Tanh(),
        )
        self.expand_to_ants.apply(init_weights)

        self.fourier = nn.Sequential(
            FourierFeature(2, 24),
            nn.Linear(96, clutter_latent_size),
            nn.GELU(),
            nn.Linear(clutter_latent_size, clutter_latent_size),
            nn.GELU(),
        )
        self.fourier.apply(init_weights)

        self.bw_integrate = nn.Sequential(
            nn.Conv1d(self.n_ants, channel_sz, 1, 1, 0),
            nn.GELU(),
            LKA1d(channel_sz, kernel_sizes=(255, 255), dilation=12),
            nn.Conv1d(channel_sz, channel_sz, 4, 2, 1),
            nn.Conv1d(channel_sz, channel_sz, 4, 2, 1),
            nn.LayerNorm(clutter_latent_size // 4),
            nn.Conv1d(channel_sz, 1, 1, 1, 0),
            nn.Linear(clutter_latent_size // 4, self.n_ants, bias=False),
            nn.Softsign(),
        )
        self.bw_integrate.apply(init_weights)

        self.window_context = nn.Sequential(
            nn.Conv1d(self.n_ants * 2, de_ch_sz, 1, 1, 0),
            LKA1d(de_ch_sz, kernel_sizes=(255, 513), dilation=12),
            nn.LayerNorm(fft_sz),
            LKA1d(de_ch_sz, kernel_sizes=(255, 513), dilation=6),
            nn.LayerNorm(fft_sz),
            LKA1d(de_ch_sz, kernel_sizes=(255, 513), dilation=3),
            nn.LayerNorm(fft_sz),
            nn.Linear(fft_sz, fft_sz),
            nn.GELU(),
            nn.Conv1d(de_ch_sz, self.n_ants * 2, 1, 1, 0),
        )
        self.window_context.apply(init_weights)

        self.window = WindowGenerate(self.fft_sz, self.n_ants)

        self.window_threshold = nn.Threshold(1e-9, 0.)

        self.plength = PulseLength()

        self.example_input_array = ([[torch.zeros((1, 32, clutter_latent_size)),
                                      torch.zeros((1, target_latent_size)),
                                      torch.tensor([[1250]]), torch.tensor([[400e6]])]])

    def forward(self, inp: list) -> torch.tensor:
        clutter, target, pulse_length, bandwidth = inp
        bw_info = self.fourier(torch.cat([pulse_length.float().view(-1, 1), bandwidth.view(-1, 1) / self.fs],
                                         dim=1))
        x = self.predict_decoder(clutter, self.target_compressor(target).unsqueeze(1))
        x = self.expand_to_ants(x + bw_info.unsqueeze(1))
        final_win = self.window(self.bw_integrate(x).squeeze(1), bandwidth.view(-1, 1) / self.fs).repeat_interleave(
            2, dim=1)
        win_pos = self.window_threshold(final_win)
        decoded = torch.cat([self.decoder.decode(x[:, n, :]) for n in range(self.n_ants)], dim=1)
        x = self.window_context(decoded * win_pos) * final_win
        x = self.plength(x, pulse_length)
        x = x.view(-1, self.n_ants, 2, self.fft_sz)
        return x, bandwidth.view(-1, 1)

    def full_forward(self, clutter_array, target_array, pulse_length: int, bandwidth: float, mu_scale: float,
                     std_scale: float) -> torch.tensor:
        # Make everything into tensors and place on device
        clut = (torch.stack([torch.tensor(clutter_array.real, dtype=torch.float32, device=self.device),
                             torch.tensor(clutter_array.imag, dtype=torch.float32, device=self.device)]) -
                mu_scale) / std_scale
        clutter = self.decoder.encode(clut.swapaxes(0, 1))
        pl = torch.tensor([[pulse_length]], device=self.device)
        bw = torch.tensor([[bandwidth]], device=self.device)

        # Now that everything is formatted, get the waveform
        unformatted_waveform = (
            self.getWaveform(nn_output=self.forward([clutter.unsqueeze(0),
                                                     torch.tensor(target_array, dtype=torch.float32,
                                                                  device=self.device).unsqueeze(0),
                                                     pl, bw])).cpu().data.numpy())

        # Return it in complex form without the leading dimension
        return unformatted_waveform[0]

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
        mfiltered = gen_waveform * gen_waveform.conj()
        crossfiltered = gen_waveform * torch.flip(gen_waveform.conj(), dims=(1,))

        # Target and clutter power functions
        targ_ac = torch.sum(
            torch.abs(torch.fft.ifft(target_spectrum.unsqueeze(1) * mfiltered, dim=2)), dim=1)
        clut_ac = torch.sum(
            torch.abs(torch.fft.ifft(clutter_spectrum.unsqueeze(1) * mfiltered, dim=2)), dim=1)
        ratio = clut_ac / (1e-12 + targ_ac)
        if torch.any((targ_ac - clut_ac) > 0):
            target_loss = ratio[(targ_ac - clut_ac) > 0].nanmean()
        else:
            target_loss = ratio.nanmean()

        # Sidelobe loss functions
        slf = torch.abs(torch.fft.ifft(mfiltered, dim=2))
        slf[slf == 0] = 1e-9
        sidelobe_func = 10 * torch.log(slf / 10)
        slf_max = nn_func.max_pool1d_with_indices(
            sidelobe_func, 17, 12, padding=8)[0].detach()[:, :, -2]
        # Get the ISLR for this waveform
        sidelobe_loss = 10. / torch.nanmean(torch.abs(slf_max - torch.max(sidelobe_func, dim=-1)[0]))

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

    def save(self, fpath, model_name='current'):
        torch.save(self.state_dict(), f'{fpath}/{model_name}_wave_model.state')
        with open(f'{fpath}/{model_name}_model_params.pic', 'wb') as f:
            pickle.dump({'fft_sz': self.fft_sz,
                         'clutter_latent_size': self.clutter_latent_size,
                         'target_latent_size': self.target_latent_size, 'n_ants': self.n_ants,
                         'state_file': f'{fpath}/{model_name}_wave_model.state', 'channel_sz': self.channel_sz}, f)

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
