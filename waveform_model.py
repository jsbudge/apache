import contextlib
import pickle
from typing import Optional, Union, Tuple, Dict
import torch
import yaml
from lightning_fabric import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn, Tensor, optim
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as nn_func
from torchaudio.functional import inverse_spectrogram
from numpy import log2, ceil
from torchvision import transforms
from scipy.signal import istft

from dataloaders import STFTModule
from layers import LSTMAttention, AttentionConv, ApplyBandwidth


def getTrainTransforms(var):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0., 0.), var),
        ]
    )


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
                 stft_win_sz: int,
                 clutter_latent_size: int,
                 target_latent_size: int,
                 n_ants: int,
                 activation: str = 'leaky',
                 fs: float = 2e9,
                 channel_sz: int = 48,
                 **kwargs,
                 ) -> None:
        super(GeneratorModel, self).__init__()

        self.n_ants = n_ants
        self.fft_sz = fft_sz
        self.stft_win_sz = stft_win_sz
        self.hop = self.stft_win_sz // 2
        self.clutter_latent_size = clutter_latent_size
        self.target_latent_size = target_latent_size
        self.fs = fs
        self.overlap = self.stft_win_sz - self.hop

        # Both the clutter and target stack standardize the output for any latent size
        self.clutter_stack = nn.Sequential(
            nn.Linear(clutter_latent_size, self.stft_win_sz - 1),
            nn.GELU(),
        )

        self.target_stack = nn.Sequential(
            nn.Linear(target_latent_size, self.stft_win_sz - 1),
            nn.GELU(),
        )

        # Mix together target and clutter
        self.mixture = nn.Sequential(
            nn.Conv1d(2, channel_sz, 1, 1, 0),
            nn.GELU(),
            nn.Conv1d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(channel_sz, 1, 1, 1, 0),
            nn.GELU(),
        )

        self.lst_hop = nn.Transformer(self.stft_win_sz, num_decoder_layers=6, num_encoder_layers=6, activation='gelu',
                                      batch_first=True)

        self.expand_to_ants = nn.Sequential(
            nn.Conv1d(1, channel_sz, 1, 1, 0),
            nn.GELU(),
            nn.Conv1d(channel_sz, channel_sz, 15, 1, 7),
            nn.GELU(),
            nn.Conv1d(channel_sz, channel_sz, 7, 1, 3),
            nn.GELU(),
            nn.Conv1d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(channel_sz, self.n_ants * 2, 1, 1, 0),
        )
        self.expand_to_ants.apply(init_weights)

        self.final = nn.Sequential(
            nn.Conv2d(self.n_ants * 2, channel_sz, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, (3, 1), 1, (1, 0)),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, (3, 1), 1, (1, 0)),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, (3, 1), 1, (1, 0)),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, (3, 1), 1, (1, 0)),
            nn.GELU(),
            nn.BatchNorm2d(channel_sz),
            nn.Conv2d(channel_sz, channel_sz, (3, 1), 1, (1, 0)),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, (3, 1), 1, (1, 0)),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, (3, 1), 1, (1, 0)),
            nn.GELU(),
            nn.Conv2d(channel_sz, channel_sz, (3, 1), 1, (1, 0)),
            nn.GELU(),
            nn.BatchNorm2d(channel_sz),
            nn.Conv2d(channel_sz, self.n_ants * 2, 1, 1, 0),
        )
        self.final.apply(init_weights)

        self.add_bw = ApplyBandwidth(fs)

        self.example_input_array = (torch.zeros((1, clutter_latent_size)), torch.zeros((1, target_latent_size)),
                                    torch.tensor([1250]), torch.tensor(400e6))

    def forward(self, clutter: torch.tensor, target: torch.tensor,
                pulse_length: [int], bandwidth: Tensor) -> torch.tensor:
        # Use only the first pulse_length because it gives batch_size random numbers as part of the dataloader
        n_frames = 1 + (pulse_length[0] - self.stft_win_sz) // self.hop
        n_windows = pulse_length[0] // self.stft_win_sz

        # Get clutter and target features, for LSTM input
        c_stack = self.clutter_stack(clutter).view(-1, 1, self.stft_win_sz - 1)
        t_stack = self.target_stack(target).view(-1, 1, self.stft_win_sz - 1)

        # Shape is (batch, 1, stft_win_sz, 1) after mixture and add_bw
        x = self.mixture(torch.concat([c_stack, t_stack], dim=1)).unsqueeze(3)
        x = self.add_bw(x, bandwidth[0])
        for n in range(n_windows - 1):
            # Target mask of the transformer is the last generated section
            if n == 0:
                l = self.lst_hop(x[:, :, :, n], torch.zeros((x.shape[0], x.shape[1], x.shape[2]), device=self.device))
            else:
                l = self.lst_hop(x[:, :, :, n], x[:, :, :, n - 1])
            x = torch.cat([x, l.unsqueeze(3)], dim=3)
        x = self.expand_to_ants(x.view(-1, 1, n_windows * self.stft_win_sz))
        x = torch.cat(
            [torch.view_as_real(torch.stft(torch.complex(x[:, n, :], x[:, n + 1, :]), self.stft_win_sz, self.hop,
                                           window=torch.windows.hann(self.stft_win_sz, device=self.device),
                                           center=True))
             for n in range(0, self.n_ants * 2, 2)], dim=3).swapaxes(1, 3).swapaxes(2, 3)
        return self.final(x)

    def loss_function(self, *args, **kwargs) -> dict:
        # These values are set here purely for debugging purposes
        dev = self.device
        n_ants = self.n_ants

        # Initialize losses to zero and place on correct device
        sidelobe_loss = torch.tensor(0., device=dev, requires_grad=False)
        target_loss = torch.tensor(0., device=dev)
        # bandwidth_loss = torch.tensor(0., device=dev, requires_grad=False)
        ortho_loss = torch.tensor(0., device=dev)

        # Get clutter spectrum into complex form and normalize to unit energy
        clutter_spectrum = torch.complex(args[1][:, :, 0], args[1][:, :, 1])
        clutter_spectrum = clutter_spectrum / torch.sqrt(torch.sum(clutter_spectrum * torch.conj(clutter_spectrum),
                                                                   dim=1))[:, None]
        # clutter_psd = clutter_spectrum * clutter_spectrum.conj()

        # Get target spectrum into complex form and normalize to unit energy
        target_spectrum = torch.complex(args[2][:, :, 0], args[2][:, :, 1])
        target_spectrum = target_spectrum / torch.sqrt(torch.sum(target_spectrum * torch.conj(target_spectrum),
                                                                 dim=1))[:, None]
        # target_psd = target_spectrum * target_spectrum.conj()

        # This is the weights for a weighted average that emphasizes locations that have more
        # energy difference between clutter and target
        # left_sig_tc = torch.abs(clutter_psd - target_psd)

        # Get waveform into complex form and normalize it to unit energy
        gen_waveform = self.getWaveform(nn_output=args[0])

        # Run losses for each channel
        for n in range(n_ants):
            g1 = gen_waveform[:, n, ...]
            slf = torch.abs(torch.fft.ifft(g1 * g1.conj(), dim=1))
            slf[slf == 0] = 1e-9
            sidelobe_func = 10 * torch.log(slf / 10)
            sll = nn_func.max_pool1d_with_indices(sidelobe_func, 65, 1,
                                                  padding=32)[0].unique(dim=1).detach()[:, 1]

            # This is orthogonality losses, so we need a persistent value across the for loop
            if n > 0:
                ortho_loss += torch.sum(torch.abs(g1 * gn)) / gen_waveform.shape[0]

            # Power in the leftover signal for both clutter and target
            # gen_psd = g1 * g1.conj()
            # left_sig_c = torch.abs(gen_psd - clutter_psd)

            clutter_return = torch.abs(clutter_spectrum - g1) ** 2
            target_return = torch.abs(target_spectrum - g1) ** 2
            g1_return = torch.abs(g1) * 100.
            ratio = (target_return / clutter_return)
            ratio[torch.logical_and(clutter_spectrum == 0, target_spectrum == 0)] = (
                g1_return)[torch.logical_and(clutter_spectrum == 0, target_spectrum == 0)]
            ratio[torch.logical_and(clutter_spectrum == 0, target_spectrum != 0)] = 0.
            I_k = torch.sum(torch.log(torch.nanmean(ratio, dim=1)))
            target_loss += I_k / (gen_waveform.shape[0] * self.n_ants)

            # Get the ISLR for this waveform
            sidelobe_loss += (torch.nansum(torch.max(sidelobe_func, dim=-1)[0] / sll)
                              / (self.n_ants * sidelobe_func.shape[0]))

            # Make sure it stays within bandwidth
            '''bwidth_fun = torch.log(g1_return)
            bwidth_target = args[3][0] / self.fs * self.fft_sz
            bwidth_fun += torch.abs(torch.max(bwidth_fun)) + 15
            bwidth_fun[bwidth_fun < 0] = 0.
            abs_bw = torch.nansum(bwidth_fun > 0, dim=1).type(torch.float32)
            rel_bw = torch.nansum(bwidth_fun, dim=1)
            abs_bw[abs_bw > bwidth_target] = rel_bw[abs_bw > bwidth_target]
            abs_bw[abs_bw <= bwidth_target] = (abs_bw[abs_bw <= bwidth_target] - bwidth_target)**2 / self.fft_sz
            bandwidth_loss += torch.sum(abs_bw) / (gen_waveform.shape[0] * self.n_ants)'''
            gn = g1.conj()  # Conjugate of current g1 for orthogonality loss on next loop

        # Apply hinge loss to sidelobes
        sidelobe_loss = (sidelobe_loss - .1) ** 2
        ortho_loss = (ortho_loss - .1) ** 2

        # Use sidelobe and orthogonality as regularization params for target loss
        # loss = torch.sqrt(target_loss**2 + sidelobe_loss**2 + ortho_loss**2)
        loss = torch.sqrt(torch.abs(target_loss * (1. + sidelobe_loss + ortho_loss)))

        return {'loss': loss, 'target_loss': target_loss,
                'sidelobe_loss': sidelobe_loss, 'ortho_loss': ortho_loss}

    def save(self, fpath, model_name='current'):
        torch.save(self.state_dict(), f'{fpath}/{model_name}_wave_model.state')
        with open(f'{fpath}/{model_name}_model_params.pic', 'wb') as f:
            pickle.dump({'fft_sz': self.fft_sz, 'stft_win_sz': self.stft_win_sz,
                         'clutter_latent_size': self.clutter_latent_size,
                         'target_latent_size': self.target_latent_size, 'n_ants': self.n_ants,
                         'state_file': f'{fpath}/{model_name}_wave_model.state'}, f)

    def getWindow(self, bin_bw):
        bin_bw += 0 if bin_bw % 2 == 0 else 1
        win_func = torch.zeros(self.stft_win_sz, device=self.device)
        win_func[:bin_bw // 2] = torch.windows.hann(bin_bw, device=self.device)[-bin_bw // 2:]
        win_func[-bin_bw // 2:] = torch.windows.hann(bin_bw, device=self.device)[:bin_bw // 2]
        return win_func

    def example_input_array(self) -> Optional[Union[Tensor, Tuple, Dict]]:
        return self.example_input_array

    def getWaveform(self, cc: Tensor = None, tc: Tensor = None, pulse_length=None,
                    bandwidth: Tensor = 400e6, nn_output: Tensor = None,
                    use_window: bool = False, scale: bool = False) -> Tensor:
        """
        Given a clutter and target spectrum, produces a waveform FFT.
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
        n_ants = self.n_ants
        stft_win = self.stft_win_sz

        # Get the STFT either from the clutter, target, and pulse length or directly from the neural net
        full_stft = self.forward(cc, tc, pulse_length, [bandwidth]) if nn_output is None else nn_output
        bin_bw = int(bandwidth / self.fs * self.stft_win_sz)
        bin_bw += 1 if bin_bw % 2 != 0 else 0
        if scale:
            new_fft_sz = int(2 ** (ceil(log2(pulse_length))))
            gen_waveform = torch.zeros((full_stft.shape[0], self.n_ants, new_fft_sz), dtype=torch.complex64,
                                       device=self.device, requires_grad=False)
        else:
            gen_waveform = torch.zeros((full_stft.shape[0], self.n_ants, self.fft_sz), dtype=torch.complex64,
                                       device=self.device, requires_grad=False)
        for n in range(0, n_ants * 2, 2):
            complex_stft = torch.complex(full_stft[:, n, :, :], full_stft[:, n + 1, :, :])

            # Apply a window if wanted for actual simulation
            if use_window:
                g1 = torch.istft(complex_stft * self.getWindow(self.bin_bw)[None, :, None],
                                 stft_win, hop_length=self.hop,
                                 window=torch.windows.hann(self.stft_win_sz, device=self.device),
                                 return_complex=True, onesided=False, center=True)
            else:
                # This is for training purposes
                g1 = torch.istft(complex_stft, stft_win, hop_length=self.hop,
                                 window=torch.windows.hann(self.stft_win_sz, device=self.device),
                                 return_complex=True, onesided=False, center=True)
            if scale:
                g1 = torch.fft.fft(g1, new_fft_sz, dim=-1)
            else:
                g1 = torch.fft.fft(g1, self.fft_sz, dim=-1)
            g1 = g1 / torch.sqrt(torch.sum(g1 * torch.conj(g1), dim=1))[:, None]  # Unit energy calculation
            gen_waveform[:, n // 2, ...] = g1
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
