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

        self.transformer = nn.Transformer(clutter_latent_size, batch_first=True)

        self.expand_to_ants = nn.Sequential(
            nn.Conv1d(1, channel_sz, 1, 1, 0),
            nn.GELU(),
            nn.Conv1d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(channel_sz, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(channel_sz, self.n_ants, 1, 1, 0),
        )
        self.expand_to_ants.apply(init_weights)

        self.example_input_array = (torch.zeros((1, clutter_latent_size)), torch.zeros((1, target_latent_size)),
                                    torch.tensor([1250]), torch.tensor(400e6))

    def forward(self, inp: list) -> torch.tensor:
        clutter, target, pulse_length, bandwidth = inp

        x = self.transformer(clutter, target.unsqueeze(1))
        x = self.expand_to_ants(x)

        return x

    def loss_function(self, *args, **kwargs) -> dict:
        # These values are set here purely for debugging purposes
        dev = self.device
        n_ants = self.n_ants

        # Initialize losses to zero and place on correct device
        sidelobe_loss = torch.tensor(0., device=dev, requires_grad=False)
        target_loss = torch.tensor(0., device=dev)
        mainlobe_loss = torch.tensor(0., device=dev, requires_grad=False)
        ortho_loss = torch.tensor(0., device=dev)

        # Get clutter spectrum into complex form and normalize to unit energy
        clutter_spectrum = torch.complex(args[1][:, 0, :], args[1][:, 1, :])
        clutter_spectrum = clutter_spectrum / torch.sqrt(torch.sum(clutter_spectrum * torch.conj(clutter_spectrum),
                                                                   dim=1))[:, None]

        # Get target spectrum into complex form and normalize to unit energy
        target_spectrum = torch.complex(args[2][:, 0, :], args[2][:, 1, :])
        target_spectrum = target_spectrum / torch.sqrt(torch.sum(target_spectrum * torch.conj(target_spectrum),
                                                                 dim=1))[:, None]

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

            clutter_return = torch.abs(clutter_spectrum - g1) ** 2
            target_return = torch.abs(target_spectrum - g1) ** 2
            g1_return = torch.abs(g1 * g1.conj()) * 100.
            ratio = (target_return / clutter_return)
            ratio[torch.logical_and(clutter_spectrum == 0, target_spectrum == 0)] = (
                g1_return)[torch.logical_and(clutter_spectrum == 0, target_spectrum == 0)]
            ratio[torch.logical_and(clutter_spectrum == 0, target_spectrum != 0)] = 0.
            ratio[torch.isnan(ratio)] = g1_return[torch.isnan(ratio)]
            I_k = torch.exp(torch.sum(torch.log(torch.nanmean(ratio, dim=1))))
            target_loss += I_k / (gen_waveform.shape[0] * self.n_ants)

            # Get the ISLR for this waveform
            sidelobe_loss += (torch.nansum(torch.max(sidelobe_func, dim=-1)[0] / sll)
                              / (self.n_ants * sidelobe_func.shape[0]))

            # Make sure it stays within bandwidth
            mainlobe_loss += (torch.sum(torch.nansum((sidelobe_func - sll[:, None]) > 0, dim=1) / self.fft_sz) /
                              (self.n_ants * sidelobe_func.shape[0]))
            gn = g1.conj()  # Conjugate of current g1 for orthogonality loss on next loop

        # Apply hinge loss to sidelobes
        sidelobe_loss = (torch.abs(sidelobe_loss - .1))**(1/8)
        ortho_loss = (torch.abs(ortho_loss - .1))**(1/8)
        mainlobe_loss = (torch.abs(mainlobe_loss))**(1/8)
        target_loss = (torch.abs(target_loss))**(1/8)

        # Use sidelobe and orthogonality as regularization params for target loss
        # loss = torch.sqrt(target_loss**2 + sidelobe_loss**2 + ortho_loss**2)
        loss = torch.sqrt(torch.abs(target_loss * (1. + sidelobe_loss + ortho_loss + mainlobe_loss)))

        return {'loss': loss, 'target_loss': target_loss,
                'sidelobe_loss': sidelobe_loss, 'ortho_loss': ortho_loss, 'mainlobe_loss': mainlobe_loss}

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
        full_stft = self.forward([cc, tc, pulse_length, [bandwidth]]) if nn_output is None else nn_output
        gen_waveform = torch.zeros((full_stft.shape[0], self.n_ants, self.fft_sz), dtype=torch.complex64, device=self.device)
        for n in range(self.n_ants):
            dec = self.decoder.to('cpu').decode(full_stft[:, n, :].to('cpu')).to(self.device)
            g1 = torch.complex(dec[:, 0, :], dec[:, 1, :])
            g1 = g1 / torch.sqrt(torch.sum(g1 * torch.conj(g1), dim=1))[:, None]  # Unit energy calculation
            gen_waveform[:, n, ...] = g1
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
