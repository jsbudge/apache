import contextlib
import pickle
from typing import List, Any, TypeVar
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from numpy import log2, ceil
from torchvision import transforms
from layers import RichConvTranspose2d, RichConv2d, Linear2d, SelfAttention, STFTAttention

Tensor = TypeVar('torch.tensor')


def getTrainTransforms(var):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0., 0.), var),
        ]
    )


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
        with contextlib.suppress(AttributeError):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)


class GeneratorModel(LightningModule):
    def __init__(self,
                 fft_sz: int,
                 stft_win_sz: int,
                 clutter_latent_size: int,
                 target_latent_size: int,
                 n_ants: int,
                 activation: str = 'leaky',
                 **kwargs,
                 ) -> None:
        super(GeneratorModel, self).__init__()

        self.n_ants = n_ants
        self.fft_sz = fft_sz
        self.stft_win_sz = stft_win_sz
        self.hop = stft_win_sz // 4
        self.bin_bw = 52
        self.clutter_latent_size = clutter_latent_size
        self.target_latent_size = target_latent_size

        stack_output_sz = self.stft_win_sz // 4
        channel_sz = 16

        # Both the clutter and target stack standardize the output for any latent size
        self.clutter_stack = nn.Sequential(
            nn.Linear(clutter_latent_size, stack_output_sz),
            nn.LeakyReLU(),
        )

        self.target_stack = nn.Sequential(
            nn.Linear(target_latent_size, stack_output_sz),
            nn.LeakyReLU(),
        )

        self.comb_stack = nn.Sequential(
            nn.Linear(stack_output_sz * 2, stack_output_sz),
            nn.LeakyReLU(),
        )

        # LSTM layer to expand the network output to the correct size for pulse length
        self.prev_stft_stack = nn.Sequential(
            nn.LSTM(stack_output_sz, stack_output_sz, num_layers=1, batch_first=True),
        )

        # Output is Nb x Nchan x stft_win_sz x n_frames
        self.backbone = nn.Sequential(
            nn.Conv2d(1, channel_sz, stack_output_sz // 2 + 1, 1, stack_output_sz // 4),
            nn.LeakyReLU(),
            nn.Conv2d(channel_sz, channel_sz, 9, 1, 4),
            nn.LeakyReLU(),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channel_sz, channel_sz, kernel_size=(4, 3), stride=(2, 1), padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channel_sz, channel_sz, kernel_size=(4, 3), stride=(2, 1), padding=1),
            nn.LeakyReLU(),
            STFTAttention(channel_sz),
        )
        # self.backbone.apply(init_weights)

        self.deep_layers = nn.ModuleList()
        self.deep_layers.extend(
            nn.Sequential(
                nn.BatchNorm2d(channel_sz),
                nn.Conv2d(channel_sz, channel_sz, nl * 2 + 3, 1, nl + 1),
                nn.LeakyReLU(),
                nn.Conv2d(channel_sz, channel_sz, nl * 2 + 3, 1, nl + 1),
                nn.LeakyReLU(),
                nn.Conv2d(channel_sz, channel_sz, nl * 2 + 3, 1, nl + 1),
                nn.LeakyReLU(),
                nn.Conv2d(channel_sz, channel_sz, nl * 2 + 3, 1, nl + 1),
                nn.LeakyReLU(),
            )
            for nl in range(15, -1, -3)
        )
        for d in self.deep_layers:
            d.apply(init_weights)

        self.final = nn.Sequential(
            nn.BatchNorm2d(channel_sz),
            nn.Conv2d(channel_sz, n_ants * 2, 3, 1, 1),
        )
        self.final.apply(init_weights)

        self.stack_output_sz = stack_output_sz

    def forward(self, clutter: Tensor, target: Tensor, pulse_length: [int]) -> Tensor:
        # Use only the first pulse_length because it gives batch_size random numbers as part of the dataloader
        n_frames = 1 + (pulse_length[0] - self.stft_win_sz) // self.hop

        # Combine clutter and target features and synthesize combined ones
        ct_stack = torch.concat([self.clutter_stack(clutter), self.target_stack(target)])
        ct_stack = self.comb_stack(ct_stack.view(-1, self.stack_output_sz * 2))
        ct_stack = ct_stack.view(-1, 1, self.stack_output_sz)

        # Concatenate the number of STFT windows we'll need and pass them through an LSTM
        ct_stack = torch.concat([ct_stack for _ in range(n_frames)], dim=1)
        lstm_stack = self.prev_stft_stack(ct_stack)[0]
        x = self.backbone(lstm_stack.reshape(-1, 1, self.stack_output_sz, n_frames))
        # x0 = self.backbone(ct_stack)
        for d in self.deep_layers:
            x = torch.add(x, d(x))
        return self.final(x)

    def loss_function(self, *args, **kwargs) -> dict:
        # These values are set here purely for debugging purposes
        dev = self.device
        n_ants = self.n_ants

        # Initialize losses to zero and place on correct device
        sidelobe_loss = torch.tensor(0., device=dev)
        target_loss = torch.tensor(0., device=dev)
        ortho_loss = torch.tensor(0., device=dev)

        # Get clutter spectrum into complex form and normalize to unit energy
        clutter_spectrum = torch.complex(args[1][:, :, 0], args[1][:, :, 1])
        clutter_spectrum = clutter_spectrum / torch.sqrt(torch.sum(clutter_spectrum * torch.conj(clutter_spectrum),
                                                                   dim=1))[:, None]
        clutter_psd = clutter_spectrum * clutter_spectrum.conj()

        # Get target spectrum into complex form and normalize to unit energy
        target_spectrum = torch.complex(args[2][:, :, 0], args[2][:, :, 1])
        target_spectrum = target_spectrum / torch.sqrt(torch.sum(target_spectrum * torch.conj(target_spectrum),
                                                                 dim=1))[:, None]
        target_psd = target_spectrum * target_spectrum.conj()

        # This is the weights for a weighted average that emphasizes locations that have more
        # energy difference between clutter and target
        left_sig_tc = torch.abs(clutter_psd - target_psd)

        # Get waveform into complex form and normalize it to unit energy
        gen_waveform = self.getWaveform(nn_output=args[0])

        # Run losses for each channel
        for n in range(n_ants):
            g1 = gen_waveform[:, n, ...]
            sidelobe_func = 10 * torch.log(torch.abs(torch.fft.ifft(g1 * g1.conj(), dim=1)) / 10)

            # This is orthogonality losses, so we need a persistent value across the for loop
            if n > 0:
                ortho_loss += torch.sum(torch.abs(g1 * gn)) / gen_waveform.shape[0]

            # Power in the leftover signal for both clutter and target
            gen_psd = g1 * g1.conj()
            left_sig_c = torch.abs(gen_psd - clutter_psd)
            # left_sig_c[torch.abs(clutter_psd) < 1e-9] = 1.

            # The scaling here sets clutter and target losses to be between 0 and 1
            target_loss += torch.sum(torch.abs(left_sig_c - left_sig_tc)) / gen_waveform.shape[0] / 2.

            # Get the ISLR for this waveform
            sll = torch.mean(sidelobe_func[:, 7:-7], dim=1)
            sidelobe_loss += torch.sum(sidelobe_func[:, 0] / sll) / (self.n_ants * sidelobe_func.shape[0])
            gn = g1.conj()  # Conjugate of current g1 for orthogonality loss on next loop

        # Apply hinge loss to sidelobes
        sidelobe_loss = max(torch.tensor(0), 2 * sidelobe_loss - 1)
        ortho_loss = max(torch.tensor(0), 2 * ortho_loss - 1)

        # Use sidelobe and orthogonality as regularization params for target loss
        # loss = torch.sqrt(target_loss**2 + sidelobe_loss**2 + ortho_loss**2)
        loss = torch.sqrt(target_loss * (1 + sidelobe_loss + ortho_loss))

        return {'loss': loss, 'target_loss': target_loss,
                'sidelobe_loss': sidelobe_loss, 'ortho_loss': ortho_loss}

    def save(self, fpath, model_name='current'):
        torch.save(self.state_dict(), f'{fpath}/{model_name}_wave_model.state')
        with open(f'{fpath}/{model_name}_model_params.pic', 'wb') as f:
            pickle.dump({'fft_sz': self.fft_sz, 'stft_win_sz': self.stft_win_sz,
                         'clutter_latent_size': self.clutter_latent_size,
                         'target_latent_size': self.target_latent_size, 'n_ants': self.n_ants,
                         'state_file': f'{fpath}/{model_name}_wave_model.state'}, f)

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

    def getWaveform(self, cc: Tensor = None, tc: Tensor = None, pulse_length: int = 1, nn_output: Tensor = None,
                    use_window: bool = False, scale: bool = False) -> Tensor:
        """
        Given a clutter and target spectrum, produces a waveform FFT.
        :param scale: If True, scales the output FFT so that it is at least pulse_length long on IFFT.
        :param pulse_length: Length of pulse in samples.
        :param use_window: if True, applies a window to the finished waveform. Set to False for training.
        :param nn_output: Optional. If the waveform data is already created, use this to avoid putting in cc and tc.
        :param cc: Tensor of clutter spectrum. Same as input to model.
        :param tc: Tensor of target spectrum. Same as input to model.
        :return: Tensor of waveform FFTs, of size (batch_sz, n_ants, fft_sz).
        """
        n_ants = self.n_ants
        stft_win = self.stft_win_sz

        # Get the STFT either from the clutter, target, and pulse length or directly from the neural net
        full_stft = self.forward(cc, tc, pulse_length) if nn_output is None else nn_output
        if scale:
            new_fft_sz = int(2 ** (ceil(log2(pulse_length))))
            gen_waveform = torch.zeros((full_stft.shape[0], self.n_ants, new_fft_sz), dtype=torch.complex64,
                                       device=self.device)
        else:
            gen_waveform = torch.zeros((full_stft.shape[0], self.n_ants, self.fft_sz), dtype=torch.complex64,
                                       device=self.device)
        for n in range(n_ants):
            complex_stft = torch.complex(full_stft[:, n, :, :], full_stft[:, n + 1, :, :])

            # Apply a window if wanted for actual simulation
            if use_window:
                win_func = torch.zeros(self.stft_win_sz, device=self.device)
                win_func[:self.bin_bw // 2] = torch.windows.hann(self.bin_bw, device=self.device)[-self.bin_bw // 2:]
                win_func[-self.bin_bw // 2:] = torch.windows.hann(self.bin_bw, device=self.device)[:self.bin_bw // 2]
                g1 = torch.fft.fft(torch.istft(complex_stft, stft_win, hop_length=self.hop, window=win_func,
                                               return_complex=True, center=False), self.fft_sz, dim=-1)
            else:
                # This is for training purposes
                g1 = torch.fft.fft(torch.istft(complex_stft, stft_win, hop_length=self.hop,
                                               window=torch.ones(self.stft_win_sz, device=self.device),
                                               return_complex=True), self.fft_sz, dim=-1)
            g1 = g1 / torch.sqrt(torch.sum(g1 * torch.conj(g1), dim=1))[:, None]  # Unit energy calculation
            if scale:
                gen_waveform[:, n, :self.fft_sz // 2] = g1[:, :self.fft_sz // 2]
                gen_waveform[:, n, -self.fft_sz // 2:] = g1[:, -self.fft_sz // 2:]
            else:
                gen_waveform[:, n, ...] = g1
        return gen_waveform


class RCSModel(LightningModule):
    def __init__(self,
                 fft_sz: int,
                 n_ants: int,
                 ) -> None:
        super(RCSModel, self).__init__()

        self.fft_sz = fft_sz
        self.n_ants = n_ants

        self.optical_stack = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 4, 2, 1),
            nn.LeakyReLU(),
        )

        self.sar_stack = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 4, 2, 1),
            nn.LeakyReLU(),
        )

        self.pose_stack = nn.Sequential(
            nn.Linear(7, 9),
            nn.LeakyReLU(),
        )

        self.comb_stack = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

        self.loss = nn.MSELoss()

    def forward(self, opt_data: Tensor, sar_data: Tensor, pose_data: Tensor) -> Tensor:
        x = torch.stack([self.optical_stack(opt_data), self.sar_stack(sar_data)], dim=1)
        x = torch.convolution(x, self.pose_stack(pose_data).view(-1, 1, 3, 3), bias=None, stride=[1, 1], padding=[1, 1],
                              output_padding=[0, 0], transposed=False, dilation=[1, 1], groups=1)
        return self.comb_stack(x)

    def loss_function(self, *args, **kwargs) -> dict:
        return {'loss': self.loss(args[0], args[1])}
