import contextlib
import pickle
from typing import Optional, Union, Tuple, Dict

import torch
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from torch.nn import functional as nn_func
from numpy import log2, ceil
from torchvision import transforms
from layers import LSTMAttention, AttentionConv


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
        channel_sz = 32

        # Both the clutter and target stack standardize the output for any latent size
        self.clutter_stack = nn.Sequential(
            nn.Linear(clutter_latent_size, stack_output_sz),
            nn.LeakyReLU(),
        )

        self.target_stack = nn.Sequential(
            nn.Linear(target_latent_size, stack_output_sz),
            nn.LeakyReLU(),
        )

        self.init_layer = nn.Sequential(
            nn.Conv2d(1, channel_sz, 1, 1, 0),
            nn.LeakyReLU(),
        )

        # Output is Nb x Nchan x stack_output_sz x n_frames
        self.backbone = nn.ModuleList()
        self.backbone.extend(
            nn.Sequential(
                nn.Conv2d(channel_sz, channel_sz, nl * 2 + 3, 1, nl + 1),
                nn.Tanh(),
                AttentionConv(channel_sz, channel_sz, 3, 1, 1, channel_sz // 4),
                nn.Tanh(),
                nn.Conv2d(channel_sz, channel_sz, nl * 2 + 3, 1, nl + 1),
                nn.Tanh(),
            )
            for nl in range(9, 1, -2)
        )
        for d in self.backbone:
            d.apply(init_weights)

        self.deep_layers = nn.ModuleList()
        self.deep_layers.extend(
            nn.Sequential(
                nn.BatchNorm2d(channel_sz),
                nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
                nn.Tanh(),
                nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
                nn.Tanh(),
                nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
                nn.Tanh(),
            )
            for _ in self.backbone
        )
        for d in self.deep_layers:
            d.apply(init_weights)

        # LSTM layers to expand the network output to the correct size for pulse length
        self.lstm_layers = nn.LSTM(stack_output_sz, stack_output_sz, num_layers=3, batch_first=True)

        self.rnn_layers = nn.LSTM(stack_output_sz * 2, stack_output_sz, num_layers=3, batch_first=True)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(channel_sz, channel_sz, kernel_size=(4, 3), stride=(2, 1), padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(channel_sz, channel_sz, kernel_size=(4, 3), stride=(2, 1), padding=1),
            nn.Tanh(),
            nn.Conv2d(channel_sz, n_ants * 2, 1, 1, 0),
        )
        self.final.apply(init_weights)

        self.attention = LSTMAttention()

        self.stack_output_sz = stack_output_sz

        mask_select = (stft_win_sz - self.bin_bw) // 2
        self.bandwidth_mask = torch.zeros((1, 1, stft_win_sz, 1), dtype=torch.bool)
        self.bandwidth_mask[0, 0, mask_select:-mask_select, 0] = 1.

        self.example_input_array = (torch.zeros((1, clutter_latent_size)), torch.zeros((1, target_latent_size)),
                                    torch.tensor([1250]))

    def forward(self, clutter: torch.tensor, target: torch.tensor, pulse_length: [int]) -> torch.tensor:
        # Use only the first pulse_length because it gives batch_size random numbers as part of the dataloader
        n_frames = 1 + (pulse_length[0] - self.stft_win_sz) // self.hop

        # Get clutter and target features, for LSTM input
        x = self.clutter_stack(clutter).unsqueeze(1)
        t_stack = self.target_stack(target).unsqueeze(1)

        # No-attention LSTM for attention input
        lstm_x, (hidden, cell) = self.lstm_layers(x)
        ctxt, attn_weights = self.attention(lstm_x, hidden[-1])

        # LSTM with attention from previous LSTM
        x = self.rnn_layers(torch.concat([t_stack, ctxt.unsqueeze(1)], dim=2))[0]

        # Pass them through the LSTMs, one at a time, building up the number of frames we need for a pulse
        for _ in range(n_frames - 1):
            lstm_x, (hidden, cell) = self.lstm_layers(x)
            ctxt, attn_weights = self.attention(lstm_x, hidden[-1])
            x = torch.concat((x, self.rnn_layers(
                torch.concat([t_stack, ctxt.unsqueeze(1)], dim=2))[0]), dim=1)
        x = x.reshape(-1, 1, self.stack_output_sz, n_frames)

        # Init layer to get the right number of channels
        x = self.init_layer(x)

        # Get feedforward connections from backbone
        outputs = []
        for b in self.backbone:
            x = b(x)
            outputs.append(x)

        # Reverse outputs for correct structure
        # outputs.reverse()

        # Combine with deep layers
        for op, d in zip(outputs, self.deep_layers):
            x = d(x) + op
        return torch.masked_select(self.final(x), self.bandwidth_mask.to(self.device))

    def loss_function(self, *args, **kwargs) -> dict:
        # These values are set here purely for debugging purposes
        dev = self.device
        n_ants = self.n_ants

        # Initialize losses to zero and place on correct device
        sidelobe_loss = torch.tensor(0., device=dev, requires_grad=False)
        target_loss = torch.tensor(0., device=dev)
        # ortho_loss = torch.tensor(0., device=dev)

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
            slf = torch.abs(torch.fft.ifft(g1 * g1.conj(), dim=1))
            slf[slf == 0] = 1e-9
            sidelobe_func = 10 * torch.log(slf / 10)
            sll = nn_func.max_pool1d_with_indices(sidelobe_func, 65, 1,
                                                  padding=32)[0].unique(dim=1).detach()[:, 1]

            # This is orthogonality losses, so we need a persistent value across the for loop
            # if n > 0:
            #     ortho_loss += torch.sum(torch.abs(g1 * gn)) / gen_waveform.shape[0]

            # Power in the leftover signal for both clutter and target
            gen_psd = g1 * g1.conj()
            left_sig_c = torch.abs(gen_psd - clutter_psd)

            # The scaling here sets clutter and target losses to be between 0 and 1
            target_loss += torch.sum(torch.abs(left_sig_c - left_sig_tc)) / gen_waveform.shape[0] / 2.

            # Get the ISLR for this waveform
            sidelobe_loss += torch.sum(sidelobe_func[:, 0] / sll) / (self.n_ants * sidelobe_func.shape[0])
            # gn = g1.conj()  # Conjugate of current g1 for orthogonality loss on next loop

        # Apply hinge loss to sidelobes
        sidelobe_loss = (sidelobe_loss - .1)**2

        # Use sidelobe and orthogonality as regularization params for target loss
        # loss = torch.sqrt(target_loss**2 + sidelobe_loss**2 + ortho_loss**2)
        loss = torch.sqrt(torch.abs(target_loss * (1. + sidelobe_loss)))

        return {'loss': loss, 'target_loss': target_loss,
                'sidelobe_loss': sidelobe_loss}

    def save(self, fpath, model_name='current'):
        torch.save(self.state_dict(), f'{fpath}/{model_name}_wave_model.state')
        with open(f'{fpath}/{model_name}_model_params.pic', 'wb') as f:
            pickle.dump({'fft_sz': self.fft_sz, 'stft_win_sz': self.stft_win_sz,
                         'clutter_latent_size': self.clutter_latent_size,
                         'target_latent_size': self.target_latent_size, 'n_ants': self.n_ants,
                         'state_file': f'{fpath}/{model_name}_wave_model.state'}, f)

    def getWindow(self, bin_bw):
        win_func = torch.zeros(self.stft_win_sz, device=self.device)
        win_func[:bin_bw // 2] = torch.windows.hann(bin_bw, device=self.device)[-bin_bw // 2:]
        win_func[-bin_bw // 2:] = torch.windows.hann(bin_bw, device=self.device)[:bin_bw // 2]
        return win_func

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

    def example_input_array(self) -> Optional[Union[Tensor, Tuple, Dict]]:
        return self.example_input_array

    def getWaveform(self, cc: Tensor = None, tc: Tensor = None, pulse_length: [int, list] = 1, nn_output: Tensor = None,
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
                                       device=self.device, requires_grad=False)
        else:
            gen_waveform = torch.zeros((full_stft.shape[0], self.n_ants, self.fft_sz), dtype=torch.complex64,
                                       device=self.device, requires_grad=False)
        for n in range(n_ants):
            complex_output = torch.complex(full_stft[:, n, :, :], full_stft[:, n + 1, :, :])
            complex_stft = torch.zeros((full_stft.shape[0], self.stft_win_sz, full_stft.shape[3]),
                                       dtype=torch.complex64)
            complex_stft[:, :full_stft.shape[2] // 2, :] = complex_output[:, -full_stft.shape[2] // 2:, :]
            complex_stft[:, -full_stft.shape[2] // 2:, :] = complex_output[:, :full_stft.shape[2] // 2, :]

            # Apply a window if wanted for actual simulation
            if use_window:
                g1 = torch.fft.fft(torch.istft(complex_stft * self.getWindow(self.bin_bw)[None, :, None],
                                               stft_win, hop_length=self.hop,
                                               window=torch.ones(self.stft_win_sz, device=self.device),
                                               return_complex=True, center=False), self.fft_sz, dim=-1)
            else:
                # This is for training purposes
                g1 = torch.fft.fft(torch.istft(complex_stft, stft_win, hop_length=self.hop,
                                               window=torch.ones(self.stft_win_sz, device=self.device),
                                               return_complex=True), self.fft_sz, dim=-1)
            if scale:
                g1 = torch.fft.fft(g1, new_fft_sz, dim=-1)
            else:
                g1 = torch.fft.fft(g1, self.fft_sz, dim=-1)
            g1 = g1 / torch.sqrt(torch.sum(g1 * torch.conj(g1), dim=1))[:, None]  # Unit energy calculation
            gen_waveform[:, n, ...] = g1
        return gen_waveform


class RCSModel(LightningModule):
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
