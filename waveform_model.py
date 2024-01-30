from typing import List, Any, TypeVar
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from numpy import log2, ceil
from torchvision import transforms
from layers import RichConvTranspose2d, RichConv2d, Linear2d, SelfAttention

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
                 ) -> None:
        super(GeneratorModel, self).__init__()

        self.n_ants = n_ants
        self.fft_sz = fft_sz
        self.stft_win_sz = stft_win_sz
        self.hop = stft_win_sz // 4
        self.bin_bw = 52

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

        self.comb_stack = nn.Sequential(
            nn.Linear(stack_output_sz * 2, stack_output_sz),
            nn.LeakyReLU(),
        )

        # LSTM layer to expand the network output to the correct size for pulse length
        self.prev_stft_stack = nn.Sequential(
            nn.LSTM(stack_output_sz, stack_output_sz, num_layers=4, batch_first=True),
        )

        self.attention_lstm = nn.LSTM(stack_output_sz, stack_output_sz, batch_first=True)

        self.attention_stack = nn.Sequential(
            nn.Conv2d(1, channel_sz, stft_win_sz // 2 + 1, 1, stft_win_sz // 4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channel_sz, channel_sz, kernel_size=(4, 3), stride=(2, 1), padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channel_sz, channel_sz, kernel_size=(4, 3), stride=(2, 1), padding=1),
            nn.Sigmoid(),
        )

        # Output is Nb x Nchan x stft_win_sz x n_frames
        self.backbone = nn.Sequential(
            nn.Conv2d(1, channel_sz, stft_win_sz // 2 + 1, 1, stft_win_sz // 4),
            nn.LeakyReLU(),
            nn.Conv2d(channel_sz, channel_sz, 7, 1, 3),
            nn.LeakyReLU(),
            nn.Conv2d(channel_sz, channel_sz, 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channel_sz, channel_sz, kernel_size=(4, 3), stride=(2, 1), padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channel_sz, channel_sz, kernel_size=(4, 3), stride=(2, 1), padding=1),
            nn.LeakyReLU(),
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

    def forward(self, clutter: Tensor, target: Tensor, pulse_length: int) -> Tensor:
        # Use only the first pulse_length because it gives batch_size random numbers as part of the dataloader
        n_frames = 1 + (pulse_length[0] - self.stft_win_sz) // self.hop
        ct_stack = torch.concat([self.clutter_stack(clutter), self.target_stack(target)])
        ct_stack = self.comb_stack(ct_stack.view(-1, self.stack_output_sz * 2))
        ct_stack = torch.concat([ct_stack.view(-1, 1, self.stack_output_sz) for _ in range(n_frames)], dim=1)
        lstm_stack = self.prev_stft_stack(ct_stack)[0]
        att_stack = self.attention_lstm(ct_stack)[0]
        att_stack = self.attention_stack(att_stack.reshape(-1, 1, self.stack_output_sz, n_frames))
        x = self.backbone(lstm_stack.reshape(-1, 1, self.stack_output_sz, n_frames)) * att_stack
        # x0 = self.backbone(ct_stack)
        for d in self.deep_layers:
            x = torch.add(x, d(x))
        return self.final(x)

    def getWindow(self, bandwidth: float, fs: float, n_frames: int):
        bwidth = int(bandwidth // (fs / self.stft_win_sz))
        bwidth += 1 if bwidth % 2 != 0 else 0
        bwin = torch.ones((bwidth,), device=self.device)
        win = torch.zeros((1, 1, self.stft_win_sz, 1), device=self.device)
        win[0, 0, :bwidth // 2, 0] = bwin[-bwidth // 2:]
        win[0, 0, -bwidth // 2:, 0] = bwin[:bwidth // 2]
        return torch.tile(win, (1, self.n_ants * 2, 1, n_frames))

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
            sidelobe_func = torch.abs(torch.fft.ifft(g1 * g1.conj(), dim=1))

            # This is orthogonality losses, so we need a persistent value across the for loop
            if n > 0:
                ortho_loss += torch.sum(torch.abs(g1 * gn)) / gen_waveform.shape[0]

            # Power in the leftover signal for both clutter and target
            gen_psd = g1 * g1.conj()
            left_sig_c = torch.abs(gen_psd - clutter_psd)
            # left_sig_c[torch.abs(clutter_psd) < 1e-9] = 1.

            # The scaling here sets clutter and target losses to be between 0 and 1
            target_loss += torch.sum(torch.abs(left_sig_c - left_sig_tc)) / gen_waveform.shape[0] / 2.
            sidelobe_loss += (
                    torch.sum(10 ** (torch.log(torch.mean(sidelobe_func, dim=1) / sidelobe_func[:, 0]) / 10)) /
                    gen_waveform.shape[0])
            gn = g1.conj()  # Conjugate of current g1 for orthogonality loss on next loop

        # Apply hinge loss to sidelobes
        sidelobe_loss = max(torch.tensor(0), 2 * sidelobe_loss - 1)
        ortho_loss = max(torch.tensor(0), 2 * ortho_loss - 1)
        loss = torch.sqrt(target_loss ** 2 + sidelobe_loss ** 2 + ortho_loss ** 2)

        return {'loss': loss, 'target_loss': target_loss,
                'sidelobe_loss': sidelobe_loss, 'ortho_loss': ortho_loss}

    def getWaveform(self, cc: Tensor = None, tc: Tensor = None, pulse_length: int = 1, nn_output: Tensor = None,
                    use_window: bool = False) -> Tensor:
        """
        Given a clutter and target spectrum, produces a waveform FFT.
        :param pulse_length: Length of pulse in samples.
        :param use_window: if True, applies a window to the finished waveform. Set to False for training.
        :param nn_output: Optional. If the waveform data is already created, use this to avoid putting in cc and tc.
        :param cc: Tensor of clutter spectrum. Same as input to model.
        :param tc: Tensor of target spectrum. Same as input to model.
        :return: Tensor of waveform FFTs, of size (batch_sz, n_ants, fft_sz).
        """
        n_ants = self.n_ants
        stft_win = self.stft_win_sz
        full_stft = self.forward(cc, tc, pulse_length) if nn_output is None else nn_output
        gen_waveform = torch.zeros((full_stft.shape[0], self.n_ants, self.fft_sz), dtype=torch.complex64,
                                   device=self.device)
        for n in range(n_ants):
            complex_stft = torch.complex(full_stft[:, n, :, :], full_stft[:, n + 1, :, :])
            if use_window:
                win_func = torch.zeros(self.stft_win_sz, device=self.device)
                win_func[:self.bin_bw // 2] = torch.windows.hann(self.bin_bw, device=self.device)[-self.bin_bw // 2:]
                win_func[-self.bin_bw // 2:] = torch.windows.hann(self.bin_bw, device=self.device)[:self.bin_bw // 2]
                g1 = torch.fft.fft(torch.istft(complex_stft, stft_win, hop_length=self.hop, window=win_func,
                                               return_complex=True, center=False), self.fft_sz, dim=-1)
            else:
                g1 = torch.fft.fft(torch.istft(complex_stft, stft_win, hop_length=self.hop,
                                               return_complex=True), self.fft_sz, dim=-1)
            g1 = g1 / torch.sqrt(torch.sum(g1 * torch.conj(g1), dim=1))[:, None]  # Unit energy calculation
            gen_waveform[:, n, ...] = g1
        return gen_waveform


class WindowModel(LightningModule):
    def __init__(self,
                 fft_sz: int,
                 n_ants: int,
                 ) -> None:
        super(WindowModel, self).__init__()

        self.fft_sz = fft_sz
        self.n_ants = n_ants

        self.init_stack = nn.Sequential(
            nn.Linear(4, 128),
            nn.LeakyReLU(),
            nn.Linear(128, fft_sz),
            nn.LeakyReLU(),
        )

        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 65, 1, 32),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 33, 1, 16),
            nn.LeakyReLU(),
            nn.Conv1d(64, 1, 5, 1, 2),
            nn.Sigmoid(),
        )

        self.loss = nn.MSELoss()

    def forward(self, windata: Tensor) -> Tensor:
        x = self.init_stack(windata)
        return self.conv_stack(x.view(-1, 1, self.fft_sz))

    def loss_function(self, *args, **kwargs) -> dict:
        return {'loss': self.loss(args[0], args[1])}
