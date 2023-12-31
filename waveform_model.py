from typing import List, Any, TypeVar
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from numpy import log2, ceil

from layers import RichConvTranspose2d, RichConv2d, Linear2d

Tensor = TypeVar('torch.tensor')


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


class GeneratorModel(LightningModule):
    def __init__(self,
                 bin_bw: int,
                 stft_params: tuple,
                 stft_win_sz: int,
                 clutter_latent_size: int,
                 target_latent_size: int,
                 n_ants: int,
                 activation: str = 'leaky',
                 ) -> None:
        super(GeneratorModel, self).__init__()

        self.n_ants = n_ants
        self.bin_bw = bin_bw
        self.fft_sz = int(2 ** (ceil(log2(bin_bw / (stft_params[0] / stft_win_sz)))))
        self.stft_win = stft_win_sz
        self.stft_bw = stft_params[0]
        self.stft_sz = stft_params[1]

        stack_output_sz = 256
        self.batch_norm = nn.BatchNorm1d(1)

        self.clutter_stack = nn.Sequential(
            nn.Linear(clutter_latent_size, stack_output_sz),
            nn.LeakyReLU(),
        )

        self.target_stack = nn.Sequential(
            nn.Linear(target_latent_size, stack_output_sz),
            nn.LeakyReLU(),
        )

        self.ffinit = nn.Sequential(
            Linear2d(16, 16, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2, 512, 19, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(512, 128, 13, 1, 0),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 7, 1, 3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(128, n_ants * 2, 3, 1, 1),
            Linear2d(52, 52, n_ants * 2)
        )

        self.stack_output_sz = stack_output_sz

    def forward(self, clutter: Tensor, target: Tensor) -> Tensor:
        # add_stack = self.batch_norm(torch.add(self.clutter_stack(clutter), self.target_stack(target)).view(
        #     -1, 1, self.stack_output_sz))
        ct_stack = torch.concat([self.clutter_stack(clutter), self.target_stack(target)])
        ct_stack = ct_stack.view(-1, 2, 16, 16)
        return self.ffinit(ct_stack)

    def loss_function(self, *args, **kwargs) -> dict:
        # These values are set here purely for debugging purposes
        dev = self.device
        n_ants = self.n_ants
        bin_bw = self.bin_bw

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
        gen_waveform = self.getWaveform(nn_output = args[0])

        # Run losses for each channel
        for n in range(n_ants):
            g1 = gen_waveform[:, n, ...]
            sidelobe_func = torch.abs(torch.fft.ifft(g1 * g1.conj(), dim=1))

            g1 = torch.fft.fftshift(g1, dim=1)[:, self.fft_sz // 2 - bin_bw // 2: self.fft_sz // 2 + bin_bw // 2]

            # This is orthogonality losses, so we need a persistent value across the for loop
            if n > 0:
                ortho_loss += torch.sum(torch.abs(g1 * gn)) / gen_waveform.shape[0]

            # Power in the leftover signal for both clutter and target
            gen_psd = g1 * g1.conj()
            left_sig_c = torch.abs(gen_psd - clutter_psd)

            # The scaling here sets clutter and target losses to be between 0 and 1
            target_loss += torch.sum(torch.abs(left_sig_c - left_sig_tc)) / gen_waveform.shape[0] / 2.
            sidelobe_loss += (
                        torch.sum(10 ** (torch.log(torch.mean(sidelobe_func, dim=1) / sidelobe_func[:, 0]) / 10)) /
                        gen_waveform.shape[0])
            gn = g1.conj()  # Conjugate of current g1 for orthogonality loss on next loop

        # Apply hinge loss to sidelobes
        sidelobe_loss = max(torch.tensor(0), 2 * sidelobe_loss - 1)
        ortho_loss = max(torch.tensor(0), 2 * ortho_loss - 1)
        loss = torch.sqrt(target_loss**2 + sidelobe_loss**2 + ortho_loss**2)

        return {'loss': loss, 'target_loss': target_loss,
                'sidelobe_loss': sidelobe_loss, 'ortho_loss': ortho_loss}

    def getWaveform(self, cc: Tensor = None, tc: Tensor = None, nn_output: Tensor = None) -> Tensor:
        """
        Given a clutter and target spectrum, produces a waveform FFT.
        :param nn_output:
        :param cc: Tensor of clutter spectrum. Same as input to model.
        :param tc: Tensor of target spectrum. Same as input to model.
        :return: Tensor of waveform FFTs, of size (batch_sz, n_ants, fft_sz).
        """
        n_ants = self.n_ants
        stft_half_bw = self.stft_bw // 2
        stft_sz = self.stft_sz
        stft_win = self.stft_win
        net_out = self.forward(cc, tc) if nn_output is None else nn_output
        win_func = torch.zeros(self.fft_sz, device=self.device)
        win_func[:self.bin_bw // 2] = torch.windows.hann(self.bin_bw, device=self.device)[-self.bin_bw // 2:]
        win_func[-self.bin_bw // 2:] = torch.windows.hann(self.bin_bw, device=self.device)[:self.bin_bw // 2]
        full_stft = torch.zeros((net_out.shape[0], n_ants, stft_win, stft_sz),
                                dtype=torch.complex64, device=self.device)
        gen_waveform = torch.zeros((net_out.shape[0], n_ants, self.fft_sz), dtype=torch.complex64, device=self.device)
        full_stft[:, :, :stft_half_bw, :] = (
            torch.complex(net_out[:, ::2, :stft_half_bw], net_out[:, 1::2, :stft_half_bw]))
        full_stft[:, :, -stft_half_bw:, :] = (
            torch.complex(net_out[:, ::2, -stft_half_bw:], net_out[:, 1::2, -stft_half_bw:]))
        for n in range(n_ants):
            g1 = torch.fft.fft(torch.istft(full_stft[:, n, :, :], stft_win, return_complex=True), self.fft_sz, dim=-1)
            g1 *= win_func[None, :]
            g1 = g1 / torch.sqrt(torch.sum(g1 * torch.conj(g1), dim=1))[:, None]  # Unit energy calculation
            gen_waveform[:, n, ...] = g1
        return gen_waveform
