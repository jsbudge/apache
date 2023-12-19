from typing import List, Any, TypeVar
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from numpy import log2, ceil

from layers import RichConvTranspose2d, RichConv2d

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
        self.fft_sz = int(2 ** (ceil(log2(bin_bw / (stft_params[0] / stft_params[1])))))
        self.stft_win = stft_win_sz
        self.stft_bw = stft_params[0]
        self.stft_sz = stft_params[1]

        stack_output_sz = 64
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
            RichConvTranspose2d(2, 128, 8, activation=activation),
            RichConvTranspose2d(128, 128, 16, activation=activation),
            RichConvTranspose2d(128, 64, 32, activation=activation),
            RichConv2d(64, 64, 64, activation=activation, maintain_shape=True),
            nn.Conv2d(64, 32, 13, 1, 0),
            nn.LeakyReLU(),
            nn.Conv2d(32, n_ants * 2, 3, 1, 1),
        )

        self.stack_output_sz = stack_output_sz

    def forward(self, clutter: Tensor, target: Tensor) -> Tensor:
        # add_stack = self.batch_norm(torch.add(self.clutter_stack(clutter), self.target_stack(target)).view(
        #     -1, 1, self.stack_output_sz))
        ct_stack = torch.concat([self.clutter_stack(clutter), self.target_stack(target)])
        ct_stack = ct_stack.view(-1, 2, 8, 8)
        return self.ffinit(ct_stack)

    def loss_function(self, *args, **kwargs) -> dict:
        # These values are set here purely for debugging purposes
        dev = self.device
        n_ants = self.n_ants
        stft_half_bw = self.stft_bw // 2
        stft_sz = self.stft_sz
        stft_win = self.stft_win
        bin_bw = self.bin_bw

        # Initialize losses to zero and place on correct device
        sidelobe_loss = torch.tensor(0., device=dev)
        clutter_loss = torch.tensor(0., device=dev)
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
        weighting = torch.abs(clutter_psd - target_psd)
        weighting = weighting / torch.sum(weighting) * target_psd.shape[1]

        # Get waveform into complex form and normalize it to unit energy
        gen_waveform = torch.zeros((args[0].shape[0], n_ants, stft_win, stft_sz), device=dev,
                                   dtype=torch.complex64)
        gen_waveform[:, :, :stft_half_bw, :] = (
            torch.complex(args[0][:, ::2, :stft_half_bw], args[0][:, 1::2, :stft_half_bw]))
        gen_waveform[:, :, -stft_half_bw:, :] = (
            torch.complex(args[0][:, ::2, -stft_half_bw:], args[0][:, 1::2, -stft_half_bw:]))

        # Run losses for each channel
        for n in range(n_ants):
            g1 = torch.fft.fft(torch.istft(gen_waveform[:, n, :, :], stft_win), self.fft_sz)
            g1 = g1 / torch.sqrt(torch.sum(g1 * torch.conj(g1), dim=1))[:, None]
            sidelobe_func = torch.abs(torch.fft.ifft(g1 * g1.conj(), dim=1))

            g1 = torch.fft.fftshift(g1, dim=1)[:, self.fft_sz // 2 - bin_bw // 2: self.fft_sz // 2 + bin_bw // 2]

            # This is orthogonality losses, so we need a persistent value across the for loop
            if n > 0:
                ortho_loss += torch.sum(torch.abs(g1 * gn)) / gen_waveform.shape[0]

            # Power in the leftover signal for both clutter and target
            gen_psd = g1 * g1.conj()
            left_sig_c = torch.abs(gen_psd - clutter_psd)
            left_sig_t = torch.abs(gen_psd - target_psd)

            # The scaling here sets clutter and target losses to be between 0 and 1
            clutter_loss += (1. - torch.abs(torch.sum(torch.sum(
                left_sig_c * weighting, dim=1))) / gen_waveform.shape[0]) / (
                                       2.)
            target_loss += (torch.abs(torch.sum(torch.sum(left_sig_t * weighting,
                                                         dim=1))) / gen_waveform.shape[0]) / 2.
            sidelobe_loss += (
                        torch.sum(10 ** (torch.log(torch.mean(sidelobe_func, dim=1) / sidelobe_func[:, 0]) / 10)) /
                        gen_waveform.shape[0])
            gn = g1.conj()  # Conjugate of current g1 for orthogonality loss on next loop

        loss = torch.sqrt((clutter_loss / 4) ** 2 + (2 * target_loss) ** 2 + sidelobe_loss**2 + ortho_loss**2)

        return {'loss': loss, 'clutter_loss': clutter_loss, 'target_loss': target_loss,
                'sidelobe_loss': sidelobe_loss, 'ortho_loss': ortho_loss}

    def getWaveform(self, cs: Tensor, ts: Tensor) -> Tensor:
        """
        Given a clutter and target spectrum, produces a waveform FFT.
        :param cs: Tensor of clutter spectrum. Same as input to model.
        :param ts: Tensor of target spectrum. Same as input to model.
        :return: Tensor of waveform FFTs, of size (batch_sz, n_ants, fft_sz).
        """
        n_ants = self.n_ants
        stft_half_bw = self.stft_bw // 2
        stft_sz = self.stft_sz
        stft_win = self.stft_win
        net_out = self.forward(cs, ts)
        full_stft = torch.zeros((net_out.shape[0], n_ants, stft_win, stft_sz), dtype=torch.complex64)
        gen_waveform = torch.zeros((net_out.shape[0], n_ants, self.fft_sz), dtype=torch.complex64)
        full_stft[:, :, :stft_half_bw, :] = (
            torch.complex(net_out[:, ::2, :stft_half_bw], net_out[:, 1::2, :stft_half_bw]))
        full_stft[:, :, -stft_half_bw:, :] = (
            torch.complex(net_out[:, ::2, -stft_half_bw:], net_out[:, 1::2, -stft_half_bw:]))
        for n in range(n_ants):
            g1 = torch.fft.fft(torch.istft(full_stft[:, n, :, :], stft_win), self.fft_sz)
            g1 = g1 / torch.sqrt(torch.sum(g1 * torch.conj(g1), dim=1))[:, None]  # Unit energy calculation
            gen_waveform[:, n, ...] = g1
        return gen_waveform
