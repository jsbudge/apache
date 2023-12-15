from typing import List, Any, TypeVar
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F

Tensor = TypeVar('torch.tensor')


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


class RichConv1d(LightningModule):
    def __init__(self, in_channels, out_channels, out_layer_sz, activation='leaky'):
        super(RichConv1d, self).__init__()
        self.branch0 = nn.Conv1d(in_channels, out_channels, 4, 2, 1)
        self.branch1 = nn.Conv1d(in_channels, out_channels, out_layer_sz + 3, 1, 1)
        self.branch2 = nn.Conv1d(in_channels, out_channels, out_layer_sz + 1, 1, 0)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        if self.activation == 'elu':
            x = torch.add(torch.add(F.elu(self.branch0(x)), F.elu(self.branch1(x))),
                          F.elu(self.branch2(x)))
        elif self.activation == 'tanh':
            x = torch.add(torch.add(F.tanh(self.branch0(x)), F.tanh(self.branch1(x))),
                          F.tanh(self.branch2(x)))
        else:
            x = torch.add(torch.add(F.leaky_relu(self.branch0(x)), F.leaky_relu(self.branch1(x))),
                          F.leaky_relu(self.branch2(x)))
        return self.batch_norm(x)


class RichConvTranspose1d(LightningModule):
    def __init__(self, in_channels, out_channels, in_layer_sz, activation='leaky'):
        super(RichConvTranspose1d, self).__init__()
        self.branch0 = nn.ConvTranspose1d(in_channels, out_channels, 4, 2, 1)
        self.branch1 = nn.ConvTranspose1d(in_channels, out_channels, in_layer_sz + 3, 1, 1)
        self.branch2 = nn.ConvTranspose1d(in_channels, out_channels, in_layer_sz + 1, 1, 0)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        if self.activation == 'elu':
            x = torch.add(torch.add(F.elu(self.branch0(x)), F.elu(self.branch1(x))),
                          F.elu(self.branch2(x)))
        elif self.activation == 'tanh':
            x = torch.add(torch.add(F.tanh(self.branch0(x)), F.tanh(self.branch1(x))),
                          F.tanh(self.branch2(x)))
        else:
            x = torch.add(torch.add(F.leaky_relu(self.branch0(x)), F.leaky_relu(self.branch1(x))),
                          F.leaky_relu(self.branch2(x)))
        return self.batch_norm(x)


class RichConvTranspose2d(LightningModule):
    def __init__(self, in_channels, out_channels, layer_sz, activation='leaky'):
        super(RichConvTranspose2d, self).__init__()
        self.branch0 = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.branch1 = nn.ConvTranspose2d(in_channels, out_channels, layer_sz + 3, 1, 1)
        self.branch2 = nn.ConvTranspose2d(in_channels, out_channels, layer_sz + 1, 1, 0)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        if self.activation == 'elu':
            x = torch.add(torch.add(F.elu(self.branch0(x)), F.elu(self.branch1(x))),
                          F.elu(self.branch2(x)))
        elif self.activation == 'tanh':
            x = torch.add(torch.add(F.tanh(self.branch0(x)), F.tanh(self.branch1(x))),
                          F.tanh(self.branch2(x)))
        else:
            x = torch.add(torch.add(F.leaky_relu(self.branch0(x)), F.leaky_relu(self.branch1(x))),
                          F.leaky_relu(self.branch2(x)))
        return self.batch_norm(x)


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
        self.stft_win = stft_win_sz
        self.stft_bw = stft_params[0]
        self.stft_sz = stft_params[1]

        stack_output_sz = 169
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
            RichConvTranspose2d(3, 64, 13, activation=activation),
            RichConvTranspose2d(64, 16, 26, activation=activation),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.Conv2d(8, n_ants * 2, 3, 1, 1),
        )

        self.stack_output_sz = stack_output_sz

    def forward(self, clutter: Tensor, target: Tensor) -> Tensor:
        add_stack = self.batch_norm(torch.add(self.clutter_stack(clutter), self.target_stack(target)).view(
            -1, 1, self.stack_output_sz))
        ct_stack = torch.concat([add_stack.squeeze(1), self.clutter_stack(clutter), self.target_stack(target)])
        ct_stack = ct_stack.view(-1, 3, 13, 13)
        return self.ffinit(ct_stack)

    def loss_function(self, *args, **kwargs) -> dict:
        dev = self.device
        n_ants = self.n_ants
        stft_half_bw = self.stft_bw // 2
        stft_sz = self.stft_sz
        stft_win = self.stft_win
        bin_bw = self.bin_bw
        sidelobe_loss = clutter_loss = target_loss = torch.tensor(0., device=dev)
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
        for n in range(n_ants):
            g1 = torch.fft.fft(torch.istft(gen_waveform[:, n, :, :], stft_win), 32768)
            g1 = g1 / torch.sqrt(torch.sum(g1 * torch.conj(g1), dim=1))[:, None]
            sidelobe_func = torch.abs(torch.fft.ifft(g1 * g1.conj(), dim=1))

            g1 = torch.fft.fftshift(g1[:, 32768 // 2 - bin_bw // 2: 32768 // 2 + bin_bw // 2], dim=1)

            gen_psd = g1 * g1.conj()

            left_sig_c = torch.abs(gen_psd - clutter_psd)
            left_sig_t = torch.abs(gen_psd - target_psd)

            # The scaling here sets clutter and target losses to be between 0 and 1
            clutter_loss += (2. - torch.abs(torch.sum(torch.sum(
                left_sig_c * weighting, dim=1))) / gen_waveform.shape[0]) / (
                                       2.)
            target_loss += (torch.abs(torch.sum(torch.sum(left_sig_t * weighting,
                                                         dim=1))) / gen_waveform.shape[0]) / 2.
            sidelobe_loss += (
                        torch.sum(10 ** (torch.log(torch.mean(sidelobe_func, dim=1) / sidelobe_func[:, 0]) / 10)) /
                        gen_waveform.shape[0])
        # ortho_loss = torch.sum(torch.abs(
        #     torch.sum(gen_waveform[:, 0, :] * torch.conj(gen_waveform[:, 1, :]), dim=1))) / gen_waveform.shape[0]

        loss = torch.sqrt((clutter_loss / 4) ** 2 + (2 * target_loss) ** 2 + sidelobe_loss**2)

        return {'loss': loss, 'clutter_loss': clutter_loss, 'target_loss': target_loss,
                'sidelobe_loss': sidelobe_loss}
