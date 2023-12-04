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
    def __init__(self, in_channels, out_channels, out_layer_sz):
        super(RichConv1d, self).__init__()
        self.branch0 = nn.Conv1d(in_channels, out_channels, 4, 2, 1)
        self.branch1 = nn.Conv1d(in_channels, out_channels, out_layer_sz + 3, 1, 1)
        self.branch2 = nn.Conv1d(in_channels, out_channels, out_layer_sz + 1, 1, 0)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = torch.add(torch.add(F.leaky_relu(self.branch0(x)), F.leaky_relu(self.branch1(x))),
                      F.leaky_relu(self.branch2(x)))
        return self.batch_norm(x)


class RichConvTranspose1d(LightningModule):
    def __init__(self, in_channels, out_channels, in_layer_sz):
        super(RichConvTranspose1d, self).__init__()
        self.branch0 = nn.ConvTranspose1d(in_channels, out_channels, 4, 2, 1)
        self.branch1 = nn.ConvTranspose1d(in_channels, out_channels, in_layer_sz + 3, 1, 1)
        self.branch2 = nn.ConvTranspose1d(in_channels, out_channels, in_layer_sz + 1, 1, 0)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = torch.add(torch.add(F.leaky_relu(self.branch0(x)), F.leaky_relu(self.branch1(x))),
                      F.leaky_relu(self.branch2(x)))
        return self.batch_norm(x)


class GeneratorModel(LightningModule):
    def __init__(self,
                 bin_bw: int,
                 clutter_latent_size: int,
                 target_latent_size: int,
                 n_ants: int,
                 ) -> None:
        super(GeneratorModel, self).__init__()

        self.n_ants = n_ants
        self.bin_bw = bin_bw

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
            RichConvTranspose1d(1, 512, stack_output_sz),
            RichConvTranspose1d(512, 256, 128),
            RichConvTranspose1d(256, 128, 256),
            RichConvTranspose1d(128, 64, 512),
            RichConvTranspose1d(64, 32, 1024),
            RichConvTranspose1d(32, 16, 2048),
            nn.Conv1d(16, n_ants * 2, 820, 1, 0),

            nn.Upsample(scale_factor=2, mode='linear')
        )

        self.stack_output_sz = stack_output_sz

    def forward(self, clutter: Tensor, target: Tensor) -> Tensor:
        ct_stack = torch.add(self.clutter_stack(clutter), self.target_stack(target))
        ct_stack = ct_stack.view(-1, 1, self.stack_output_sz)
        return self.ffinit(self.batch_norm(ct_stack))

    def loss_function(self, *args, **kwargs) -> dict:
        gen_waveform = torch.complex(args[0][:, ::2, ...], args[0][:, 1::2, ...])
        gen_waveform = gen_waveform / torch.sqrt(torch.sum(gen_waveform * torch.conj(gen_waveform), dim=2))[:, :, None]
        clutter_spectrum = torch.complex(args[1][:, :, 0], args[1][:, :, 1])
        clutter_spectrum = clutter_spectrum / torch.sqrt(torch.sum(clutter_spectrum * torch.conj(clutter_spectrum),
                                                                   dim=1))[:, None]
        target_spectrum = torch.complex(args[2][:, :, 0], args[2][:, :, 1])
        target_spectrum = target_spectrum / torch.sqrt(torch.sum(target_spectrum * torch.conj(target_spectrum),
                                                                 dim=1))[:, None]
        left_sig_c = (gen_waveform - clutter_spectrum[:, None, :])
        left_sig_t = (gen_waveform - target_spectrum[:, None, :])

        clutter_loss = 4. * self.n_ants - torch.abs(torch.sum(torch.sum(left_sig_c * torch.conj(left_sig_c),
                                                                                   dim=2))) / gen_waveform.shape[0]
        target_loss = torch.abs(torch.sum(torch.sum(left_sig_t * torch.conj(left_sig_t),
                                                               dim=2))) / gen_waveform.shape[0]
        # wave_sidelobe = torch.abs(torch.cov(gen_waveform))

        loss = clutter_loss + 2 * target_loss  # + wave_sidelobe

        return {'loss': loss, 'clutter_loss': clutter_loss, 'target_loss': target_loss}
