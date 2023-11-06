from typing import List, Any, TypeVar

# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')
import torch
from torch import nn
from torch.nn import functional as F
from abc import abstractmethod
from pytorch_lightning import LightningModule


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

        self.clutter_stack = nn.Sequential(
            nn.Linear(clutter_latent_size, 32),
            nn.LeakyReLU(),
        )

        self.target_stack = nn.Sequential(
            nn.Linear(target_latent_size, 32),
            nn.LeakyReLU(),
        )

        self.ffinit = nn.Sequential(
            nn.Conv1d(1, 512, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 512, 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(512, 256, 35, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(256, 256, 67, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(256, 64, 131, 1, 1),
            nn.LeakyReLU(),
            nn.Conv1d(64, n_ants * 2, 33, 1, 1),
            nn.Upsample(scale_factor=29, mode='linear')
        )

    def forward(self, clutter: Tensor, target: Tensor) -> Tensor:
        ct_stack = torch.add(self.clutter_stack(clutter), self.target_stack(target))
        ct_stack = ct_stack.view(-1, 1, 32)
        return self.ffinit(ct_stack)

    def loss_function(self, *args, **kwargs) -> dict:

        gen_waveform = torch.complex(args[0][:, 0, ...], args[0][:, 1, ...])
        gen_waveform = gen_waveform / torch.sum(torch.abs(gen_waveform), dim=1)[:, None] * 100
        clutter_spectrum = torch.complex(args[1][:, :, 0], args[1][:, :, 1])
        clutter_spectrum = clutter_spectrum / torch.sum(torch.abs(clutter_spectrum), dim=1)[:, None] * 100
        target_spectrum = torch.complex(args[2][:, :, 0], args[2][:, :, 1])
        target_spectrum = target_spectrum / torch.sum(torch.abs(target_spectrum), dim=1)[:, None] * 100

        clutter_loss = torch.abs(.0001 / (.5 * (gen_waveform - clutter_spectrum)**2).mean(dtype=torch.complex64))
        target_loss = torch.abs((.5 * (gen_waveform - target_spectrum)**2).mean(dtype=torch.complex64))
        # wave_sidelobe = torch.abs(torch.cov(gen_waveform))

        loss = clutter_loss + 200 * target_loss#  + wave_sidelobe

        return {'loss': loss, 'clutter_loss': clutter_loss, 'target_loss': target_loss}