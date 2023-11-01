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
            nn.Conv1d(1, 64, 4, 1, 1),
            nn.Conv1d(64, 64, 4, 2, 1),
        )

        self.final = nn.Linear(2 * 16 * 64, bin_bw * n_ants)

    def forward(self, clutter: Tensor, target: Tensor) -> Tensor:
        ct_stack = torch.add(self.clutter_stack(clutter), self.target_stack(target))
        ct_stack = ct_stack.view(-1, 1, 32)
        ct = self.ffinit(ct_stack)
        ct = ct.view(-1, 2 * 16 * 64)
        wvs = self.final(ct)
        wvs = wvs.view(-1, self.n_ants, self.bin_bw)
        return wvs

    def loss_function(self, *args, **kwargs) -> dict:

        gen_waveform = torch.complex(args[0][:, 0, ...], args[0][:, 1, ...])
        gen_waveform = gen_waveform / torch.sum(torch.abs(gen_waveform), dim=1)
        clutter_spectrum = torch.complex(args[1][:, 0, ...], args[1][:, 1, ...])
        clutter_spectrum = clutter_spectrum / torch.sum(torch.abs(clutter_spectrum), dim=1)
        target_spectrum = torch.complex(args[2][:, 0, ...], args[2][:, 1, ...])
        target_spectrum = target_spectrum / torch.sum(torch.abs(target_spectrum), dim=1)

        clutter_loss = 1. / F.mse_loss(gen_waveform, clutter_spectrum)
        target_loss = F.mse_loss(gen_waveform, target_spectrum)
        wave_sidelobe = torch.corrcoef(gen_waveform, gen_waveform.conj())

        loss = clutter_loss + target_loss + wave_sidelobe

        return {'loss': loss, 'clutter_loss': clutter_loss, 'target_loss': target_loss, 'sidelobe_loss': wave_sidelobe}