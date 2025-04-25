import contextlib
import pickle
from typing import Any

import torch
from neuralop import TFNO1d, TFNO2d
from torch import nn, optim, Tensor
from torch.nn import functional as tf
import numpy as np

from cbam import CBAM, GETheta
from config import Config
from layers import LKA, LKATranspose, LKA1d, LKATranspose1d, Block2d, Block2dTranspose
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt

from utils import _xavier_init
from waveform_model import FlatModule


def calc_conv_size(inp_sz, kernel_sz, stride, padding):
    return np.floor((inp_sz - kernel_sz + 2 * padding) / stride) + 1


def init_weights(m):
    with contextlib.suppress(ValueError):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_normal_(m.weight)
        # sourcery skip: merge-nested-ifs
        if hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.fill_(.01)


class TargetEmbedding(FlatModule):
    def __init__(self,
                 config: Config,
                 training_mode: int = 0,
                 **kwargs) -> None:
        super(TargetEmbedding, self).__init__(config)
        self.save_hyperparameters()
        self.latent_dim = config.latent_dim
        self.channel_sz = config.channel_sz
        self.in_channels = config.range_samples * 2
        self.automatic_optimization = False
        self.temperature = config.temperature
        self.mode = training_mode

        # Parameters for normalizing data correctly
        levels = config.levels
        # out_sz = (config.angle_samples // (2 ** levels), config.fft_len // (2 ** levels))
        out_sz = (config.angle_samples // (2 ** levels), config.angle_samples // (2 ** levels))

        nonlinearity = nn.SiLU() if config.nonlinearity == 'silu' else nn.GELU() if config.nonlinearity == 'gelu' else nn.SELU()

        # Encoder
        # self.encoder_inflate = nn.Conv2d(self.in_channels, self.channel_sz, 1, 1, 0)
        self.encoder_inflate = nn.Sequential(
            nn.Conv2d(self.in_channels, self.channel_sz, 1, 1, 0),
            nonlinearity,
            # CBAM(self.channel_sz, reduction_factor=1, kernel_size=9),
        )
        prev_lev_enc = self.channel_sz
        self.encoder_reduce = nn.ModuleList()
        self.encoder_conv = nn.ModuleList()
        for l in range(levels):
            ch_lev_enc = prev_lev_enc * 2
            layer_sz = (config.angle_samples // (2 ** (l + 1)), config.fft_len // (2 ** (l + 1)))
            self.encoder_reduce.append(nn.Sequential(
                nn.Conv2d(prev_lev_enc, ch_lev_enc, 4, 2, 1),
                nonlinearity,
            ))
            self.encoder_conv.append(nn.Sequential(
                CBAM(ch_lev_enc, reduction_factor=4, kernel_size=3),
                Block2d(ch_lev_enc, 3, 1, 1, nonlinearity=nonlinearity),
                # LKA(ch_lev_enc, (5, 3), dilation=3, activation=config.nonlinearity),
            ))
            prev_lev_enc = ch_lev_enc + 0

        prev_lev_dec = prev_lev_enc
        self.encoder_flatten = nn.Sequential(
            nn.Conv2d(prev_lev_enc, 1, 1, 1, 0),
            nonlinearity,
            nn.Conv2d(1, 1, (out_sz[0], 1), 1, 0),
            nonlinearity,
        )
        self.fc_z = nn.Sequential(
            nn.Linear(out_sz[1], self.latent_dim),
            nonlinearity,
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.decoder = DecoderHead(config.latent_dim, config.channel_sz, self.in_channels, out_sz,
                                   (config.angle_samples, config.fft_len), nonlinearity=nonlinearity, levels=levels)

        _xavier_init(self)

        self.out_sz = out_sz
        # self.example_input_array = torch.randn((1, self.in_channels, config.angle_samples, config.fft_len))

    def encode(self, inp: Tensor, **kwargs) -> Tensor:
        """
                        Encodes the input by passing through the encoder network
                        and returns the latent codes.
                        :param inp: (Tensor) Input tensor to encoder [N x C x H x W]
                        :return: (Tensor) List of latent codes
                        """
        inp = self.encoder_inflate(inp)
        for conv, red in zip(self.encoder_conv, self.encoder_reduce):
            inp = conv(red(inp))
        return self.fc_z(self.encoder_flatten(inp).view(-1, self.out_sz[1]))

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        enc = self.encode(inp, **kwargs)
        return self.decoder(enc)

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero and self.logger:
            self.logger.log_graph(self, self.example_input_array)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_validation_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        else:
            sch.step()
        if self.trainer.is_global_zero and not self.config.is_training and self.config.loss_landscape:
            self.optim_path.append(self.get_flat_params())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.config.lr,
                                      weight_decay=self.config.weight_decay,
                                      betas=self.config.betas,
                                      eps=1e-7)
        if self.config.scheduler_gamma is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config.scheduler_gamma,
                                                           verbose=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=self.config.eta_min)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx, kind='train'):
        img, idx = batch
        # n_img = (img - torch.mean(img)) / torch.std(img)

        feats = self.encode(img)
        reconstructions = self.decoder(feats)

        # RECONSTRUCTION LOSS
        rec_loss = torch.mean(torch.square(img - reconstructions))
        '''rec_loss = sum(
            torch.mean(torch.abs(i - r)) for i, r in zip(img, reconstructions)
        )'''

        # COMBINATION LOSS
        cll = rec_loss + torch.mean(torch.abs(img - reconstructions)) * .01

        # Logging ranking metrics
        self.log_dict({f'{kind}_total_loss': cll, f'{kind}_rec_loss': rec_loss,
                       'lr': self.lr_schedulers().get_last_lr()[0]}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)

        return cll


class DecoderHead(LightningModule):

    def __init__(self, latent_dim: int, channel_sz: int, in_channels: int, out_sz: tuple, in_sz: tuple,
                 nonlinearity: nn.Module, levels: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.channel_sz = channel_sz
        self.in_channels = in_channels
        self.out_sz = out_sz
        inter_channels = channel_sz // (2**levels)

        n_dec_layers = int(in_sz[0] / out_sz[0])

        self.decoder_inflate = nn.Sequential(
            nn.ConvTranspose2d(1, inter_channels, (out_sz[0], 1), 1, 0),
            nonlinearity,
        )
        self.z_fc = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nonlinearity,
            nn.Linear(self.latent_dim, out_sz[1]),
            nonlinearity,
        )

        self.dec_layers = nn.ModuleList()
        for l in range(levels - 1):
            inc = inter_channels * 2
            self.dec_layers.append(nn.Sequential(
                nn.ConvTranspose2d(inter_channels, inc, 4, 2, 1),
                nonlinearity,
                # LKATranspose(inc, (5, 3), dilation=3, activation='silu'),
                Block2dTranspose(inc, 3, 1, 1, nonlinearity=nn.SiLU()),
                CBAM(inc, reduction_factor=4, kernel_size=3),
            ))
            inter_channels = inc
        self.dec_layers.append(nn.Sequential(
            nn.ConvTranspose2d(inter_channels, in_channels, 4, 2, 1),
        ))

    def forward(self, x):
        x = self.z_fc(x)
        x = self.decoder_inflate(x.view(-1, 1, 1, self.out_sz[1]))
        for l in self.dec_layers:
            x = l(x)
        return x





class PulseClassifier(LightningModule):
    def __init__(self, config: Config, embedding_model: LightningModule = None, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.config = config
        self.channel_sz = config.channel_sz
        self.in_channels = config.in_channels
        self.label_sz = config.label_sz
        self.automatic_optimization = False
        self.embedding_model = embedding_model
        self.embedding_model.eval()
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        self.embedding_model.to(self.device)

        self.first_layer = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels),
            nn.GELU(),
        )
        self.feedthrough = nn.Sequential(
            nn.Conv1d(1, self.channel_sz, 1, 1, 0),
            nn.GELU(),
            nn.Conv1d(self.channel_sz, self.channel_sz, 15, 1, 7),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(self.channel_sz, self.channel_sz, 33, 1, 16),
            nn.GELU(),
            nn.Conv1d(self.channel_sz, 1, 1, 1, 0),
            nn.GELU(),
        )
        self.final_layer = nn.Sequential(
            nn.Linear(self.in_channels // 2, self.label_sz),
            nn.Sigmoid()
        )

        _xavier_init(self)

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        with torch.no_grad():
            self.embedding_model.eval()
            inp = self.embedding_model(inp)
        inp = self.first_layer(inp)
        inp = inp.view(-1, 1, self.in_channels)
        inp = self.feedthrough(inp)
        inp = inp.view(-1, self.in_channels // 2)
        inp = self.final_layer(inp)
        return inp

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero and self.logger:
            self.logger.log_graph(self, self.example_input_array)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_after_backward(self) -> None:
        if self.trainer.is_global_zero and self.global_step % 100 == 0 and self.logger:
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.global_step)

    def on_validation_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        else:
            sch.step()
        if self.trainer.is_global_zero and not self.config.is_tuning and self.config.loss_landscape:
            self.optim_path.append(self.get_flat_params())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.config.lr,
                                      weight_decay=self.config.weight_decay,
                                      betas=self.config.betas,
                                      eps=1e-7)
        if self.config.scheduler_gamma is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config.scheduler_gamma,
                                                           verbose=True)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx, kind='train'):
        img, idx = batch
        randidxes = torch.randperm(img.shape[0])
        feats = self.forward(img[randidxes])
        nll = tf.binary_cross_entropy(feats, tf.one_hot(idx[randidxes]).float())

        # Logging ranking metrics
        self.log_dict({f'{kind}_loss': nll}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return nll

def load(mdl, param_file):
    try:
        with open(param_file, 'rb') as f:
            generator_params = pickle.load(f)
        loaded_mdl = mdl(**generator_params)
        loaded_mdl.load_state_dict(torch.load(generator_params['state_file']), weights_only=True)
        loaded_mdl.eval()
        return loaded_mdl
    except RuntimeError as e:
        return None