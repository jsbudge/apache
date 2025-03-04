import contextlib
import pickle
import torch
from neuralop import TFNO1d
from torch import nn, optim, Tensor
from torch.nn import functional as tf
import numpy as np

from config import Config
from layers import LKA, LKATranspose, LKA1d, LKATranspose1d
from pytorch_lightning import LightningModule
import matplotlib.pyplot as plt

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
                 **kwargs) -> None:
        super(TargetEmbedding, self).__init__(config)
        self.save_hyperparameters()
        self.latent_dim = config.latent_dim
        self.channel_sz = config.channel_sz
        self.in_channels = config.in_channels
        self.automatic_optimization = False
        self.temperature = 1.0

        # Parameters for normalizing data correctly
        levels = 2
        out_sz = config.fft_len // (2 ** levels)

        # Encoder
        self.encoder_inflate = nn.Conv1d(self.in_channels, self.channel_sz, 1, 1, 0)
        prev_lev_enc = self.channel_sz
        self.encoder_reduce = nn.ModuleList()
        self.encoder_conv = nn.ModuleList()
        for l in range(levels):
            ch_lev_enc = prev_lev_enc * 2
            layer_sz = config.fft_len // (2 ** (l + 1))
            self.encoder_reduce.append(nn.Sequential(
                nn.Conv1d(prev_lev_enc, ch_lev_enc, 4, 2, 1),
                nn.GELU(),
            ))
            self.encoder_conv.append(nn.Sequential(
                TFNO1d(n_modes_height=16, in_channels=ch_lev_enc, out_channels=ch_lev_enc, hidden_channels=ch_lev_enc),
                nn.LayerNorm(layer_sz),
            ))
            prev_lev_enc = ch_lev_enc + 0

        prev_lev_dec = prev_lev_enc
        self.encoder_flatten = nn.Sequential(
            TFNO1d(n_modes_height=8, in_channels=prev_lev_dec, out_channels=1, hidden_channels=prev_lev_dec),
        )
        self.fc_z = nn.Sequential(
            nn.Linear(out_sz, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        '''self.contrast_g = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )'''

        self.out_sz = out_sz
        self.example_input_array = torch.randn((1, 2, config.fft_len))

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        """
                Encodes the input by passing through the encoder network
                and returns the latent codes.
                :param inp: (Tensor) Input tensor to encoder [N x C x H x W]
                :return: (Tensor) List of latent codes
                """
        inp = self.encoder_inflate(inp)
        for conv, red in zip(self.encoder_conv, self.encoder_reduce):
            inp = conv(red(inp))
        inp = self.encoder_flatten(inp).view(-1, self.out_sz)
        return self.fc_z(inp)

    def contrast_map(self, inp: Tensor, **kwargs):
        # return self.contrast_g(self.forward(inp))
        return self.forward(inp)

    # def loss_function(self, y, y_pred):
    #     return tf.cosine_similarity(y, y_pred)

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
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx, kind='train'):
        img, idx = batch

        feats = self.contrast_map(img)
        cos_sim = tf.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        
        # Logging ranking metrics
        self.log_dict({f'{kind}_loss': nll, f"{kind}_acc_top1": (sim_argsort == 0).float().mean(),
                       f"{kind}_acc_top5": (sim_argsort < 5).float().mean(),
                       f"{kind}_acc_mean_pos": 1 + sim_argsort.float().mean()}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return nll


class PulseClassifier(LightningModule):
    def __init__(self,
                 config: Config,
                 embedding_model: LightningModule = None,
                 **kwargs) -> None:
        super(PulseClassifier, self).__init__(config)


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