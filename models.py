import contextlib
import pickle
import torch
from neuralop import TFNO1d
from torch import nn, optim, Tensor
from torch.nn import functional as tf
import numpy as np

from config import Config
from layers import LKA, LKATranspose, LKA1d, LKATranspose1d
from waveform_model import FlatModule
import matplotlib.pyplot as plt


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


class Encoder(FlatModule):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 fft_len: int,
                 params: dict,
                 channel_sz: int = 32,
                 **kwargs) -> None:
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.fft_len = fft_len
        self.params = params
        self.channel_sz = channel_sz
        self.in_channels = in_channels
        self.automatic_optimization = False
        levels = 2
        fft_scaling = 2 ** levels
        first_layer_size = 513

        # Encoder
        self.encoder_inflate = nn.Sequential(
            nn.Conv1d(in_channels, channel_sz, 1, 1, 0),
            nn.GELU(),
        )
        self.encoder_attention = nn.ModuleList()
        self.encoder_reduce = nn.ModuleList()
        self.encoder_conv = nn.ModuleList()
        decoder_reduce = []
        decoder_conv = []
        decoder_attention = []
        for n in range(levels):
            curr_channel_sz = channel_sz * 2**n
            next_channel_sz = channel_sz * 2**(n + 1)
            lin_sz = fft_len // (2 ** (n + 1))
            dec_lin_sz = fft_len // (2 ** n)
            self.encoder_reduce.append(nn.Sequential(
                nn.Conv1d(curr_channel_sz, curr_channel_sz, 4, 2, 1),
                nn.GELU(),
            ))
            self.encoder_conv.append(nn.Sequential(
                LKA1d(curr_channel_sz, kernel_sizes=(first_layer_size, 65), dilation=126),
                nn.LayerNorm(lin_sz),
                LKA1d(curr_channel_sz, kernel_sizes=(first_layer_size, 129), dilation=66),
                nn.LayerNorm(lin_sz),
                LKA1d(curr_channel_sz, kernel_sizes=(first_layer_size, 257), dilation=66),
                nn.LayerNorm(lin_sz),
                nn.Conv1d(curr_channel_sz, next_channel_sz, 1, 1, 0),
                nn.GELU(),
            ))
            self.encoder_attention.append(nn.Sequential(
                LKA1d(curr_channel_sz, kernel_sizes=(first_layer_size, 65), dilation=120),
                nn.Conv1d(curr_channel_sz, curr_channel_sz, 4, 2, 1),
                nn.Sigmoid(),
            ))
            decoder_reduce.append(nn.Sequential(
                nn.ConvTranspose1d(next_channel_sz, next_channel_sz, 4, 2, 1),
                nn.GELU(),
            ))
            decoder_conv.append(nn.Sequential(
                nn.ConvTranspose1d(next_channel_sz, curr_channel_sz, 1, 1, 0),
                nn.GELU(),
                LKATranspose1d(curr_channel_sz, kernel_sizes=(first_layer_size, 257), dilation=66),
                nn.LayerNorm(dec_lin_sz),
                LKATranspose1d(curr_channel_sz, kernel_sizes=(first_layer_size, 129), dilation=66),
                nn.LayerNorm(dec_lin_sz),
                LKATranspose1d(curr_channel_sz, kernel_sizes=(first_layer_size, 65), dilation=126),
                nn.LayerNorm(dec_lin_sz),

            ))
            decoder_attention.append(nn.Sequential(
                LKATranspose1d(next_channel_sz, kernel_sizes=(first_layer_size, 65), dilation=120),
                nn.ConvTranspose1d(next_channel_sz, next_channel_sz, 4, 2, 1),
                nn.Sigmoid(),
            ))
        self.encoder_squash = nn.Sequential(
            nn.Conv1d(channel_sz * 2**levels, channel_sz, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(channel_sz, 1, 1, 1, 0),
            nn.GELU(),
        )
        self.fc_z = nn.Sequential(
            nn.Linear(fft_len // fft_scaling, fft_len // fft_scaling),
            nn.GELU(),
            nn.Linear(fft_len // fft_scaling, latent_dim),
        )

        # Decoder
        self.z_fc = nn.Sequential(
            nn.Linear(latent_dim, fft_len // fft_scaling),
            nn.GELU(),
            nn.Linear(fft_len // fft_scaling, fft_len // fft_scaling),
            nn.GELU(),
        )

        # Flip the lists
        decoder_attention.reverse()
        decoder_conv.reverse()
        decoder_reduce.reverse()
        self.decoder_conv = nn.ModuleList(decoder_conv)
        self.decoder_attention = nn.ModuleList(decoder_attention)
        self.decoder_reduce = nn.ModuleList(decoder_reduce)
        self.decoder_inflate = nn.Sequential(
                nn.ConvTranspose1d(1, channel_sz * 2**levels, 1, 1, 0),
            nn.GELU(),
            nn.ConvTranspose1d(channel_sz * 2**levels, channel_sz * 2**levels, 3, 1, 1),
            nn.GELU(),
        )
        self.decoder_output = nn.Sequential(
            nn.ConvTranspose1d(channel_sz, in_channels, 1, 1, 0),
            nn.LayerNorm(fft_len)
        )

        self.example_input_array = torch.randn((1, 2, self.fft_len))

    def encode(self, inp: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param inp: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        inp = self.encoder_inflate(inp)
        for conv, red, att in zip(self.encoder_conv, self.encoder_reduce, self.encoder_attention):
            inp = conv(red(inp) + att(inp))
        inp = self.encoder_squash(inp).squeeze(1)
        return self.fc_z(inp)

    def decode(self, z: Tensor) -> Tensor:
        result = self.z_fc(z).unsqueeze(1)
        result = self.decoder_inflate(result)
        for conv, red, att in zip(self.decoder_conv, self.decoder_reduce, self.decoder_attention):
            result = conv(red(result) + att(result))
        return self.decoder_output(result)

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        z = self.encode(inp)
        return self.decode(z)

    def loss_function(self, y, y_pred):
        target_spectrum = torch.complex(y[:, 0, :], y[:, 1, :])
        tspec_mag = torch.sqrt(torch.sum(target_spectrum * torch.conj(target_spectrum),
                                                                 dim=1))[:, None]
        target_spectrum = target_spectrum / tspec_mag
        enc_spectrum = torch.complex(y_pred[:, 0, :], y_pred[:, 1, :])
        enc_mag = torch.sqrt(torch.sum(enc_spectrum * torch.conj(enc_spectrum),
                                                                 dim=1))[:, None]
        enc_spectrum = enc_spectrum / enc_mag
        return tf.mse_loss(torch.view_as_real(target_spectrum), torch.view_as_real(enc_spectrum))

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

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero and not self.params['is_tuning']:
            torch.save(self.state_dict(), './model/inference_model.state')
            print('Model saved to disk.')

            if self.current_epoch % 5 == 0 and self.params['output_images']:
                # Log an image to get an idea of progress
                img, _ = next(iter(self.trainer.val_dataloaders))
                rec = self.forward(img.to(self.device))
                rec = rec.to('cpu').data.numpy()
                fig = plt.figure()
                plt.subplot(2, 2, 1)
                plt.title('Real Original')
                plt.plot(img[0, 0, :])
                plt.subplot(2, 2, 2)
                plt.title('Imag Original')
                plt.plot(img[0, 1, :])
                plt.subplot(2, 2, 3)
                plt.title('Real Reconstructed')
                plt.plot(rec[0, 0, :])
                plt.subplot(2, 2, 4)
                plt.title('Imag Reconstructed')
                plt.plot(rec[0, 1, :])
                self.logger.experiment.add_figure('Reconstruction', fig, self.current_epoch)

    def on_train_epoch_end(self) -> None:
        if self.trainer.is_global_zero and not self.params['is_tuning'] and self.params['loss_landscape']:
            self.optim_path.append(self.model.get_flat_params())

    def on_validation_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
            self.log('LR', sch.get_last_lr()[0], rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'],
                                betas=self.params['betas'],
                                eps=1e-7)
        optims = [optimizer]
        if self.params['scheduler_gamma'] is None:
            return optims
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optims[0], cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)
        scheds = [scheduler]

        return optims, scheds

    def train_val_get(self, batch, batch_idx, kind='train'):
        img, img = batch

        results = self.forward(img)
        train_loss = self.loss_function(results, img)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss


class TargetEncoder(FlatModule):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 params: dict,
                 channel_sz: int = 32,
                 mu: float = .01,
                 var: float = 4.9,
                 **kwargs) -> None:
        super(TargetEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.params = params
        self.channel_sz = channel_sz
        self.in_channels = in_channels
        self.automatic_optimization = False

        # Parameters for normalizing data correctly
        self.mu = mu
        self.var = var
        levels = 3
        out_sz = 256 // (2 ** levels)

        # Encoder
        self.encoder_inflate = nn.Conv2d(in_channels, channel_sz, 1, 1, 0)
        prev_lev_enc = channel_sz
        self.encoder_reduce = nn.ModuleList()
        self.encoder_pool = nn.ModuleList()
        self.encoder_conv = nn.ModuleList()
        for l in range(levels):
            ch_lev_enc = prev_lev_enc * 2
            layer_sz = [256 // (2 ** (l + 1)), 256 // (2 ** (l + 1))]
            self.encoder_reduce.append(nn.Sequential(
                nn.Conv2d(prev_lev_enc, ch_lev_enc, 4, 2, 1),
                nn.GELU(),
            ))
            self.encoder_pool.append(nn.Sequential(nn.Conv2d(prev_lev_enc, ch_lev_enc, 1, 1, 0),
                                                   nn.MaxPool2d(2)))
            self.encoder_conv.append(nn.Sequential(
                LKA(ch_lev_enc, kernel_sizes=(15, 15), dilation=6),
                nn.LayerNorm(layer_sz),
                nn.Conv2d(ch_lev_enc, ch_lev_enc, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(ch_lev_enc, ch_lev_enc, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(ch_lev_enc, ch_lev_enc, 3, 1, 1),
                nn.GELU(),
                nn.LayerNorm(layer_sz),
            ))
            prev_lev_enc = ch_lev_enc + 0

        prev_lev_dec = prev_lev_enc
        self.encoder_flatten = nn.Sequential(
            nn.Conv2d(prev_lev_dec, 1, 1, 1, 0),
            nn.GELU(),
        )
        self.fc_z = nn.Sequential(
            nn.Linear(out_sz ** 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
        )

        # Decoder
        self.z_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, out_sz ** 2),
        )
        self.decoder_flatten = nn.Sequential(
            nn.Conv2d(1, prev_lev_dec, 1, 1, 0),
            nn.GELU(),
        )
        self.decoder_reduce = nn.ModuleList()
        self.decoder_pool = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()
        for l in range(1, levels + 1):
            ch_lev_dec = prev_lev_dec // 2
            layer_sz = [out_sz * (2 ** l), out_sz * (2 ** l)]
            self.decoder_reduce.append(nn.Sequential(
                nn.ConvTranspose2d(prev_lev_dec, ch_lev_dec, 4, 2, 1),
                nn.GELU(),
            ))
            self.decoder_pool.append(nn.Sequential(
                nn.ConvTranspose2d(prev_lev_dec, ch_lev_dec, 1, 1, 0),
                nn.UpsamplingNearest2d(scale_factor=2)))
            self.decoder_conv.append(nn.Sequential(
                LKATranspose(ch_lev_dec, kernel_sizes=(15, 15), dilation=6),
                nn.LayerNorm(layer_sz),
                nn.ConvTranspose2d(ch_lev_dec, ch_lev_dec, 3, 1, 1),
                nn.GELU(),
                nn.ConvTranspose2d(ch_lev_dec, ch_lev_dec, 3, 1, 1),
                nn.GELU(),
                nn.ConvTranspose2d(ch_lev_dec, ch_lev_dec, 3, 1, 1),
                nn.GELU(),
                nn.LayerNorm(layer_sz),
            ))
            prev_lev_dec = ch_lev_dec + 0
        self.decoder_output = nn.Sequential(
            nn.ConvTranspose2d(channel_sz, in_channels, 1, 1, 0),
        )

        self.latent_dim = latent_dim
        self.out_sz = out_sz
        self.example_input_array = torch.randn((1, 2, 256, 256))

    def encode(self, inp: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param inp: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        inp = self.encoder_inflate(inp)
        for conv, red, pool in zip(self.encoder_conv, self.encoder_reduce, self.encoder_pool):
            inp = conv(red(inp) * pool(inp))
        inp = self.encoder_flatten(inp).view(-1, self.out_sz ** 2)
        return self.fc_z(inp)

    def decode(self, z: Tensor) -> Tensor:
        result = self.z_fc(z).view(-1, 1, self.out_sz, self.out_sz)
        result = self.decoder_flatten(result)
        for conv, red, pool in zip(self.decoder_conv, self.decoder_reduce, self.decoder_pool):
            result = conv(red(result) * pool(result))
        return self.decoder_output(result)

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        z = self.encode(inp)
        return self.decode(z)

    def full_encode(self, inp: np.ndarray):
        return self.encode(torch.tensor((inp - self.mu) / self.var, torch.float32, device=self.device))

    def full_decode(self, z: np.ndarray):
        return self.decode(torch.tensor(z, torch.float32, device=self.device)).detach().numpy() * self.var + self.mu

    def loss_function(self, y, y_pred):
        return tf.mse_loss(y, y_pred)

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

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero and not self.params['is_tuning']:
            torch.save(self.state_dict(), './model/target_model.state')
            print('Model saved to disk.')

            if self.current_epoch % 5 == 0:
                pass

    def on_train_epoch_end(self) -> None:
        if self.trainer.is_global_zero and not self.params['is_tuning'] and self.params['loss_landscape']:
            self.optim_path.append(self.model.get_flat_params())

    def on_validation_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
            self.log('LR', sch.get_last_lr()[0], rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'],
                                betas=self.params['betas'],
                                eps=1e-7)
        optims = [optimizer]
        if self.params['scheduler_gamma'] is None:
            return optims
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optims[0], cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)
        scheds = [scheduler]

        return optims, scheds

    def train_val_get(self, batch, batch_idx, kind='train'):
        img, img = batch

        results = self.forward(img)
        train_loss = self.loss_function(results, img)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss


class TargetEmbedding(FlatModule):
    def __init__(self,
                 config: Config,
                 **kwargs) -> None:
        super(TargetEmbedding, self).__init__(config)
        self.latent_dim = config.latent_dim
        self.channel_sz = config.channel_sz
        self.in_channels = config.in_channels
        self.mu = config.mu
        self.var = config.var
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
                # LKA1d(ch_lev_enc, kernel_sizes=(513, 513), dilation=20),
                LKA1d(ch_lev_enc, kernel_sizes=(129, 129), dilation=60),
                nn.LayerNorm(layer_sz),
                # nn.Conv1d(ch_lev_enc, ch_lev_enc, 257, 1, 128),
                # nn.GELU(),
            ))
            prev_lev_enc = ch_lev_enc + 0

        prev_lev_dec = prev_lev_enc
        self.encoder_flatten = nn.Sequential(
            TFNO1d(n_modes_height=6, in_channels=prev_lev_dec,
                   out_channels=prev_lev_dec, hidden_channels=prev_lev_dec),
            nn.Conv1d(prev_lev_dec, 1, 1, 1, 0),
            nn.GELU(),
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


class PulseClassifier(FlatModule):
    def __init__(self,
                 config: Config,
                 embedding_model: FlatModule = None,
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