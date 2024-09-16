import contextlib
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as tf
import numpy as np
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
        levels = 3
        fft_scaling = 2 ** levels
        first_layer_size = 129

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
                LKA1d(curr_channel_sz, kernel_sizes=(first_layer_size, 65), dilation=9),
                nn.LayerNorm(lin_sz),
                LKA1d(curr_channel_sz, kernel_sizes=(first_layer_size, 129), dilation=6),
                nn.LayerNorm(lin_sz),
                LKA1d(curr_channel_sz, kernel_sizes=(first_layer_size, 257), dilation=3),
                nn.LayerNorm(lin_sz),
                nn.Conv1d(curr_channel_sz, next_channel_sz, 1, 1, 0),
                nn.GELU(),
            ))
            self.encoder_attention.append(nn.Sequential(
                LKA1d(curr_channel_sz, kernel_sizes=(first_layer_size, 9), dilation=12),
                nn.Conv1d(curr_channel_sz, curr_channel_sz, 4, 2, 1),
                nn.Sigmoid(),
            ))
            decoder_reduce.append(nn.Sequential(
                nn.ConvTranspose1d(next_channel_sz, next_channel_sz, 4, 2, 1),
                nn.GELU(),
            ))
            decoder_conv.append(nn.Sequential(
                LKATranspose1d(next_channel_sz, kernel_sizes=(first_layer_size, 65), dilation=9),
                nn.LayerNorm(dec_lin_sz),
                LKATranspose1d(next_channel_sz, kernel_sizes=(first_layer_size, 129), dilation=6),
                nn.LayerNorm(dec_lin_sz),
                LKATranspose1d(next_channel_sz, kernel_sizes=(first_layer_size, 257), dilation=3),
                nn.LayerNorm(dec_lin_sz),
                nn.ConvTranspose1d(next_channel_sz, curr_channel_sz, 1, 1, 0),
                nn.GELU(),
            ))
            decoder_attention.append(nn.Sequential(
                LKATranspose1d(next_channel_sz, kernel_sizes=(first_layer_size, 9), dilation=12),
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
            nn.ConvTranspose1d(channel_sz, channel_sz, 3, 1, 1),
            nn.ConvTranspose1d(channel_sz, channel_sz, 1, 1, 0),
            nn.ConvTranspose1d(channel_sz, channel_sz, 1, 1, 0),
            nn.Conv1d(channel_sz, in_channels, 1, 1, 0),
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
        # y = y / torch.sqrt(torch.sum(y * torch.conj(y), dim=1))[:, None]
        # y_pred = y_pred / torch.sqrt(torch.sum(y_pred * torch.conj(y_pred), dim=1))[:, None]
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

