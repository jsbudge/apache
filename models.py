import pickle
from typing import Any
import torch
from torch import nn, Tensor
import numpy as np
from cbam import CBAM
from config import Config
from layers import Block2d, Block2dTranspose, Fourier2D
from pytorch_lightning import LightningModule
from utils import _xavier_init, nonlinearities
from waveform_model import FlatModule


def calc_conv_size(inp_sz, kernel_sz, stride, padding):
    return np.floor((inp_sz - kernel_sz + 2 * padding) / stride) + 1


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
        latent_pow2_square = np.ceil(np.log2(self.latent_dim))
        latent_pow2_square = int(
            2 ** latent_pow2_square if latent_pow2_square % 2 == 0 else 2 ** (latent_pow2_square + 1))
        levels = int(config.angle_samples / np.sqrt(latent_pow2_square)) - 2
        # levels = config.levels
        # out_sz = (config.angle_samples // (2 ** levels), config.fft_len // (2 ** levels))
        out_sz = config.angle_samples // (2 ** levels)

        nonlinearity = nonlinearities[config.nonlinearity]

        # Encoder
        # self.encoder_inflate = nn.Conv2d(self.in_channels, self.channel_sz, 1, 1, 0)
        self.encoder_inflate = nn.Sequential(
            Fourier2D(10., 6),
            nn.Conv2d(self.in_channels * 6 * 2, self.channel_sz, 1, 1, 0),
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
                CBAM(ch_lev_enc, reduction_factor=4, kernel_size=7 - l * 2),
                Block2d(ch_lev_enc, 7 - l * 2, 1, 3 - l, nonlinearity=config.nonlinearity),
                # LKA(ch_lev_enc, (5, 3), dilation=3, activation=config.nonlinearity),
            ))
            prev_lev_enc = ch_lev_enc + 0

        prev_lev_dec = prev_lev_enc
        self.encoder_flatten = nn.Sequential(
            nn.Conv2d(prev_lev_enc, 1, (out_sz, 1), 1, 0),
            nonlinearity,
        )
        self.fc_z = nn.Sequential(
            nn.Linear(out_sz, self.latent_dim),
            nonlinearity,
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Softsign(),
        )

        self.decoder = DecoderHead(config.latent_dim, config.channel_sz, self.in_channels, out_sz,
                                   config.angle_samples, nonlinearity=config.nonlinearity, levels=levels)

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
        return self.fc_z(self.encoder_flatten(inp).view(-1, self.out_sz))

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
        # Normalize the image
        img = (img - torch.tensor(self.config.mu, device=self.device)[None, :, None, None]) / torch.tensor(self.config.var, device=self.device)[None, :, None, None]

        feats = self.encode(img)
        reconstructions = self.decoder(feats)

        # RECONSTRUCTION LOSS
        rec_loss = torch.mean(torch.square(img - reconstructions))
        '''rec_loss = sum(
            torch.mean(torch.abs(i - r)) for i, r in zip(img, reconstructions)
        )'''

        # CLIP LOSS
        idx = idx + 1
        clip_loss = torch.cosine_similarity(feats[None, :, :], feats[:, None, :], dim=-1)**2
        clip_loss = clip_loss * torch.nn.Hardsigmoid()(abs(1e-9 + torch.outer(idx, idx) - idx ** 2) ** 2 - 3)
        clip_loss = clip_loss.sum(1).mean() / img.shape[0]


        # COMBINATION LOSS
        cll = rec_loss + clip_loss * .01

        # Logging ranking metrics
        self.log_dict({f'{kind}_total_loss': cll, f'{kind}_clip_loss': clip_loss, f'{kind}_rec_loss': rec_loss,
                       'lr': self.lr_schedulers().get_last_lr()[0]}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)

        return cll


class DecoderHead(LightningModule):

    def __init__(self, latent_dim: int, channel_sz: int, in_channels: int, out_sz: int, in_sz: int,
                 nonlinearity: str, levels: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.channel_sz = channel_sz
        self.in_channels = in_channels
        inter_channels = channel_sz // (2**levels)

        latent_pow2_square = np.ceil(np.log2(self.latent_dim))
        latent_pow2_square = int(2**latent_pow2_square if latent_pow2_square % 2 == 0 else 2**(latent_pow2_square + 1))
        self.latent_sqrt = int(np.sqrt(latent_pow2_square))

        n_dec_layers = int(in_sz / np.sqrt(latent_pow2_square)) - 2
        nlin = nonlinearities[nonlinearity]

        self.decoder_inflate = nn.Sequential(
            Fourier2D(.33, 6),
            nn.ConvTranspose2d(6 * 2, inter_channels, 3, 1, 1),
            nlin,
            Block2dTranspose(inter_channels, 3, 1, 1, nonlinearity=nonlinearity)
        )


        self.z_fc = nn.Sequential(
            nlin,
            nn.Linear(self.latent_dim, latent_pow2_square),
            nlin,
            nn.Linear(latent_pow2_square, latent_pow2_square),
        )

        self.dec_layers = nn.ModuleList()
        for l in range(n_dec_layers - 1):
            inc = inter_channels * 2
            self.dec_layers.append(nn.Sequential(
                nlin,
                nn.ConvTranspose2d(inter_channels, inc, 4, 2, 1),
                # LKATranspose(inc, (5, 3), dilation=3, activation='silu'),
                Block2dTranspose(inc, 3 + l * 2, 1, 1 + l, nonlinearity=nonlinearity),
                CBAM(inc, reduction_factor=4, kernel_size=3 + l * 2),
            ))
            inter_channels = inc
        self.dec_layers.append(nn.Sequential(
            Block2dTranspose(inter_channels, 3, 1, 1, nonlinearity=nonlinearity),
            Block2dTranspose(inter_channels, 3, 1, 1, nonlinearity=nonlinearity),
            nn.ConvTranspose2d(inter_channels, in_channels, 4, 2, 1),
        ))

    def forward(self, x):
        x = self.z_fc(x)
        x = self.decoder_inflate(x.view(-1, 1, self.latent_sqrt, self.latent_sqrt))
        for l in self.dec_layers:
            x = l(x)
        return x

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

if __name__ == '__main__':
    from config import get_config
    from torchviz import make_dot
    from pytorch_lightning import Trainer, loggers, seed_everything
    torch.set_float32_matmul_precision('medium')
    gpu_num = 1
    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(43, workers=True)

    target_config = get_config('target_exp', './vae_config.yaml')
    mdl = TargetEmbedding(target_config)

    dummy = torch.zeros((1, 4, target_config.angle_samples, target_config.angle_samples))
    output = mdl(dummy)
    dot = make_dot(output, params=dict(mdl.named_parameters()))

    dot.format = 'png'
    dot.render('target_autoencoder')

