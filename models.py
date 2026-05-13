import pickle
from typing import Any
import torch
from torch import nn, Tensor
import numpy as np
from cbam import CBAM
from config import Config
from layers import Block2d, Block2dTranspose, Fourier2D, SwiGLU, PositionalEncoding, FourierFeatureTrain, \
    MultiHeadLocationAwareAttention, RelativeMultiHeadAttention, xLSTM
from schedulers import CosineWarmupScheduler
from pytorch_lightning import LightningModule
from utils import _xavier_init, nonlinearities
import matplotlib.pyplot as plt


def calc_conv_size(inp_sz, kernel_sz, stride, padding):
    return np.floor((inp_sz - kernel_sz + 2 * padding) / stride) + 1


class FlatModule(LightningModule):
    """
    Base Module for different encoder models and generators. This adds
    parameters to flatten the model to use in a loss landscape, should it be desired.
    """

    def __init__(self, config: Config = None):
        super(FlatModule, self).__init__()
        self.config = config
        self.optim_path = []

    def get_flat_params(self):
        """Get flattened and concatenated params of the model."""
        return torch.cat([torch.flatten(p) for _, p in self._get_params().items()])

    def _get_params(self):
        return {name: param.data for name, param in self.named_parameters()}

    def init_from_flat_params(self, flat_params):
        """Set all model parameters from the flattened form."""
        assert isinstance(flat_params, torch.Tensor), "Argument to init_from_flat_params() must be torch.Tensor"
        state_dict = _unflatten_to_state_dict(flat_params, self._get_param_shapes())
        for name, params in self.state_dict().items():
            if name not in state_dict:
                state_dict[name] = params
        self.load_state_dict(state_dict, strict=True)

    def _get_param_shapes(self):
        return [
            (name, param.shape, param.numel())
            for name, param in self.named_parameters()
        ]

    def plot_grad_flow(self, named_parameters):
        ave_grads = []
        layers = []
        for name, p in named_parameters:
            if p.grad is None:
                continue
            elif p.requires_grad and ("bias" not in name):
                layers.append(name)
                ave_grads.append(p.grad.abs().mean().cpu().data.numpy())
                self.logger.experiment.add_histogram(tag=name, values=p.grad,
                                                     global_step=self.trainer.global_step)
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads), linewidth=1, color="k")
        plt.xticks(list(range(len(ave_grads))), layers, rotation='vertical')
        plt.xlim(left=0, right=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.rcParams["figure.figsize"] = (20, 5)




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

    def forward(self, *args, **kwargs) -> Tensor:
        pass

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
        pass


class TargetEmbedding(LightningModule):
    def __init__(self, input_dim, model_dim, embedding_size, num_layers, lr, warmup, max_iters, dropout=0.0,
                 input_dropout=0.1, nonlinearity='silu', *args, **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        nlin = nonlinearities[self.hparams.nonlinearity]
        # Input size is 256x4000
        # Input dim -> Model dim
        self.compress = nn.Sequential(
            nn.Linear(4000, 256),
            nlin,
            nn.Conv2d(2, 16, 5, padding='same'),
            nlin,
            nn.MaxPool2d((2, 2)),
            nn.LayerNorm(256 // 2),
            nn.Conv2d(16, 16, 5, padding='same'),
            nlin,
            nn.MaxPool2d((2, 2)),
            nn.LayerNorm(256 // 4),
            nn.Conv2d(16, 16, 5, padding='same'),
            nlin,
            nn.MaxPool2d((2, 2)),
            nn.LayerNorm(256 // 8),
            nn.Conv2d(16, 16, 5, padding='same'),
            nlin,
            nn.MaxPool2d((2, 2)),
            nn.LayerNorm(256 // 16),
        )

        self.flatten = nn.Sequential(
            nn.Linear(4096, self.hparams.embedding_size),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hparams.embedding_size, self.hparams.embedding_size),
            nlin,
            nn.Linear(self.hparams.embedding_size, 9),
            nn.Softmax(dim=1),
        )

        self.triplet = nn.TripletMarginLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

        _xavier_init(self)

    def forward(self, x):
        return self.classify(self.embed(x))

    def embed(self, x):
        x = self.compress(x)
        x = x.view(-1, 4096)
        return self.flatten(x)

    def classify(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss)
        # Avoid exploding gradients from high learning rates
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()
        sch = self.lr_schedulers()
        sch.step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_validation_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def on_train_epoch_end(self) -> None:
        pass
        # sch = self.lr_schedulers()
        # sch.step()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=0.0,
                                      eps=1e-7)
        scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=self.config.eta_min)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx, kind='train'):
        anchor, pos, neg, tidx = batch

        anchor_pass = self.embed(anchor)
        pos_pass = self.embed(pos)
        neg_pass = self.embed(neg)

        trip_loss = self.triplet(anchor_pass, pos_pass, neg_pass)
        cosine_loss = self.cosine_loss(anchor_pass, pos_pass, torch.ones(pos_pass.shape[0], device=pos_pass.device))
        classifier_loss = torch.nn.functional.binary_cross_entropy(self.classify(anchor_pass), tidx)

        loss = trip_loss + cosine_loss + classifier_loss

        # Logging ranking metrics
        self.log_dict({f'{kind}_loss': loss, f'{kind}_trip_loss': trip_loss, f'{kind}_cos_loss': cosine_loss,
                       f'{kind}_classifier_loss': classifier_loss,
                       'lr': self.lr_schedulers().get_last_lr()[0]}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)

        return loss


class ClutterTransformer(LightningModule):

    def __init__(self, input_dim, model_dim, num_layers, lr, warmup, max_iters, dropout=0.0,
        input_dropout=0.1, nonlinearity='silu', scheduler_gamma: float = .99, *args, **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        nlin = nonlinearities[self.hparams.nonlinearity]
        # Input dim -> Model dim
        self.input_real = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim),
            nlin,
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nlin,
            nn.LayerNorm(self.hparams.model_dim),
        )

        self.input_imag = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim),
            nlin,
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nlin,
            nn.LayerNorm(self.hparams.model_dim),
        )

        self.cross_attention = RelativeMultiHeadAttention(self.hparams.model_dim, 18)
        self.pos_enc = PositionalEncoding(self.hparams.model_dim, max_len=256)

        # Transformer

        self.encoder_output = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.Tanh(),
        )

        self.output_real = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.input_dim),
        )

        self.output_imag = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.input_dim),
        )

        _xavier_init(self)

    def encode(self, x):
        y = self.input_imag(x[..., 1, :])
        x = self.input_real(x[..., 0, :])
        pe = self.pos_enc(x)
        xy, _ = self.cross_attention(x, y, y, pe)
        return self.encoder_output(xy)

    def get_next_n(self, x, n):
        # batch x Sequence Length x model_dim
        x = torch.cat([x[..., 1:, :, :], self.forward(x)[..., -1:, :, :]], dim=-3)
        for _ in range(n):
            x = torch.cat([x[..., 1:, :, :], self.forward(x)[..., -1:, :, :]], dim=-3)
        return x

    def decode(self, x):
        return torch.cat([self.output_real(x).unsqueeze(-2), self.output_imag(x).unsqueeze(-2)], dim=-2)

    def forward(self, x):
        # The forward function is used for training the autoencoder
        return self.decode(self.encode(x))

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss)
        # Avoid exploding gradients from high learning rates
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()
        # sch = self.lr_schedulers()
        # sch.step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_validation_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()
        sch.step()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=0.0,
                                      eps=1e-7)
        # scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.scheduler_gamma)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=self.config.eta_min)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx, kind='train'):
        clutter_sequence, target_sequence, _, _, _, _, _, _ = batch
        clut_noise = clutter_sequence + torch.randn_like(clutter_sequence) * .001
        n = 2

        clutter_rec = self.get_next_n(clut_noise[:, :-n], n)

        # RECONSTRUCTION LOSS
        rec_loss = torch.mean(torch.square(clutter_sequence[:, -n:] - clutter_rec[:, -n:]))
        norm_loss = torch.mean(torch.linalg.norm(clutter_sequence[:, -n:] - clutter_rec[:, -n:], dim=-2))
        baseline = torch.mean(torch.square(clutter_sequence[:, -n:]))

        loss = norm_loss + rec_loss * self.current_epoch / 1e3

        # Logging ranking metrics
        self.log_dict({f'{kind}_rec_loss': rec_loss, f'{kind}_norm_loss': norm_loss, f'{kind}_baseline': baseline,
                       f'{kind}_loss': loss,
                       'lr': self.lr_schedulers().get_last_lr()[0]}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)

        return loss

    '''def on_after_backward(self):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 100 == 0 and self.trainer.global_step != 0:
           self.plot_grad_flow(self.named_parameters())'''



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
    from pytorch_lightning import seed_everything
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

