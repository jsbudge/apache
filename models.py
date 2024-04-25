import contextlib
from typing import List, Any
from layers import RichConv2d, RichConvTranspose2d, Block1d
import torch
from torch import nn, optim, Tensor
from torch.nn import functional as tf
from abc import abstractmethod
from pytorch_lightning import LightningModule
import numpy as np
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


class BaseVAE(LightningModule):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class BetaVAE(BaseVAE):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma: float = 1000.,
                 max_capacity: int = 25,
                 capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = capacity_max_iter

        self.save_hyperparameters(
            'in_channels',
            'latent_dim',
            'hidden_dims',
            'beta',
            'gamma',
            'max_capacity',
            'capacity_max_iter',
            'loss_type',
            *list(kwargs.keys())
        )

        # Kernel size, stride, padding
        layer_params = [8, 4, 2]
        if hidden_dims is None:
            hidden_dims = [128, 128, 128]
        self.final_sz = hidden_dims[-1]
        initial_channels = in_channels + 0

        modules = [
            nn.Sequential(
                RichConv2d(in_channels, hidden_dims[0], 16),
                nn.Conv2d(hidden_dims[0], hidden_dims[0], 3, 1, 1),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[0], hidden_dims[0], 3, 1, 1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(hidden_dims[0]),
            )
        ]
        for h_dim, l_params in zip(hidden_dims, layer_params):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(h_dim, h_dim, 4, 2, 1),
                    nn.LeakyReLU(),
                    nn.BatchNorm2d(h_dim),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder.apply(init_weights)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        layer_params.reverse()

        modules.extend(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[i + 1], hidden_dims[i + 1], 4, 2, 1
                ),
                nn.LeakyReLU(),
                nn.BatchNorm2d(hidden_dims[i + 1]),
            )
            for i in range(len(hidden_dims) - 1)
        )

        modules.append(nn.Sequential(
            RichConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 16),
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], 3, 1, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(hidden_dims[-1]),
        ))
        self.decoder = nn.Sequential(*modules)
        self.decoder.apply(init_weights)

        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], out_channels=initial_channels,
                      kernel_size=3, padding=1))

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.final_sz, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = tf.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class InfoVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 alpha: float = -0.5,
                 beta: float = 5.0,
                 reg_weight: int = 100,
                 kernel_type: str = 'imq',
                 latent_var: float = 2.,
                 **kwargs) -> None:
        super(InfoVAE, self).__init__()

        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var

        assert alpha <= 0, 'alpha must be negative or zero.'

        self.alpha = alpha
        self.beta = beta

        modules = []
        # Kernel size, stride, padding
        layer_params = [16, 8, 4, 2]
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128]
        else:
            hidden_dims = [hidden_dims for _ in layer_params]
        self.final_sz = hidden_dims[-1]
        initial_channels = in_channels + 0

        # Build Encoder
        for h_dim, l_params in zip(hidden_dims, layer_params):
            modules.append(
                RichConv2d(in_channels, h_dim, l_params)
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        layer_params.reverse()

        modules.extend(
            RichConvTranspose2d(
                hidden_dims[i], hidden_dims[i + 1], layer_params[i]
            )
            for i in range(len(hidden_dims) - 1)
        )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=initial_channels,
                      kernel_size=4, padding=1))

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.final_sz, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, z, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        z = args[2]
        mu = args[3]
        log_var = args[4]

        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = tf.mse_loss(recons, input)
        mmd_loss = self.compute_mmd(z)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = self.beta * recons_loss + \
               (1. - self.alpha) * kld_weight * kld_loss + \
               (self.alpha + self.reg_weight - 1.) / bias_corr * mmd_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'MMD': mmd_loss, 'KLD': -kld_loss}

    def compute_kernel(self,
                       x1: Tensor,
                       x2: Tensor) -> Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result

    def compute_rbf(self,
                    x1: Tensor,
                    x2: Tensor,
                    eps: float = 1e-7) -> Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self,
                              x1: Tensor,
                              x2: Tensor,
                              eps: float = 1e-7) -> Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z: Tensor) -> Tensor:
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + \
              z__kernel.mean() - \
              2 * priorz_z__kernel.mean()
        return mmd

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


class WAE_MMD(BaseVAE):
    """
    Wasserstein AutoEncoder - Maximum Mean Discrepancy
    https://arxiv.org/pdf/1711.01558.pdf

    This is essentially an adversarial autoencoder that encourages the coded distribution to match the prior. It is
    a different flavor of VAE.
    """

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 fft_len: int,
                 channel_sz: int = 32,
                 reg_weight: int = 100,
                 kernel_type: str = 'imq',
                 latent_var: float = 2.,
                 **kwargs) -> None:
        super(WAE_MMD, self).__init__()

        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var
        self.fft_len = fft_len

        self.channel_sz = channel_sz
        self.in_channels = in_channels
        levels = 3
        fft_scaling = 2 ** levels

        # Encoder
        env = [
            nn.Sequential(
                nn.Conv1d(in_channels, channel_sz, 1, 1, 0),
                nn.GELU(),
            )]
        env += [
            nn.Sequential(
                nn.Conv1d(
                    channel_sz,
                    channel_sz,
                    4,
                    2,
                    1,
                ),
                nn.GELU(),
                nn.Linear(fft_len // (2**n), fft_len // (2**n)),
                nn.GELU(),
            )
            for n in range(1, levels + 1)
        ]
        self.envelope_encoder = nn.Sequential(*env)
        self.envelope_encoder.apply(init_weights)
        self.encoder_reduce = nn.Sequential(
            nn.Conv1d(channel_sz, 1, 1, 1, 0),
            nn.GELU(),
        )
        self.fc_z = nn.Sequential(
            nn.Linear(fft_len // fft_scaling, latent_dim),
            nn.Tanh(),
        )

        # Decoder
        self.z_fc = nn.Linear(latent_dim, fft_len // fft_scaling)
        self.decoder_reduce = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(1, channel_sz, 1, 1, 0),
        )
        self.decoder_reduce.apply(init_weights)
        env = [
            nn.Sequential(
                nn.GELU(),
                nn.ConvTranspose1d(channel_sz, channel_sz, 1, 1, 0),
            )
        ] + [
            nn.Sequential(
                nn.GELU(),
                nn.Linear(fft_len // (2 ** n), fft_len // (2 ** n)),
                nn.GELU(),
                nn.ConvTranspose1d(channel_sz, channel_sz, 4, 2, 1),
            )
            for n in range(levels, 0, -1)
        ]
        self.envelope_decoder = nn.Sequential(*env)
        self.envelope_decoder.apply(init_weights)
        self.decoder_output = nn.Conv1d(channel_sz, in_channels, 1, 1, 0)

    def encode(self, input: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.envelope_encoder(input)
        result = self.encoder_reduce(result).squeeze(1)

        return self.fc_z(result)

    def decode(self, z: Tensor) -> Tensor:
        result = self.z_fc(z).unsqueeze(1)
        result = self.decoder_reduce(result)
        result = self.envelope_decoder(result)
        return self.decoder_output(result)

    def forward(self, inp: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(inp)
        return [self.decode(z), inp, z]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        batch_size = args[1].size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.reg_weight / max(1, bias_corr)

        recons_loss = tf.mse_loss(args[0], args[1])

        mmd_loss = self.compute_mmd(args[2], reg_weight)

        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'MMD': mmd_loss}

    def compute_kernel(self,
                       x1: Tensor,
                       x2: Tensor) -> Tensor:
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2)  # Make it into a column tensor
        x2 = x2.unsqueeze(-3)  # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result

    def compute_rbf(self,
                    x1: Tensor,
                    x2: Tensor,
                    eps: float = 1e-7) -> Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var

        return torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))

    def compute_inv_mult_quad(self,
                              x1: Tensor,
                              x2: Tensor,
                              eps: float = 1e-7) -> Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = C / (C + abs(x_1 - x_2)**2)
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))

        return kernel.sum() - kernel.diag().sum()

    def compute_mmd(self, z: Tensor, reg_weight: float) -> Tensor:
        # Sample from prior (Gaussian) distribution
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        return (
                reg_weight * prior_z__kernel.mean()
                + reg_weight * z__kernel.mean()
                - 2 * reg_weight * priorz_z__kernel.mean()
        )

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        return self.decode(z)

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


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
        levels = 4
        fft_scaling = 2 ** levels

        # Encoder
        self.encoder_inflate = nn.Conv1d(in_channels, channel_sz, 3, 1, 1)
        self.encoder_reduce = nn.ModuleList()
        self.encoder_conv = nn.ModuleList()
        self.encoder_attention = nn.ModuleList()
        for n in range(levels):
            ch0 = channel_sz * (2**n)
            ch1 = channel_sz * (2**(n + 1))
            self.encoder_reduce.append(nn.Sequential(
                nn.BatchNorm1d(ch0),
                nn.Conv1d(ch0, ch1, 4, 2, 1),
                nn.GELU(),
            ))
            self.encoder_conv.append(nn.Sequential(
                Block1d(ch0),
                Block1d(ch0),
                Block1d(ch0),
            ))
            self.encoder_attention.append(nn.Sequential(
                Block1d(ch0),
                nn.Softmax(dim=1),
            ))
        self.encoder_squash = nn.Sequential(
            nn.Conv1d(channel_sz * (2**levels), 1, 3, 1, 1),
            nn.GELU(),
        )
        self.fc_z = nn.Sequential(
            nn.Linear(fft_len // fft_scaling, latent_dim),
        )

        # Decoder
        self.z_fc = nn.Linear(latent_dim, fft_len // fft_scaling)
        self.decoder_inflate = nn.Conv1d(1, channel_sz * (2**levels), 3, 1, 1)
        self.decoder_reduce = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()
        self.decoder_attention = nn.ModuleList()
        for n in range(levels, 0, -1):
            ch0 = channel_sz * (2**n)
            ch1 = channel_sz * (2 ** (n - 1))
            self.decoder_reduce.append(nn.Sequential(
                nn.BatchNorm1d(ch0),
                nn.ConvTranspose1d(ch0, ch1, 4, 2, 1),
                nn.GELU(),
            ))
            self.decoder_conv.append(nn.Sequential(
                Block1d(ch0, transpose=True),
                Block1d(ch0, transpose=True),
                Block1d(ch0, transpose=True),
            ))
            self.decoder_attention.append(nn.Sequential(
                Block1d(ch0, transpose=True),
                nn.Softmax(dim=1),
            ))
        self.decoder_output = nn.Conv1d(channel_sz, in_channels, 3, 1, 1)

        self.loss_function = nn.MSELoss()
        self.example_input_array = torch.randn((1, 2, self.fft_len))

    def encode(self, inp: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param inp: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        inp = self.encoder_inflate(inp)
        for att, conv, red in zip(self.encoder_attention, self.encoder_conv, self.encoder_reduce):
            inp = red(conv(inp) * att(inp))
        inp = self.encoder_squash(inp).squeeze(1)
        return self.fc_z(inp)

    def decode(self, z: Tensor) -> Tensor:
        result = self.z_fc(z).unsqueeze(1)
        result = self.decoder_inflate(result)
        for att, conv, red in zip(self.decoder_attention, self.decoder_conv, self.decoder_reduce):
            result = red(conv(result) * att(result))
        return self.decoder_output(result)

    def forward(self, inp: Tensor, **kwargs) -> Tensor:
        z = self.encode(inp)
        return self.decode(z)

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero and self.logger:
            self.logger.log_graph(self, self.example_input_array)

    def training_step(self, batch, batch_idx):
        return self.train_val_get(batch, batch_idx)

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

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'],
                                betas=self.params['betas'],
                                eps=1e-7)
        optims = [optimizer]
        if self.params['scheduler_gamma'] is None:
            return optims
        scheduler = optim.lr_scheduler.StepLR(optims[0], step_size=self.params['step_size'],
                                              gamma=self.params['scheduler_gamma'])
        scheds = [scheduler]

        return optims, scheds

    def train_val_get(self, batch, batch_idx, kind='train'):
        img, img = batch
        self.automatic_optimization = True

        results = self.forward(img)
        train_loss = self.loss_function(results, img)

        self.log_dict({f'{kind}_loss': train_loss}, sync_dist=True, prog_bar=True)
        return train_loss
