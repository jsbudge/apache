import contextlib
import os
import torch
from torch import optim, Tensor
from models import BaseVAE
from typing import TypeVar, Union, List
import pytorch_lightning as pl
from pathlib import Path
import torchvision.utils as vutils
from torch.nn import functional as nn_func
from simulib import db
import matplotlib.pyplot as plt
import numpy as np


def normalize(data):
    return data / np.expand_dims(np.sqrt(np.sum(data * data.conj(), axis=-1).real), axis=len(data.shape) - 1)


class FAMO:
    """
    Fast Adaptive Multitask Optimization.
    """
    prev_loss: float = 0.

    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            gamma: float = 0.01,  # the regularization coefficient
            w_lr: float = 0.025,  # the learning rate of the task logits
            max_norm: float = 1.0,  # the maximum gradient norm
    ):
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.device = device

    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses):
        self.prev_loss = losses
        z = nn_func.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        return (D.log() * z / c).sum()

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(nn_func.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()

    def backward(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
    ) -> Union[torch.Tensor, None]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        Returns
        -------
        Loss, extra outputs
        """
        loss = self.get_weighted_loss(losses=losses)
        loss.backward()
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return loss


class VAExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        if 'retain_first_backpass' in params:
            self.hold_graph = self.params['retain_first_backpass']

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device
        self.automatic_optimization = True

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True, prog_bar=True)

        return train_loss['loss']

    def on_fit_start(self) -> None:
        # Make sure the reconstruction paths are there if we're outputting images
        if self.params['output_images'] and self.trainer.is_global_zero:
            Path(f'{self.logger.log_dir}/Reconstructions').mkdir(parents=True, exist_ok=True)
            Path(f'{self.logger.log_dir}/Samples').mkdir(parents=True, exist_ok=True)
        if self.trainer.is_global_zero and self.logger:
            self.logger.log_graph(self, self.model.example_input_array)

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        if self.params['output_images']:
            self.sample_images()
        if self.trainer.is_global_zero and not self.params['is_tuning']:
            torch.save(self.model.state_dict(), './model/inference_model.state')
            print('Model saved to disk.')

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        #         test_input, test_label = batch
        if self.current_epoch % self.params['log_epoch'] == 0:
            recons = self.model.generate(test_input, labels=test_label)
            # A little finagling to get our 2-channel data into a 3-channel image format
            recons = torch.log10(torch.abs(torch.complex(recons[:, 0, :, :], recons[:, 1, :, :]))).unsqueeze(1).repeat(
                1, 3, 1, 1)
            vutils.save_image(recons.data,
                              os.path.join(self.logger.log_dir,
                                           "Reconstructions",
                                           f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)

            with contextlib.suppress(Warning):
                samples = self.model.sample(144,
                                            self.curr_device,
                                            labels=test_label)
                # A little finagling to get our 2-channel data into a 3-channel image format
                samples = torch.log10(torch.abs(torch.complex(samples[:, 0, :, :], samples[:, 1, :, :]))).unsqueeze(
                    1).repeat(1, 3, 1, 1)
                vutils.save_image(samples.cpu().data,
                                  os.path.join(self.logger.log_dir,
                                               "Samples",
                                               f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                                  normalize=True,
                                  nrow=12)

    def configure_optimizers(self):

        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims = [optimizer]
        # Check if more than 1 optimizer is required (Used for adversarial training)
        with contextlib.suppress(Exception):
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                with contextlib.suppress(Exception):
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                return optims, scheds
        except Exception:
            return optims


class GeneratorExperiment(pl.LightningModule):

    def __init__(self,
                 wave_model: pl.LightningModule,
                 params: dict) -> None:
        super(GeneratorExperiment, self).__init__()

        self.model = wave_model
        self.params = params
        self.hold_graph = False
        self.optim_path = []
        # self.famo = FAMO(n_tasks=4, device='cpu')
        if 'retain_first_backpass' in self.params:
            self.hold_graph = self.params['retain_first_backpass']

        # Set automatic optimization to false for FAMO
        self.automatic_optimization = False
        self.example_input_array = wave_model.example_input_array

    def forward(self, inp: list) -> Tensor:
        return self.model(inp)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss['loss'])
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True,
                      prog_bar=True, rank_zero_only=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        losses = self.train_val_get(batch, batch_idx)
        self.log_dict({key: val.item() for key, val in losses.items()}, sync_dist=True, prog_bar=True,
                      rank_zero_only=True)

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero and not self.params['is_tuning'] and self.params['save_model']:
            self.model.save('./model')
            print('Model saved to disk.')
            '''if self.current_epoch % 5 == 0:
                # Log an image to get an idea of progress
                clutter_enc, target_enc, clutter_spec, target_spec, pulse_length = (
                    next(iter(self.trainer.val_dataloaders)))
                pulse_length[1:] = pulse_length[0]

                bandwidth = torch.ones(clutter_enc.shape[0], 1, device=self.device) * self.params['bandwidth']
                rec = self.forward([clutter_enc.to(self.device), target_enc.to(self.device),
                                    pulse_length.to(self.device), bandwidth.to(self.device)])
                waves = self.model.getWaveform(nn_output=rec).cpu().data.numpy()

                clutter = clutter_spec.cpu().data.numpy()
                clutter = normalize(clutter[:, 0, :] + 1j * clutter[:, 1, :])
                targets = target_spec.cpu().data.numpy()
                targets = normalize(targets[:, 0, :] + 1j * targets[:, 1, :])

                # Run some plots for an idea of what's going on
                freqs = np.fft.fftshift(np.fft.fftfreq(self.model.fft_sz, 1 / self.model.fs))
                fig = plt.figure('Waveform PSD')
                plt.plot(freqs, db(np.fft.fftshift(waves[0, 0])))
                plt.plot(freqs, db(np.fft.fftshift(waves[0, 1])))
                plt.plot(freqs, db(targets[0]), linestyle='--', linewidth=.3)
                plt.plot(freqs, db(clutter[0]), linestyle=':', linewidth=.3)
                plt.legend(['Waveform 1', 'Waveform 2', 'Target', 'Clutter'])
                plt.ylabel('Relative Power (dB)')
                plt.xlabel('Freq (Hz)')
                self.logger.experiment.add_figure('Waveforms', fig, self.current_epoch)'''

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["loss_epoch"])
            self.log('LR', sch.get_last_lr()[0], rank_zero_only=True)
        if self.trainer.is_global_zero and not self.params['is_tuning'] and self.params['loss_landscape']:
            self.optim_path.append(self.model.get_flat_params())

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'],
                                betas=self.params['betas'],
                                eps=1e-7)
        optims = [optimizer]
        if self.params['scheduler_gamma'] is None:
            return optims
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optims[0], cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'])
        scheds = [scheduler]

        return optims, scheds

    def train_val_get(self, batch, batch_idx):
        clutter_enc, target_enc, clutter_spec, target_spec, pulse_length, bandwidth = batch

        results = self.forward([clutter_enc, target_enc, pulse_length, bandwidth])
        train_loss = self.model.loss_function(results, clutter_spec, target_spec, bandwidth)

        train_loss['loss'] = torch.sqrt(torch.abs(
            train_loss['target_loss'] * (1 + train_loss['sidelobe_loss'] + train_loss['ortho_loss'])))
        return train_loss


class RCSExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: pl.LightningModule,
                 params: dict) -> None:
        super(RCSExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.hold_graph = False
        with contextlib.suppress(Exception):
            self.hold_graph = self.params['retain_first_backpass']

    def forward(self, optical_data: Tensor, pose_data: Tensor) -> Tensor:
        return self.model(optical_data, pose_data)

    def training_step(self, batch, batch_idx):
        opt_img, sar_img, pose = batch
        self.automatic_optimization = True

        results = self.forward(opt_img, pose)
        train_loss = self.model.loss_function(results, sar_img)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True, prog_bar=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        opt_img, sar_img, pose = batch
        self.automatic_optimization = True

        results = self.forward(opt_img, pose)
        val_loss = self.model.loss_function(results, sar_img)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True, prog_bar=True)

    def on_validation_end(self) -> None:
        if self.params['output_images']:
            self.sample_images()
        if self.trainer.is_global_zero and not self.params['is_tuning']:
            torch.save(self.model.state_dict(), './model/rcs_model.state')
            print('Model saved to disk.')

    def on_fit_start(self) -> None:
        # Make sure the reconstruction paths are there if we're outputting images
        if self.params['output_images'] and self.trainer.is_global_zero:
            Path(f'{self.logger.log_dir}/Reconstructions').mkdir(parents=True, exist_ok=True)
            Path(f'{self.logger.log_dir}/Samples').mkdir(parents=True, exist_ok=True)

    def sample_images(self):
        # Get sample reconstruction image
        opt_img, sar_img, pose = next(iter(self.trainer.datamodule.test_dataloader()))
        opt_img = opt_img.to(self.device)
        pose = pose.to(self.device)

        if self.current_epoch % self.params['log_epoch'] == 0:
            recons = self.model(opt_img, pose)
            # A little finagling to get our 1-channel data into a 3-channel image format
            recons = recons.repeat(1, 3, 1, 1)
            vutils.save_image(recons.data,
                              os.path.join(self.logger.log_dir,
                                           "Reconstructions",
                                           f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)

            with contextlib.suppress(Warning):
                # A little finagling to get our 1-channel data into a 3-channel image format
                samples = sar_img.repeat(1, 3, 1, 1)
                vutils.save_image(samples.data,
                                  os.path.join(self.logger.log_dir,
                                               "Samples",
                                               f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                                  normalize=True,
                                  nrow=12)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims = [optimizer]
        if self.params['scheduler_gamma'] is None:
            return optims
        scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                     gamma=self.params['scheduler_gamma'])
        scheds = [scheduler]

        return optims, scheds
