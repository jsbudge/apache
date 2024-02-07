import contextlib
import os
import torch
from torch import optim
from models import BaseVAE
from typing import TypeVar
import pytorch_lightning as pl
import torchvision.utils as vutils

# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')


class VAExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

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

            try:
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
            except Warning:
                pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims


class AExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: pl.LightningModule,
                 params: dict) -> None:
        super(AExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims


class GeneratorExperiment(pl.LightningModule):

    def __init__(self,
                 wave_model: pl.LightningModule,
                 params: dict) -> None:
        super(GeneratorExperiment, self).__init__()

        self.model = wave_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.optim_path = []
        if 'retain_first_backpass' in self.params:
            self.hold_graph = self.params['retain_first_backpass']

    def forward(self, clutter: Tensor, target: Tensor, pulse_length: int) -> Tensor:
        return self.model(clutter, target, pulse_length)

    def training_step(self, batch, batch_idx):
        train_loss = self.train_val_get(batch, batch_idx)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx)

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero and not self.params['is_tuning'] and self.params['save_model']:
            self.model.save('./model')
            print('Model saved to disk.')

    def on_train_epoch_end(self) -> None:
        if self.trainer.is_global_zero and not self.params['is_tuning'] and self.params['loss_landscape']:
            self.optim_path.append(self.model.get_flat_params())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'],
                               eps=1e-7)
        optims = [optimizer]
        if self.params['scheduler_gamma'] is None:
            return optims
        scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                     gamma=self.params['scheduler_gamma'])
        scheds = [scheduler]

        return optims, scheds

    def train_val_get(self, batch, batch_idx):
        clutter_cov, target_cov, clutter_spec, target_spec, pulse_length = batch
        self.curr_device = clutter_cov.device
        self.automatic_optimization = True

        results = self.forward(clutter_cov, target_cov, pulse_length=pulse_length)
        train_loss = self.model.loss_function(results, clutter_spec, target_spec)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True, prog_bar=True)
        return train_loss


class RCSExperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: pl.LightningModule,
                 params: dict) -> None:
        super(RCSExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        with contextlib.suppress(Exception):
            self.hold_graph = self.params['retain_first_backpass']

    def forward(self, optical_data: Tensor, sar_data: Tensor, pose_data: Tensor) -> Tensor:
        return self.model(optical_data, sar_data, pose_data)

    def training_step(self, batch, batch_idx):
        opt_img, sar_img, pose = batch
        self.curr_device = opt_img.device
        self.automatic_optimization = True

        results = self.forward(opt_img, sar_img, pose)
        train_loss = self.model.loss_function(results, sar_img)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        opt_img, sar_img, pose = batch
        self.curr_device = opt_img.device
        self.automatic_optimization = True

        results = self.forward(opt_img, sar_img, pose)
        val_loss = self.model.loss_function(results, sar_img)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

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
