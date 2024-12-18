import torch
from torch import optim, Tensor
from typing import Union, List
import pytorch_lightning as pl
from torch.nn import functional as nn_func
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
        # self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()
        self.log_dict(train_loss, sync_dist=True,
                      prog_bar=True, rank_zero_only=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self.log_dict(self.train_val_get(batch, batch_idx), sync_dist=True, prog_bar=True,
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
            '''cc, tc, cs, ts, plength, dset_bandwidth = next(iter(self.trainer.datamodule.test_dataloader()))

            nn_output = self.model([cc.to(self.device), tc.to(self.device), plength.to(self.device),
                                  dset_bandwidth.to(self.device)])

            waves = self.model.getWaveform(nn_output=nn_output).cpu().data.numpy()
            print('Loaded waveforms...')

            clutter = cs.cpu().data.numpy()
            clutter = normalize(clutter[:, 0, :] + 1j * clutter[:, 1, :])
            targets = ts.cpu().data.numpy()
            targets = normalize(targets[:, 0, :] + 1j * targets[:, 1, :])
            print('Loaded clutter and target data...')

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

    def on_validation_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        else:
            sch.step()
        self.log('LR', sch.get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def on_train_epoch_end(self) -> None:
        if self.trainer.is_global_zero and not self.params['is_tuning'] and self.params['loss_landscape']:
            self.optim_path.append(self.model.get_flat_params())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.params['LR'],
                                      weight_decay=self.params['weight_decay'],
                                      betas=self.params['betas'],
                                      eps=1e-7)
        if self.params['scheduler_gamma'] is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params['scheduler_gamma'],
                                                           verbose=True)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx):
        clutter_spec, target_spec, target_enc, pulse_length = batch

        results = self.forward([clutter_spec, target_enc, pulse_length])
        train_loss = self.model.loss_function(results, clutter_spec, target_spec, target_enc, pulse_length)

        train_loss['loss'] = torch.sqrt(torch.abs(
            train_loss['sidelobe_loss'] * (1 + train_loss['target_loss'] + train_loss['ortho_loss'])))
        return train_loss
