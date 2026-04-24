from config import get_config
from utils import upsample, fs, narrow_band, getMatchedFilter
import numpy as np
from simulib.simulation_functions import genPulse, db
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.signal.windows import taylor
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloaders import WaveDataModule
from models import ClutterTransformer


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))



if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    force_cudnn_initialization()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.empty_cache()

    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(107, workers=True)

    config = get_config('transformer_exp', './vae_config.yaml')

    nr = 5000  # int((config['perf_params']['vehicle_slant_range_min'] * 2 / c0 - 1 / TAC) * fs)
    # Since these are dependent on apache params, we set them up here instead of in the yaml file
    print('Setting up data generator...')
    config.dataset_params['max_pulse_length'] = nr
    config.dataset_params['min_pulse_length'] = 1000

    data = WaveDataModule(device=device, **config.dataset_params)
    data.setup()

    print('Initializing encoder...')
    if config.warm_start:
        transformer = ClutterTransformer.load_from_checkpoint(f'{config.weights_path}/{config.model_name}.ckpt', config=config, strict=False)
    else:
        transformer = ClutterTransformer(**config.model_params, **config.training_params)
    logger = loggers.TensorBoardLogger(config.log_dir,
                                       name=config.model_name, log_graph=False)
    expected_lr = 1e-6
    if config.distributed:
        trainer = Trainer(logger=logger, max_epochs=config.max_epochs, num_sanity_val_steps=0, default_root_dir=config.weights_path,
                          log_every_n_steps=config.log_epoch, check_val_every_n_epoch=1000, devices=[0, 1], strategy='ddp', callbacks=
                          [EarlyStopping(monitor='train_rec_loss', patience=config.patience, check_finite=True),
                           ModelCheckpoint(monitor='train_rec_loss_epoch')])
    else:
        trainer = Trainer(logger=logger, max_epochs=config.max_epochs, num_sanity_val_steps=0,
                          default_root_dir=config.weights_path, check_val_every_n_epoch=1000,
                          log_every_n_steps=config.log_epoch, devices=[0], callbacks=
                          [EarlyStopping(monitor='train_rec_loss', patience=config.patience, check_finite=True),
                           ModelCheckpoint(monitor='train_rec_loss_epoch')])
    print("======= Training =======")
    try:
        trainer.fit(transformer, datamodule=data)
    except KeyboardInterrupt:
        if trainer.is_global_zero:
            print('Training interrupted.')
        else:
            print('adios!')
            exit(0)

    if trainer.global_rank == 0:
        if config.save_model:
            trainer.save_checkpoint(f'{config.weights_path}/{config.model_name}.ckpt')
            print('Checkpoint saved.')

        with torch.no_grad():
            transformer.to(device)
            transformer.eval()
            data_iter = iter(data.train_dataloader())

            seq, tseq, _, _, _, _, _, _ = next(data_iter)

            encoded_seq = transformer.encode(seq.to(transformer.device))
            rec_seq = transformer.decode(encoded_seq).data.cpu().numpy()[0]
            rec_seq = np.log10(abs(np.fft.ifft(rec_seq[:, 0] + 1j * rec_seq[:, 1])))

            encoded_tseq = transformer.encode(tseq.to(transformer.device))
            rec_tseq = transformer.decode(encoded_tseq).data.cpu().numpy()[0]
            rec_tseq = np.log10(abs(np.fft.ifft(rec_tseq[:, 0] + 1j * rec_tseq[:, 1])))

            plot_seq = seq.data.cpu().numpy()[0]
            plot_seq = np.log10(abs(np.fft.ifft(plot_seq[:, 0] + 1j * plot_seq[:, 1])))
            plot_tseq = tseq.data.cpu().numpy()[0]
            plot_tseq = np.log10(abs(np.fft.ifft(plot_tseq[:, 0] + 1j * plot_tseq[:, 1])))

            plt.figure('Clutter Sequence')
            plt.subplot(2, 2, 1)
            plt.title('Clutter Sequence')
            plt.plot(plot_seq[-1], label='Clutter Sequence')
            plt.plot(rec_seq[-1], label='Target Sequence')
            plt.subplot(2, 2, 2)
            plt.title('Error')
            plt.imshow(abs(plot_seq[:1] - rec_seq[:-1]))
            plt.axis('tight')
            plt.subplot(2, 2, 3)
            plt.title('Original')
            plt.imshow(plot_seq, label='Clutter Sequence')
            plt.axis('tight')
            plt.subplot(2, 2, 4)
            plt.title('Reconstruction')
            plt.imshow(rec_seq, label='Target Sequence')
            plt.axis('tight')

            plt.figure('Target Sequence')
            plt.subplot(2, 2, 1)
            plt.title('Clutter Sequence')
            plt.plot(plot_tseq[-1], label='Clutter Sequence')
            plt.plot(rec_tseq[-1], label='Target Sequence')
            plt.subplot(2, 2, 2)
            plt.title('Error')
            plt.imshow(abs(plot_tseq[1:] - rec_tseq[:-1]))
            plt.axis('tight')
            plt.subplot(2, 2, 3)
            plt.title('Original')
            plt.imshow(plot_tseq, label='Clutter Sequence')
            plt.axis('tight')
            plt.subplot(2, 2, 4)
            plt.title('Reconstruction')
            plt.imshow(rec_tseq, label='Target Sequence')
            plt.axis('tight')

            plt.figure('Encoded Sequences')
            plt.subplot(2, 2, 1)
            plt.title('Clutter Sequence')
            plt.plot(encoded_seq.data.cpu().numpy()[0, 1], label='Clutter Sequence')
            plt.subplot(2, 2, 2)
            plt.title('Target Sequence')
            plt.plot(encoded_tseq.data.cpu().numpy()[0, 1], label='Clutter Sequence')
            plt.subplot(2, 2, 3)
            plt.title('Overlay')
            plt.plot(encoded_seq.data.cpu().numpy()[0, 1], label='Clutter Sequence')
            plt.plot(encoded_tseq.data.cpu().numpy()[0, 1], label='Clutter Sequence')
            plt.subplot(2, 2, 4)
            plt.title('Differences')
            plt.plot(encoded_seq.data.cpu().numpy()[0, 1] - encoded_tseq.data.cpu().numpy()[0, 1], label='Clutter Sequence')