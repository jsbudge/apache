import pickle
from pathlib import Path

from config import get_config
from utils import upsample, normalize, fs, narrow_band
import numpy as np
from simulib.simulation_functions import genPulse, db, findPowerOf2
import matplotlib.pyplot as plt
from scipy.signal import stft
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
import yaml
from dataloaders import WaveDataModule
from experiment import GeneratorExperiment
from models import TargetEmbedding, load
from waveform_model import GeneratorModel
from os import listdir


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
    # seed_everything(17, workers=True)

    config = get_config('wave_exp', './vae_config.yaml')
    target_config = get_config('target_exp', './vae_config.yaml')

    fft_len = config.fft_len
    nr = 5000  # int((config['perf_params']['vehicle_slant_range_min'] * 2 / c0 - 1 / TAC) * fs)
    # Since these are dependent on apache params, we set them up here instead of in the yaml file
    print('Setting up data generator...')
    config.dataset_params['max_pulse_length'] = nr
    config.dataset_params['min_pulse_length'] = 1000

    data = WaveDataModule(clutter_latent_dim=config.clutter_latent_size,
                          target_latent_dim=config.target_latent_size, device=device,
                          fft_sz=fft_len,
                          **config.dataset_params)
    data.setup()

    print('Setting up embedding model...')
    embedding = TargetEmbedding.load_from_checkpoint(f'{target_config.weights_path}/{target_config.model_name}.ckpt', config=target_config, strict=False)

    print('Initializing wavemodel...')
    if config.warm_start:
        wave_mdl = GeneratorModel.load_from_checkpoint(f'{config.weights_path}/{config.model_name}.ckpt', config=config, embedding=embedding, strict=False)
    else:
        wave_mdl = GeneratorModel(config=config, embedding=embedding)
    logger = loggers.TensorBoardLogger(config.log_dir,
                                       name=config.model_name, log_graph=True)
    expected_lr = max((config.lr * config.scheduler_gamma ** (config.max_epochs * config.swa_start)), 1e-9)
    trainer = Trainer(logger=logger, max_epochs=config.max_epochs, num_sanity_val_steps=0, default_root_dir=config.weights_path,
                      log_every_n_steps=config.log_epoch, devices=[1], callbacks=
                      [EarlyStopping(monitor='target_loss', patience=config.patience, check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr, swa_epoch_start=config.swa_start),
                       ModelCheckpoint(monitor='loss_epoch')])

    print("======= Training =======")
    try:
        trainer.fit(wave_mdl, datamodule=data)
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
            wave_mdl.to(device)
            wave_mdl.eval()

            cc, tc, ts, plength = next(iter(data.train_dataloader()))
            cc = cc.to(device)
            ts = ts.to(device)
            plength = plength.to(device)

            nn_output = wave_mdl([cc, ts, plength])
            # nn_numpy = nn_output[0, 0, ...].cpu().data.numpy()

            waves = wave_mdl.getWaveform(nn_output=nn_output).cpu().data.numpy()
            # waves = save_waves
            # waves = np.fft.fft(np.fft.ifft(waves, axis=2)[:, :, :nr], fft_len, axis=2)
            print('Loaded waveforms...')

            clutter = cc.cpu().data.numpy()
            clutter = normalize(clutter[:, 0, 0, :] + 1j * clutter[:, 0, 1, :])
            targets = tc.cpu().data.numpy()
            targets = normalize(targets[:, 0, :] + 1j * targets[:, 1, :])
            print('Loaded clutter and target data...')

            # Run some plots for an idea of what's going on
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, 1 / fs))
            plt.figure('Waveform PSD')
            plt.plot(freqs, db(np.fft.fftshift(waves[0, 0])))
            if wave_mdl.n_ants > 1:
                plt.plot(freqs, db(np.fft.fftshift(waves[0, 1])))
            plt.plot(freqs, db(targets[0]), linestyle='--', linewidth=.3)
            plt.plot(freqs, db(clutter[0]), linestyle=':', linewidth=.3)
            if wave_mdl.n_ants > 1:
                plt.legend(['Waveform 1', 'Waveform 2', 'Target', 'Clutter'])
            else:
                plt.legend(['Waveform', 'Target', 'Clutter'])
            plt.ylabel('Relative Power (dB)')
            plt.xlabel('Freq (Hz)')

            if wave_mdl.n_ants > 1:
                clutter_corr = np.fft.ifft(
                    np.fft.fftshift(clutter[0]) * waves[0, 0] * waves[0, 0].conj() + np.fft.fftshift(clutter[0]) * waves[
                        0, 1] * waves[0, 1].conj())
                target_corr = np.fft.ifft(
                    np.fft.fftshift(targets[0]) * waves[0, 0] * waves[0, 0].conj() + np.fft.fftshift(targets[0]) * waves[
                        0, 1] * waves[0, 1].conj())
            else:
                clutter_corr = np.fft.ifft(
                    np.fft.fftshift(clutter[0]) * waves[0, 0] * waves[0, 0].conj())
                target_corr = np.fft.ifft(
                    np.fft.fftshift(targets[0]) * waves[0, 0] * waves[0, 0].conj())
            plt.figure('MIMO Correlations')
            plt.plot(db(clutter_corr))
            plt.plot(db(target_corr))
            plt.legend(['Clutter', 'Target'])
            plt.xlabel('Lag')
            plt.ylabel('Power (dB)')

            plt.figure()
            plt.plot(db(np.fft.ifft(np.fft.fftshift(clutter[0]) * np.fft.fftshift(clutter[0]).conj())))

            # Save the model structure out to a PNG
            # plot_model(mdl, to_file='./mdl_plot.png', show_shapes=True)
            # waveforms = np.fft.fftshift(waveforms, axes=2)
            plt.figure('Autocorrelation')
            linear = np.fft.fft(
                genPulse(np.linspace(0, 1, 10),
                         np.linspace(0, 1, 10), nr, fs, config.fc,
                         config.bandwidth), fft_len)
            linear = linear / sum(linear * linear.conj())  # Unit energy
            inp_wave = waves[0, 0] * waves[0, 0].conj()
            autocorr1 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
            if wave_mdl.n_ants > 1:
                inp_wave = waves[0, 1] * waves[0, 1].conj()
                autocorr2 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
                inp_wave = waves[0, 0] * waves[0, 1].conj()
                autocorrcr = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
            perf_autocorr = np.fft.fftshift(db(np.fft.ifft(upsample(linear * linear.conj()))))
            lags = np.arange(len(autocorr1)) - len(autocorr1) // 2
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     autocorr1[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr1.max())
            if wave_mdl.n_ants > 1:
                plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                         autocorr2[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr2.max())
                plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                         autocorrcr[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr1.max())
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     perf_autocorr[len(lags) // 2 - 200:len(lags) // 2 + 200] - perf_autocorr.max(),
                     linestyle='--')
            if wave_mdl.n_ants > 1:
                plt.legend(['Waveform 1', 'Waveform 2', 'Cross Correlation', 'Linear Chirp'])
            else:
                plt.legend(['Waveform', 'Linear Chirp'])
            plt.xlabel('Lag')

            plt.figure('Time Series')
            wave1 = waves.copy()
            plot_t = np.arange(nr) / fs
            plt.plot(plot_t, np.fft.ifft(wave1[0, 0]).real[:nr])
            if wave_mdl.n_ants > 1:
                plt.plot(plot_t, np.fft.ifft(wave1[0, 1]).real[:nr])
                plt.legend(['Waveform 1', 'Waveform 2'])
            plt.xlabel('Time')

            wave_t = np.fft.ifft(waves[0, 0])[:nr]
            win = torch.windows.hann(256).data.numpy()
            freq_stft, t_stft, wave_stft = stft(wave_t, return_onesided=False, window=win, fs=2e9)
            plt.figure('Wave STFT')
            plt.pcolormesh(t_stft, np.fft.fftshift(freq_stft), np.fft.fftshift(db(wave_stft), axes=0))
            plt.ylabel('Freq')
            plt.xlabel('Time')
            plt.colorbar()

'''wave = np.fft.fft(np.fft.ifft(waves[0, 0]), 32768)
rp_back = np.load('/home/jeff/repo/simulib/scripts/single_rp_back.npy')
rp_old_mf = np.load('/home/jeff/repo/simulib/scripts/single_mf_pulse_back.npy')
rp_fft = np.fft.fft(rp_back[0, 0], 32768)
rp_imf = rp_fft * wave * wave.conj()
rp_mf = np.zeros(32768 * 8, dtype=np.complex128)
rp_mf[:16384] = rp_imf[:16384]
rp_mf[-16384:] = rp_imf[-16384:]
rp_mf = np.fft.ifft(rp_mf)[:rp_old_mf.shape[1]]
plt.figure()
plt.plot(db(rp_mf))
plt.plot(db(rp_old_mf[0]))'''

waf, tau, theta = narrow_band(np.fft.ifft(waves[0, 0]), np.arange(512) - 256)

plt.figure('Ambiguity Function')
plt.imshow(db(waf[4096 - 256:4096 + 256, :]))
# plt.clim([-100, -76])
plt.axis('tight')
plt.show()