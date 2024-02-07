import numpy as np
from simulib.simulation_functions import genPulse, findPowerOf2, db
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import plotly.io as pio
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml
from dataloaders import WaveDataModule
from experiment import GeneratorExperiment
from models import BetaVAE, InfoVAE, WAE_MMD
from waveform_model import GeneratorModel, init_weights

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


def upsample(val, fac=8):
    upval = np.zeros(len(val) * fac, dtype=np.complex128)
    upval[:len(val) // 2] = val[:len(val) // 2]
    upval[-len(val) // 2:] = val[-len(val) // 2:]
    return upval


'''def outBeamTime(theta_az, theta_el):
    return (np.pi ** 2 * wheel_height_m - 8 * np.pi * blade_chord_m * np.tan(theta_el) -
            4 * wheel_height_m * theta_az) / (8 * np.pi * wheel_height_m * rotor_velocity_rad_s)'''


def normalize(data):
    return data / np.expand_dims(np.sqrt(np.sum(data * data.conj(), axis=-1).real), axis=len(data.shape) - 1)


def getRange(alt, theta_el):
    return alt * np.sin(theta_el) * 2 / c0


if __name__ == '__main__':
    force_cudnn_initialization()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.empty_cache()

    seed_everything(123, workers=True)

    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    fft_len = config['generate_data_settings']['fft_sz']
    nr = 5000  # int((config['perf_params']['vehicle_slant_range_min'] * 2 / c0 - 1 / TAC) * fs)

    # Get the VAE set up
    print('Setting up model...')
    if config['exp_params']['model_type'] == 'InfoVAE':
        vae_mdl = InfoVAE(**config['model_params'])
    elif config['exp_params']['model_type'] == 'WAE_MMD':
        vae_mdl = WAE_MMD(**config['model_params'])
    else:
        vae_mdl = BetaVAE(**config['model_params'])
    vae_mdl.load_state_dict(torch.load('./model/inference_model.state'))
    vae_mdl.eval()  # Set to inference mode
    # vae_mdl.to(device)  # Move to GPU

    print('Setting up data generator...')
    wave_mdl = GeneratorModel(fft_sz=fft_len,
                              stft_win_sz=config['settings']['stft_win_sz'],
                              clutter_latent_size=config['model_params']['latent_dim'],
                              target_latent_size=config['model_params']['latent_dim'], n_ants=2)

    # Since these are dependent on apache params, we set them up here instead of in the yaml file
    config['dataset_params']['max_pulse_length'] = nr
    config['dataset_params']['min_pulse_length'] = config['settings']['stft_win_sz'] + 64

    data = WaveDataModule(vae_model=vae_mdl, device=device, **config["dataset_params"])
    data.setup()

    vae_mdl.to('cpu')

    print('Setting up experiment...')
    experiment = GeneratorExperiment(wave_mdl, config['wave_exp_params'])
    logger = loggers.TensorBoardLogger(config['train_params']['log_dir'],
                                       name="WaveModel")
    trainer = Trainer(logger=logger, max_epochs=config['train_params']['max_epochs'],
                      log_every_n_steps=config['exp_params']['log_epoch'],
                      strategy='ddp', gradient_clip_val=.5, callbacks=
                      [EarlyStopping(monitor='loss', patience=config['wave_exp_params']['patience'],
                                     check_finite=True), StochasticWeightAveraging(swa_lrs=1e-2)])

    print("======= Training =======")
    trainer.fit(experiment, datamodule=data)

    if trainer.global_rank == 0:
        with torch.no_grad():
            wave_mdl.to(device)
            wave_mdl.eval()

            cc, tc, cs, ts, _ = next(iter(data.train_dataloader()))
            cc = cc.to(device)
            tc = tc.to(device)
            cs = cs.to(device)
            ts = ts.to(device)

            waves = wave_mdl.getWaveform(cc, tc, nr).cpu().data.numpy()
            print('Loaded waveforms...')

            clutter = cs.data.numpy()
            clutter = normalize(clutter[:, :, 0] + 1j * clutter[:, :, 1])
            targets = ts.data.numpy()
            targets = normalize(targets[:, :, 0] + 1j * targets[:, :, 1])
            print('Loaded clutter and target data...')

            # Run some plots for an idea of what's going on
            freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, 1 / fs))
            plt.figure('Waveform PSD')
            plt.plot(freqs, db(np.fft.fftshift(waves[0, 0])))
            plt.plot(freqs, db(np.fft.fftshift(waves[0, 1])))
            plt.plot(freqs, db(np.fft.fftshift(targets[0])), linestyle='--', linewidth=.3)
            plt.plot(freqs, db(np.fft.fftshift(clutter[0])), linestyle=':', linewidth=.3)
            plt.legend(['Waveform 1', 'Waveform 2', 'Target', 'Clutter'])
            plt.ylabel('Relative Power (dB)')
            plt.xlabel('Freq (Hz)')

            # Save the model structure out to a PNG
            # plot_model(mdl, to_file='./mdl_plot.png', show_shapes=True)
            # waveforms = np.fft.fftshift(waveforms, axes=2)
            plt.figure('Autocorrelation')
            linear = np.fft.fft(
                genPulse(np.linspace(0, 1, 10),
                         np.linspace(0, 1, 10), nr, fs, config['settings']['fc'],
                         config['settings']['bandwidth']), fft_len)
            linear = linear / sum(linear * linear.conj())  # Unit energy
            inp_wave = waves[0, 0] * waves[0, 0].conj()
            autocorr1 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
            inp_wave = waves[0, 1] * waves[0, 1].conj()
            autocorr2 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
            inp_wave = waves[0, 0] * waves[0, 1].conj()
            autocorrcr = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
            perf_autocorr = np.fft.fftshift(db(np.fft.ifft(upsample(linear * linear.conj()))))
            lags = np.arange(len(autocorr1)) - len(autocorr1) // 2
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     autocorr1[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr1.max())
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     autocorr2[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr2.max())
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     autocorrcr[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr1.max())
            plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                     perf_autocorr[len(lags) // 2 - 200:len(lags) // 2 + 200] - perf_autocorr.max(),
                     linestyle='--')
            plt.legend(['Waveform 1', 'Waveform 2', 'Cross Correlation', 'Linear Chirp'])
            plt.xlabel('Lag')

            waves = wave_mdl.getWaveform(cc, tc, nr, scale=True).cpu().data.numpy()

            plt.figure('Time Series')
            wave1 = waves.copy()
            plot_t = np.arange(nr) / fs
            plt.plot(plot_t, np.fft.ifft(wave1[0, 0]).real[:nr])
            plt.plot(plot_t, np.fft.ifft(wave1[0, 1]).real[:nr])
            plt.legend(['Waveform 1', 'Waveform 2'])
            plt.xlabel('Time')

            wave_t = np.fft.ifft(waves[0, 0])[:nr]
            freq_stft, t_stft, wave_stft = stft(wave_t, return_onesided=False, fs=2e9, window=np.ones(256))
            plt.figure('Wave STFT')
            plt.pcolormesh(t_stft, np.fft.fftshift(freq_stft), np.fft.fftshift(db(wave_stft), axes=0))
            plt.ylabel('Freq')
            plt.xlabel('Time')
            plt.colorbar()

            plt.show()

        if trainer.is_global_zero:
            try:
                wave_mdl.save('./model')
                print('Model saved to disk.')
            except Exception as e:
                print(f'Model not saved: {e}')
