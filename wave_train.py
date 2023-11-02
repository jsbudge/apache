import numpy as np
from simulib.simulation_functions import genPulse, findPowerOf2, db
# import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import rayleigh
from data_converter.SDRParsing import SDRParse, load
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pickle
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
import yaml
from glob import glob
from torchvision import transforms
from pathlib import Path
from dataloaders import CovDataModule, WaveDataModule
from experiment import VAExperiment, GeneratorExperiment
from models import BetaVAE, InfoVAE, WAE_MMD, init_weights
from waveform_model import GeneratorModel

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808


def upsample(val, fac=8):
    upval = np.zeros(len(val) * fac, dtype=np.complex128)
    upval[:len(val) // 2] = val[:len(val) // 2]
    upval[-len(val) // 2:] = val[-len(val) // 2:]
    return upval


'''def outBeamTime(theta_az, theta_el):
    return (np.pi ** 2 * wheel_height_m - 8 * np.pi * blade_chord_m * np.tan(theta_el) -
            4 * wheel_height_m * theta_az) / (8 * np.pi * wheel_height_m * rotor_velocity_rad_s)'''


def getRange(alt, theta_el):
    return alt * np.sin(theta_el) * 2 / c0


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed_everything(43, workers=True)

    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    bin_bw = int(config['settings']['bandwidth'] // (fs / 32768))
    bin_bw += 1 if bin_bw % 2 != 0 else 0

    franges = np.linspace(config['perf_params']['vehicle_slant_range_min'],
                          config['perf_params']['vehicle_slant_range_max'], 1000) * 2 / c0
    nrange = franges[0]
    pulse_length = (nrange - 1 / TAC) * config['settings']['plp']
    duty_cycle_time_s = pulse_length + franges
    nr = int(pulse_length * fs)

    # Get the VAE set up
    if config['exp_params']['model_type'] == 'InfoVAE':
        vae_mdl = InfoVAE(**config['model_params'])
    elif config['exp_params']['model_type'] == 'WAE_MMD':
        vae_mdl = WAE_MMD(**config['model_params'])
    else:
        vae_mdl = BetaVAE(**config['model_params'])
    vae_mdl.load_state_dict(torch.load('./model/inference_model.state'))
    vae_mdl.eval()  # Set to inference mode
    # vae_mdl.to(device)  # Move to GPU

    wave_mdl = GeneratorModel(bin_bw=bin_bw, clutter_latent_size=config['model_params']['latent_dim'],
                              target_latent_size=config['model_params']['latent_dim'], n_ants=1)

    data = WaveDataModule(vae_model=vae_mdl, device=device, **config["dataset_params"])
    data.setup()

    experiment = GeneratorExperiment(wave_mdl, config['exp_params'])
    logger = loggers.TensorBoardLogger(config['train_params']['log_dir'],
                                       name=f"WaveModel")
    trainer = Trainer(logger=logger, max_epochs=config['train_params']['max_epochs'],
                      log_every_n_steps=config['exp_params']['log_epoch'],
                      strategy='ddp', deterministic=True,
                      callbacks=[EarlyStopping(monitor='loss', patience=50, check_finite=True)])

    print(f"======= Training =======")
    trainer.fit(experiment, datamodule=data)

    wave_mdl.eval()



    # Run some plots for an idea of what's going on
    '''freqs = np.fft.fftshift(np.fft.fftfreq(dec_fftsz, 1 / fs))
    plt.figure(f'Waveform PSD - run {run}')
    w0 = np.fft.fftshift(waveform_plot[0, :, 0])
    plt.plot(freqs, w0)
    w1 = np.fft.fftshift(waveform_plot[0, :, 1])
    plt.plot(freqs, w1)
    targ = np.fft.fftshift(targets[0, :, 0])
    plt.plot(freqs, db(targ), linestyle='--')
    clut = np.fft.fftshift(targets[0, :, 1])
    plt.plot(freqs, db(clut), linestyle=':')
    plt.legend(['Waveform 1', 'Waveform 2', 'Target', 'Clutter'])
    plt.ylabel('Relative Power (dB)')
    plt.xlabel('Freq (Hz)')

    # Save the model structure out to a PNG
    # plot_model(mdl, to_file='./mdl_plot.png', show_shapes=True)
    # waveforms = np.fft.fftshift(waveforms, axes=2)
    plt.figure(f'Autocorrelation - run {run}')
    linear = np.fft.fft(
        genPulse(np.linspace(0, 1, 10),
                 np.linspace(0, 1, 10), nr, fs, fc, bandwidth), dec_fftsz)
    inp_wave = waveforms[0, :, 0] * waveforms[0, :, 0].conj()
    autocorr1 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
    inp_wave = waveforms[0, :, 1] * waveforms[0, :, 1].conj()
    autocorr2 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
    inp_wave = waveforms[0, :, 0] * waveforms[0, :, 1].conj()
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

    plt.figure(f'Time Series - run {run}')
    plot_t = np.arange(dec_fftsz) / fs
    plt.plot(plot_t, np.fft.ifft(np.fft.fftshift(waveforms[0, :, 0])).real)
    plt.plot(plot_t, np.fft.ifft(np.fft.fftshift(waveforms[0, :, 1])).real)
    plt.legend(['Waveform 1', 'Waveform 2'])
    plt.xlabel('Time')

    plt.figure('Training')
    plt.subplot(2, 1, 1)
    plt.plot(total_hist[:3, :].T)
    plt.vlines(np.cumsum(hist_lines) - 1, np.min(total_hist), np.max(total_hist), colors='red', linestyles='--',
               linewidth=.5)
    plt.ylabel('Loss')
    plt.xlabel('Training Epoch')
    plt.subplot(2, 1, 2)
    plt.plot(total_hist[3:, :].T)
    plt.vlines(np.cumsum(hist_lines) - 1,
               np.min(total_hist[3:, :]), np.max(total_hist[3:, :]), colors='red', linestyles='--',
               linewidth=.5)
    plt.ylabel('Loss')
    plt.xlabel('Training Epoch')
    plt.show()'''