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

fft_sz = 32768
dec_factor = 8
n_ants = 2
l_sz = 512
num_filters = 32
n_sublayers = 3
n_epochs = 1000
n_runs = 3
save_model_bool = False
save_histogram = True


def upsample(val, fac=8):
    upval = np.zeros(len(val) * fac, dtype=np.complex128)
    upval[:len(val) // 2] = val[:len(val) // 2]
    upval[-len(val) // 2:] = val[-len(val) // 2:]
    return upval


def outBeamTime(theta_az, theta_el):
    return (np.pi ** 2 * wheel_height_m - 8 * np.pi * blade_chord_m * np.tan(theta_el) -
            4 * wheel_height_m * theta_az) / (8 * np.pi * wheel_height_m * rotor_velocity_rad_s)


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
    vae_mdl.to(device)  # Move to GPU

    wave_mdl = GeneratorModel(bin_bw=bin_bw, clutter_latent_size=config['model_params']['latent_dim'],
                              target_latent_size=config['model_params']['latent_dim'], n_ants=1)

    data = WaveDataModule(vae_model=vae_mdl, **config["dataset_params"])
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
    freqs = np.fft.fftshift(np.fft.fftfreq(dec_fftsz, 1 / fs))
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
    plt.show()

    if save_model_bool:
        print('Saving model...')
        mdl.save_weights('./model/model', save_format='h5')
        with open('./model/model_params.pic', 'wb') as f:
            pickle.dump({'bandwidth': bandwidth, 'num_filters': num_filters,
                         'mdl_fft': fft_sz, 'data_mu': data_mu, 'data_std': data_std,
                         'bin_bw': bin_bw}, f)

    '''print('Evaluating layers...')
    inp = mdl.input  # input placeholder
    outputs = [layer.output for layer in mdl.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layer_outs = [func([(clutter_data, target_data)]) for func in functors]
    weights = [layer.weights for layer in mdl.layers]'''

    # Loss function testing
    '''yp = np.fft.fftshift(np.fft.fft(genPulse(np.linspace(0, 1, 10),
                               np.linspace(0, 1, 10), nr, fs, fc, bandwidth),
                                dec_fftsz)).reshape((-1, 1))
    y_pred = np.tile(np.stack([yp.real, yp.imag]).T, (1, 1, 2))
    # y_pred = mdl.predict((clutter_vae, target_vae))

    y_true = np.tile(abs(np.fft.fftshift(yp / sum(abs(yp)))), (1, 1, 2))
    # y_true = targets

    ret = 0
    n = 0
    # Get the complex conjugate of the waveform spectrum
    energy = K.sum(K.sqrt(K.square(y_pred[:, :, n * 2]) + K.square(y_pred[:, :, n * 2 + 1])), axis=1)
    yp2 = tf.signal.fftshift(tf.complex(y_pred[:, :, n * 2] / energy[:, None],
                                        -y_pred[:, :, n * 2 + 1] / energy[:, None]), axes=1)
    ypconj = tf.signal.fftshift(
        tf.complex(y_pred[:, :, n * 2] / energy[:, None], y_pred[:, :, n * 2 + 1] / energy[:, None]), axes=1)
    autocorr = tf.math.log(K.abs(tf.signal.ifft(ypconj * yp2)))
    autocorr += K.abs(K.min(autocorr))
    autocorr /= autocorr[:, 0][:, None]
    front = autocorr[:, 1:] - autocorr[:, :-1]
    sidelobe_locs = tf.where(tf.logical_and(front[:, 1:] < 0, front[:, :-1] > 0))
    ret += K.mean(autocorr, axis=1) / K.max(autocorr, axis=1) / n_ants
    y, idx, count = tf.unique_with_counts(sidelobe_locs[:, 0])
    idx = tf.cumsum(count)
    idx -= idx[0]
    idxes = tf.gather(sidelobe_locs, idx)

    # Sidelobe height
    ret += 1 - tf.gather_nd(autocorr, tf.transpose(
        tf.stack([idxes[:, 0], idxes[:, 1] + 1])))

    slobes = sidelobe_locs.numpy()
    slobes = slobes[slobes[:, 0] == 0, 1] + 1
    plt.figure()
    plt.plot(autocorr[0, :].numpy().flatten())
    plt.plot(front[0, :].numpy().flatten())
    plt.vlines(slobes, -1, 1, color='red')'''
