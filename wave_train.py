import numpy as np
from simulib.simulation_functions import genPulse, findPowerOf2, db
from tensorflow import keras
import keras.backend as K
from keras.optimizers import Adam, Adadelta
from keras.constraints import NonNeg
import tensorflow as tf
# from tensorflow.profiler import profile, ProfileOptionBuilder
from keras.layers import Input, Flatten, Dense, BatchNormalization, \
    Dropout, GaussianNoise, Concatenate, Conv1D, Lambda, MaxPooling1D, ActivityRegularization, \
    LocallyConnected2D, Reshape, LeakyReLU, ZeroPadding1D, Permute, Conv2D
from keras.models import Model, save_model
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l1_l2
# import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy.signal import welch
from data_converter.SDRParsing import SDRParse, load
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pickle

from autoencoding import VAE

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808


def compileWaveformData(sdr, fft_length, lsz, bw, wf_fc, slant_min, slant_max, sample_f, binbw, vae, tdmu=None,
                        tdstd=None, cd_mu=None, cd_std=None, cdmu=None, cdstd=None, target_signal=None, use_sdr=True):
    cwd_td = np.zeros((lsz, binbw, 2), dtype=np.float64)
    for n in range(lsz):
        if target_signal is None:
            tpsd = genTargetPSD(bw, wf_fc, slant_min, slant_max,
                                fft_length, sample_f).reshape((fft_length, 1))
            tpsd /= np.linalg.norm(tpsd)
        else:
            tpsd = target_signal.reshape((fft_length, 1))
        cwd_td[n, :binbw // 2, 0] = abs(tpsd[:binbw // 2, 0])
        cwd_td[n, -binbw // 2:, 0] = abs(tpsd[-binbw // 2:, 0])
    tdmu = np.mean(cwd_td[:, :, 0]) if tdmu is None else tdmu
    tdstd = np.std(cwd_td[:, :, 0]) if tdstd is None else tdstd

    cwd_td = np.fft.fftshift((cwd_td - tdmu) / tdstd, axes=1)

    # Generate the Clutter PSDs from the SDR file
    if use_sdr:
        cov_dt = np.cov(sdr.getPulses(sdr[0].frame_num[np.arange(0, 32)], 0).T)
        cov_split = np.stack(((cov_dt.real - cd_mu[0]) / cd_std[0],
                              (cov_dt.imag - cd_mu[1]) / cd_std[1]), axis=2).reshape((1, 32, 32, 2))
        cd_vaedata = np.tile(vae.encoder.predict(cov_split)[0],
                             (lsz, 1))
    else:
        cd_vaedata = np.fft.fft(np.ones((sdr[0].nsam, lsz), dtype=np.complex128), fft_length, axis=0).T

    cd_fftdata = abs(np.fft.fft(sdr.getPulses(sdr[0].frame_num[np.arange(lsz)], 0), fft_length, axis=0))
    cwd_td[:, :binbw // 2, 1] = cd_fftdata[-binbw // 2:, :].T
    cwd_td[:, -binbw // 2:, 1] = cd_fftdata[:binbw // 2, :].T

    cdmu = np.mean(cwd_td[:, :, 1]) if cdmu is None else cdmu
    cdstd = np.std(cwd_td[:, :, 1]) if cdstd is None else cdstd

    return cwd_td, cd_vaedata, cdmu, cdstd, tdmu, tdstd


def getWaveFromData(wfd_mdl, targetdata, clutterdata, cdmu, cdstd, lsz, nants, fft_length,
                    binbw):
    wave_output = wfd_mdl.predict((clutterdata, targetdata)) * cdstd - cdmu
    return np.fft.fftshift(wave_output[:, :, :fft_length] + 1j * wave_output[:, :, fft_length:], axes=2)


def upsample(val, fac=8):
    upval = np.zeros(len(val) * fac, dtype=np.complex128)
    upval[:len(val) // 2] = val[:len(val) // 2]
    upval[-len(val) // 2:] = val[-len(val) // 2:]
    return upval


# Constants from the Apache Longbow
blade_chord_m = .533
rotor_diameter_m = 14.63
wheel_height_m = 8 / m_to_ft
rotor_velocity_rad_s = 29.7404

# Constraints on performance
dismount_slant_range_min = 500
dismount_slant_range_max = 15000
vehicle_slant_range_min = 500
vehicle_slant_range_max = 25000
alt_min = 100 / m_to_ft
alt_max = 5000 / m_to_ft
az_min_bw = .75 * DTR
el_min_bw = 2 * DTR

# The antenna can run Ka and Ku
# on individual pulses
sdr_file = ['/data6/SAR_DATA/2023/08092023/SAR_08092023_143927.sar',
            '/data6/SAR_DATA/2023/08092023/SAR_08092023_112016.sar',
            '/data6/SAR_DATA/2023/08092023/SAR_08092023_144437.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_114640.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_144235.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_091003.sar']
# sdr_file = ['/data6/SAR_DATA/2023/08092023/SAR_08092023_143927.sar']
# sdr_file = ['/home/jeff/repo/SAR_DATA/03112022/SAR_03112022_135854.sar']

fc = 10e9
bandwidth = 400e6
plp = .5
franges = np.linspace(vehicle_slant_range_min, vehicle_slant_range_max, 1000) * 2 / c0
nrange = franges[0]
pulse_length = (nrange - 1 / TAC) * plp
duty_cycle_time_s = pulse_length + franges
nr = int(pulse_length * fs)
fft_sz = 2048
n_ants = 2
l_sz = 64
n_conv_filters = 8
kernel_sz = 36
n_epochs = 30000
n_runs = 500
save_model_bool = True
use_generated_tpsd = True

bin_bw = int(bandwidth // (fs / fft_sz))
bin_bw += 1 if bin_bw % 2 != 0 else 0


def opt_loss(y_true, y_pred):
    ret = 0
    for n in range(n_ants):
        # Get the complex conjugate of the waveform spectrum
        yp2 = tf.signal.fftshift(tf.complex(y_pred[:, n, :fft_sz], -y_pred[:, n, fft_sz:]), axes=1)
        ypconj = tf.signal.fftshift(tf.complex(y_pred[:, n, :fft_sz], y_pred[:, n, fft_sz:]), axes=1)
        clutter_diff = K.abs(y_true[:, :, 0] - K.abs(yp2))
        target_diff = K.abs(y_true[:, :, 1] - K.abs(yp2))
        autocorr = tf.math.log(K.abs(tf.signal.ifft(ypconj * yp2)))

        ret += K.mean(-K.mean(clutter_diff, axis=1) + 4 * K.mean(target_diff, axis=1) +
                K.mean(autocorr, axis=1) / autocorr[:, 0]) / n_ants

        # Account for cross-correlations
        for m in range(n_ants):
            if m != n:
                yp1 = tf.signal.fftshift(tf.complex(y_pred[:, m, :fft_sz], y_pred[:, m, fft_sz:]), axes=1)
                cross_corr = tf.math.log(K.abs(tf.signal.ifft(yp1 * yp2)))
                ret += (1 - K.mean(cross_corr) / K.max(cross_corr)) / n_ants

    return ret


def genTargetPSD(bw, fc, rng_min, rng_max, spec_sz, fs, sz_m=15, alpha=None):
    """
    Generates a target power spectral density using a bunch of random params
    :param sz_m: (float) Radial size of the target in meters.
    :param alpha: (ndarray) Array of shape parameters. Must be 0, .5, or 1 in each element.
    :return: Normalized power spectral density.
    """
    # Number of bins occupied by target
    M = int(2 * sz_m * bw / c0)
    freqs = np.fft.fftfreq(spec_sz, 1 / fs)
    # Range of target
    rng = np.random.uniform(rng_min, rng_max) + c0 / (2 * bw) * np.arange(M)
    # Shape parameters fo individual scatterers
    alpha = np.random.choice([0, .5, 1], M) if alpha is None else alpha
    # Complex electrical field amplitude
    Am = np.random.rand(M) + 1j * np.random.rand(M)
    # Get a center frequency for the target response
    t_fc = fc + bw / 2 * np.random.uniform(-1, 1)
    # Overall spectrum of target response given the above parameters
    psd = np.sum([Am[n] / rng[n] ** 4 * (1j * freqs / t_fc) ** alpha[n] *
                  np.exp(-1j * 4 * np.pi * freqs / c0 * rng[n]) for n in range(M)], axis=0)
    return psd / np.linalg.norm(psd)


def outBeamTime(theta_az, theta_el):
    return (np.pi ** 2 * wheel_height_m - 8 * np.pi * blade_chord_m * np.tan(theta_el) -
            4 * wheel_height_m * theta_az) / (8 * np.pi * wheel_height_m * rotor_velocity_rad_s)


def getRange(alt, theta_el):
    return alt * np.sin(theta_el) * 2 / c0


def genModel(binbw, ncvfilt, kernelsz, vae_shape, fftsz):
    # Let's define the network for function approximation
    # Clutter branch
    clutter_input = Input(shape=vae_shape, dtype=tf.float64)
    cx = Dense(fftsz)(clutter_input)
    cx = LeakyReLU()(cx)
    cx = Dense(fftsz)(cx)

    # Expected target branch
    target_input = Input(shape=(bin_bw, 2), dtype=tf.float64)
    tx = Conv1D(filters=ncvfilt, kernel_size=kernelsz)(target_input)
    # tx = MaxPooling1D(pool_size=ncvfilt)(tx)
    tx = Conv1D(filters=ncvfilt, kernel_size=kernelsz // ncvfilt)(tx)
    tx = Conv1D(filters=ncvfilt, kernel_size=kernelsz // ncvfilt)(tx)
    tx = Flatten()(tx)
    tx = Dense(fftsz)(tx)

    # Concatenation of the branches
    xx = Concatenate()([cx, tx])
    xx = Dense(fftsz)(xx)
    xx = LeakyReLU()(xx)
    xx = Dense(fftsz)(xx)
    xx = LeakyReLU()(xx)
    xx = Dense(fftsz)(xx)
    xx = LeakyReLU()(xx)

    # Magnitude and phase of wave 1
    x_mag = Dense(fftsz)(xx)
    x_mag = LeakyReLU()(x_mag)
    x_phase = Dense(fftsz)(xx)
    x_phase = LeakyReLU()(x_phase)

    # Magnitude and phase of wave 2
    x_mag0 = Dense(fftsz)(xx)
    x_mag0 = LeakyReLU()(x_mag0)
    x_phase0 = Dense(fftsz)(xx)
    x_phase0 = LeakyReLU()(x_phase0)

    # Merge and zero-pad
    x_mw0 = Dense(binbw)(x_mag)
    x_pw0 = Dense(binbw)(x_phase)
    x_mw1 = Dense(binbw)(x_mag0)
    x_pw1 = Dense(binbw)(x_phase0)

    xw0 = Concatenate()([x_mw0, x_mw1])
    xw0 = Reshape((binbw, 2), dtype=tf.float64)(xw0)
    xw1 = Concatenate()([x_pw0, x_pw1])
    xw1 = Reshape((binbw, 2), dtype=tf.float64)(xw1)
    xw0 = ZeroPadding1D(padding=((fftsz - binbw) // 2, (fftsz - binbw) // 2))(xw0)
    xw0 = Permute((2, 1))(xw0)
    xw1 = ZeroPadding1D(padding=((fftsz - binbw) // 2, (fftsz - binbw) // 2))(xw1)
    xw1 = Permute((2, 1))(xw1)

    # Concatenation step
    x_output = Concatenate(dtype=tf.float64)([xw0, xw1])

    return Model((clutter_input, target_input), x_output)


if __name__ == '__main__':

    # First, load the VAE for clutter and target
    with open('./model/vae/vae_params.pic', 'rb') as f:
        vae_params = pickle.load(f)
    encoder_new = keras.models.load_model('./model/vae/encoder_arch')  # Loading the encoder model
    decoder_new = keras.models.load_model('./model/vae/decoder_arch')  # Loading the decoder model

    vae = VAE(encoder_new, decoder_new)
    vae.get_layer('encoder').load_weights(
        './model/vae/encoder_weights.h5')  # On a given encoder model defined by vae_new we want to load the weights
    vae.get_layer('decoder').load_weights('./model/vae/decoder_weights.h5')  # for encoder and decoder
    vae.compile(optimizer=keras.optimizers.Adadelta())

    mdl = genModel(bin_bw, n_conv_filters, kernel_sz, 16, fft_sz)
    mdl.compile(optimizer=Adadelta(learning_rate=1.), loss=opt_loss)

    # with open('./test_target.pic', 'rb') as f:
    #     tpsd_f = pickle.load(f)

    print('Loading SAR file...')
    total_hist = []
    hist_lines = []
    td_mu = None
    td_std = None
    cd_mu = None
    cd_std = None
    # Training phase
    for fn in sdr_file:
        sdr_f = load(fn)
        print(f'File is {fn}')

        # Generate the Target PSDs. These are stochastic, so they'll be different for each run
        valid_pulses = sdr_f[0].frame_num
        for run in range(min(n_runs, sdr_f[0].nframes // l_sz)):
            print(f'Generating PSDs for pulses {l_sz * run}-{l_sz * run + l_sz} - run {run}')
            target_data, vae_output, cd_mu, cd_std, td_mu, td_std = \
                compileWaveformData(sdr_f, fft_sz, l_sz, bandwidth, fc, vehicle_slant_range_min,
                                    vehicle_slant_range_max, fs, bin_bw, vae, td_mu, td_std,
                                    vae_params['mu'], vae_params['std'], cd_mu, cd_std,
                                    target_signal=np.random.randn(fft_sz), use_sdr=True)

            targets = np.zeros((l_sz, fft_sz, 2), dtype=np.float64)
            targets[:, fft_sz // 2 - bin_bw // 2:fft_sz // 2 + bin_bw // 2, :] = target_data

            hist_dict = mdl.fit((vae_output, target_data), targets, epochs=n_epochs,
                                callbacks=[EarlyStopping(patience=20, monitor='loss', restore_best_weights=True),
                                           TerminateOnNaN()])
            hist_lines.append(len(hist_dict.history['loss']))
            total_hist = total_hist + hist_dict.history['loss']

    # Testing phase
    for fn in sdr_file:
        sdr_f = load(fn)
        valid_pulses = sdr_f[0].frame_num
        print('Generating PSDs for testing...')
        run = np.random.randint(0, n_runs)
        target_data, vae_output, cd_mu, cd_std, td_mu, td_std = \
            compileWaveformData(sdr_f, fft_sz, l_sz, bandwidth, fc, vehicle_slant_range_min,
                                vehicle_slant_range_max, fs, bin_bw, vae, td_mu, td_std,
                                vae_params['mu'], vae_params['std'], cd_mu, cd_std,
                                target_signal=np.random.randn(fft_sz), use_sdr=True)

        waveforms = getWaveFromData(mdl, target_data, vae_output, cd_mu, cd_std,
                                                      l_sz, n_ants, fft_sz, bin_bw)
        waveform_plot = db(waveforms)

        # Run some plots for an idea of what's going on
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_sz, 1 / fs))
        plt.figure(f'Waveform PSD - {fn.split("/")[-1]}')
        w0 = np.fft.fftshift(waveform_plot[0, 0, :])
        plt.plot(freqs, w0 - w0.max())
        w1 = np.fft.fftshift(waveform_plot[0, 1, :])
        plt.plot(freqs, w1 - w1.max())
        targ = (target_data[0, :, 0] * td_std) + td_mu
        targ_up = np.zeros(fft_sz, dtype=np.complex128)
        targ_up[-bin_bw // 2:] = targ[:bin_bw // 2]
        targ_up[:bin_bw // 2] = targ[-bin_bw // 2:]
        plt.plot(freqs, np.fft.fftshift(db(targ_up) - db(targ_up).max()), linestyle='--')
        clut = (target_data[0, :, 1] * cd_std) + cd_mu
        clut_up = np.zeros(fft_sz, dtype=np.complex128)
        clut_up[-bin_bw // 2:] = clut[:bin_bw // 2]
        clut_up[:bin_bw // 2] = clut[-bin_bw // 2:]
        plt.plot(freqs, np.fft.fftshift(db(clut_up) - db(clut_up).max()), linestyle=':')
        plt.legend(['Waveform 1', 'Waveform 2', 'Target', 'Clutter'])
        plt.ylabel('Relative Power (dB)')
        plt.xlabel('Freq (Hz)')

        # Save the model structure out to a PNG
        # plot_model(mdl, to_file='./mdl_plot.png', show_shapes=True)
        # waveforms = np.fft.fftshift(waveforms, axes=2)
        plt.figure(f'Autocorrelation - {fn.split("/")[-1]}')
        linear = np.fft.fft(
            genPulse(np.linspace(0, 1, 10),
                     np.linspace(0, 1, 10), nr, fs, fc, bandwidth), fft_sz)
        inp_wave = waveforms[0, 0, :] * waveforms[0, 0, :].conj()
        autocorr1 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
        inp_wave = waveforms[0, 1, :] * waveforms[0, 1, :].conj()
        autocorr2 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
        inp_wave = waveforms[0, 0, :] * waveforms[0, 1, :].conj()
        autocorrcr = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
        perf_autocorr = np.fft.fftshift(db(np.fft.ifft(upsample(linear * linear.conj()))))
        lags = np.arange(len(autocorr1)) - len(autocorr1) // 2
        plt.subplot(2, 1, 1)
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
        plt.subplot(2, 1, 2)
        plt.title('Target Correlation')
        inp_wave = waveforms[0, 0, :] * targ_up.conj()
        autocorr1 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
        inp_wave = waveforms[0, 1, :] * targ_up.conj()
        autocorr2 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
        inp_wave = linear * targ_up.conj()
        autocorrcr = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
        plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                 autocorr1[len(lags) // 2 - 200:len(lags) // 2 + 200])
        plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                 autocorr2[len(lags) // 2 - 200:len(lags) // 2 + 200])
        plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
                 autocorrcr[len(lags) // 2 - 200:len(lags) // 2 + 200],
                 linestyle='--')
        plt.xlabel('Lag')

        plt.figure(f'Time Series - {fn.split("/")[-1]}')
        plot_t = np.arange(fft_sz) / fs
        plt.plot(plot_t, np.fft.ifft(np.fft.fftshift(waveforms[0, 0, :])).real)
        plt.plot(plot_t, np.fft.ifft(np.fft.fftshift(waveforms[0, 1, :])).real)
        plt.legend(['Waveform 1', 'Waveform 2'])
        plt.xlabel('Time')

    plt.figure('Training')
    plt.plot(total_hist)
    plt.vlines(np.cumsum(hist_lines) - 1, min(total_hist), max(total_hist), colors='red', linestyles='--', linewidth=.5)
    plt.ylabel('Loss')
    plt.xlabel('Training Epoch')
    plt.show()

    if save_model_bool:
        print('Saving model...')
        mdl.save_weights('./model/model', save_format='h5')
        with open('./model/model_params.pic', 'wb') as f:
            pickle.dump({'bandwidth': bandwidth, 'kernel_sz': kernel_sz, 'n_conv_filters': n_conv_filters,
                         'mdl_fft': fft_sz, 'cd_mu': cd_mu, 'cd_std': cd_std, 'td_mu': td_mu, 'td_std': td_std,
                         'bin_bw': bin_bw}, f)

    '''print('Evaluating layers...')
    inp = mdl.input  # input placeholder
    outputs = [layer.output for layer in mdl.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layer_outs = [func([(clutter_data, target_data)]) for func in functors]
    weights = [layer.weights for layer in mdl.layers]'''

