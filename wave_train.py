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
    LocallyConnected1D, Reshape, LeakyReLU, ZeroPadding1D, Permute, Conv2D, ELU, UpSampling1D, Add, Conv1DTranspose
from keras.models import Model, save_model
from complexnn.conv import ComplexConv2D, ComplexConv1D
from complexnn.dense import ComplexDense
from keras.callbacks import TerminateOnNaN, EarlyStopping, TensorBoard
from keras.regularizers import l1_l2
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
import pyarrow.parquet as pq
import pyarrow as pa

from autoencoding import VAE

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808

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

fc = 10e9
bandwidth = 400e6
plp = .5
franges = np.linspace(vehicle_slant_range_min, vehicle_slant_range_max, 1000) * 2 / c0
nrange = franges[0]
pulse_length = (nrange - 1 / TAC) * plp
duty_cycle_time_s = pulse_length + franges
nr = int(pulse_length * fs)
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


def compileWaveformData(vae, vae_mu, vae_std, drange, clutter_file, target_file, fft_sz, dec_factor, data_mu=None,
                        data_std=None):
    if clutter_file:
        vae_clutter = np.array([clutter_file[0][n].as_py() for n in drange])
        vae_clutter = (vae_clutter - vae_mu) / vae_std
        clutter_spectrum = np.array([np.array(clutter_file[1][n].as_py()) /
                                     sum(clutter_file[1][n].as_py()) * 10 for n in drange])
        # Validation for weird zero data
        weirdos = []
        for c in range(clutter_spectrum.shape[0]):
            if sum(clutter_spectrum[c, :200]) == 0:
                weirdos.append(c)
        nonweirdo = 0
        while True:
            if nonweirdo in weirdos:
                nonweirdo += 1
            else:
                break
        for w in weirdos:
            clutter_spectrum[w, :] = clutter_spectrum[nonweirdo, :]
            vae_clutter[w, ...] = vae_clutter[nonweirdo, ...]

    if target_file:
        target_spec = np.random.choice(np.arange(target_file.num_rows), clutter_spectrum.shape[0])
        vae_target = np.array([target_file[0][n].as_py() for n in target_spec])
        vae_target = (vae_target - vae_mu) / vae_std
        target_spectrum = np.array([target_file[1][n].as_py() for n in target_spec])
        # Must resize the target spectrum to fit that of the clutter
        int_target_spectrum = np.zeros(clutter_spectrum.shape)
        for t in range(target_spectrum.shape[0]):
            int_target_spectrum[t, :] = np.interp(
                np.linspace(0, target_spectrum.shape[1], clutter_spectrum.shape[1]),
                np.arange(target_spectrum.shape[1]), target_spectrum[t, :])
            int_target_spectrum[t, :] = int_target_spectrum[t, :] / sum(int_target_spectrum[t, :]) * 10
        target_spectrum = np.fft.fftshift(int_target_spectrum)

    ct_specdata = np.zeros((clutter_spectrum.shape[0], clutter_spectrum.shape[1], 2))
    ct_specdata[:, :, 0] = target_spectrum
    ct_specdata[:, :, 1] = clutter_spectrum

    ct_fftsz_specdata = np.zeros((clutter_spectrum.shape[0], fft_sz, 2))
    ct_fftsz_specdata[:, :clutter_spectrum.shape[1] // 2, :] = ct_specdata[:, :clutter_spectrum.shape[1] // 2, :]
    ct_fftsz_specdata[:, -clutter_spectrum.shape[1] // 2:, :] = ct_specdata[:, -clutter_spectrum.shape[1] // 2:, :]
    ct_fftsz_specdata = ct_fftsz_specdata[:, ::dec_factor, :]

    clutter_vaedata = vae.encoder.predict(vae_clutter)[2]
    target_vaedata = vae.encoder.predict(vae_target)[2]

    if data_mu is None:
        data_mu = clutter_vaedata.mean(axis=0)
        data_std = clutter_vaedata.std(axis=0)
    clutter_vaedata = (clutter_vaedata - data_mu) / data_std
    target_vaedata = (target_vaedata - data_mu) / data_std

    return ct_fftsz_specdata, target_vaedata, clutter_vaedata, data_mu, data_std


def getWaveFromData(wfd_mdl, targetdata, clutterdata, targets):
    wave_output = np.fft.fftshift(wfd_mdl.predict((clutterdata, targetdata, targets)), axes=1)
    wave = wave_output[:, :, ::2] + 1j * wave_output[:, :, 1::2]
    wave /= np.sum(abs(wave), axis=1)[:, None, :]
    return wave


def upsample(val, fac=8):
    upval = np.zeros(len(val) * fac, dtype=np.complex128)
    upval[:len(val) // 2] = val[:len(val) // 2]
    upval[-len(val) // 2:] = val[-len(val) // 2:]
    return upval


def opt_loss(y_true, y_pred):
    gamma = (2 * clutter_loss(y_true, y_pred) + 2 * target_loss(y_true, y_pred) + autocorr_loss(y_true, y_pred) / 2)**2
    return ortho_loss(y_true, y_pred) * gamma + gamma


def clutter_loss(y_true, y_pred):
    ret = 0
    for n in range(n_ants):
        # Get the complex conjugate of the waveform spectrum
        energy = K.sum(K.sqrt(K.square(y_pred[:, :, n * 2]) + K.square(y_pred[:, :, n * 2 + 1])), axis=1)
        yp2 = tf.signal.fftshift(tf.complex(y_pred[:, :, n * 2] / energy[:, None],
                                            -y_pred[:, :, n * 2 + 1] / energy[:, None]), axes=1)
        clutter_diff = K.abs(y_true[:, :, 1] - K.abs(yp2))
        ret += 1 / K.sum(clutter_diff, axis=1)

    return ret


def target_loss(y_true, y_pred):
    ret = 0
    for n in range(n_ants):
        # Get the complex conjugate of the waveform spectrum
        energy = K.sum(K.sqrt(K.square(y_pred[:, :, n * 2]) + K.square(y_pred[:, :, n * 2 + 1])), axis=1)
        yp2 = tf.signal.fftshift(tf.complex(y_pred[:, :, n * 2] / energy[:, None],
                                            -y_pred[:, :, n * 2 + 1] / energy[:, None]), axes=1)
        target_diff = K.abs(y_true[:, :, 0] - K.abs(yp2))
        ret += K.sum(target_diff, axis=1)

    return ret


def autocorr_loss(y_true, y_pred):
    ret = 0
    for n in range(n_ants):
        # Get the complex conjugate of the waveform spectrum
        energy = K.sum(K.sqrt(K.square(y_pred[:, :, n * 2]) + K.square(y_pred[:, :, n * 2 + 1])), axis=1)
        yp2 = tf.signal.fftshift(tf.complex(y_pred[:, :, n * 2] / energy[:, None],
                                            -y_pred[:, :, n * 2 + 1] / energy[:, None]), axes=1)
        ypconj = tf.signal.fftshift(
            tf.complex(y_pred[:, :, n * 2] / energy[:, None], y_pred[:, :, n * 2 + 1] / energy[:, None]), axes=1)
        autocorr = tf.math.log(K.abs(tf.signal.ifft(ypconj * yp2)))
        autocorr += K.abs(K.min(autocorr))
        autocorr /= autocorr[:, 0][:, None]
        # ret += K.mean(autocorr, axis=1) / K.max(autocorr, axis=1) / n_ants
        front = autocorr[:, 1:] - autocorr[:, :-1]
        sidelobe_locs = tf.where(tf.logical_and(front[:, 1:] < 0, front[:, :-1] > 0))
        ret += K.mean(autocorr, axis=1) / K.max(autocorr, axis=1) / n_ants
        y, idx, count = tf.unique_with_counts(sidelobe_locs[:, 0])
        try:
            idx = tf.cumsum(count)
            idx -= idx[0]
            idxes = tf.gather(sidelobe_locs, idx)

            # Sidelobe height
            ret += tf.gather_nd(autocorr, tf.transpose(
                tf.stack([idxes[:, 0], idxes[:, 1] + 1])))

            # Mainlobe width
            ret += tf.cast(idxes[:, 1], tf.float32) / (autocorr.shape[1] / 4)
        except:
            ret += 5

    return ret


def ortho_loss(y_true, y_pred):
    ret = 0
    for n in range(n_ants):
        # Get the complex conjugate of the waveform spectrum
        energy = K.sum(K.sqrt(K.square(y_pred[:, :, n * 2]) + K.square(y_pred[:, :, n * 2 + 1])), axis=1)
        yp2 = tf.signal.fftshift(tf.complex(y_pred[:, :, n * 2] / energy[:, None],
                                            -y_pred[:, :, n * 2 + 1] / energy[:, None]), axes=1)

        # Account for cross-correlations
        for m in range(n_ants):
            if m != n:
                energy = K.sum(K.sqrt(K.square(y_pred[:, :, m * 2]) + K.square(y_pred[:, :, m * 2])), axis=1)
                yp1 = tf.signal.fftshift(tf.complex(y_pred[:, :, m * 2] / energy[:, None],
                                                    y_pred[:, :, m * 2 + 1] / energy[:, None]), axes=1)
                cross_corr = tf.math.log(K.abs(tf.signal.ifft(yp1 * yp2)))
                cross_corr += K.abs(K.min(cross_corr))
                ret += (1 - K.mean(cross_corr, axis=1) / K.max(cross_corr, axis=1)) / n_ants

    return ret


def outBeamTime(theta_az, theta_el):
    return (np.pi ** 2 * wheel_height_m - 8 * np.pi * blade_chord_m * np.tan(theta_el) -
            4 * wheel_height_m * theta_az) / (8 * np.pi * wheel_height_m * rotor_velocity_rad_s)


def getRange(alt, theta_el):
    return alt * np.sin(theta_el) * 2 / c0


def genModel(binbw, nfilts, n_sublayers, vae_shape, fftsz):
    # Let's define the network for function approximation
    # Clutter branch
    clx = Input(shape=vae_shape, dtype=tf.float64)

    # Expected target branch
    tlx = Input(shape=vae_shape, dtype=tf.float64)

    # Power spectrum branches
    plx = Input(shape=(binbw, 2), dtype=tf.float64)
    ply = Flatten()(plx)

    # Concatenation of the branches
    hsz = findPowerOf2(binbw)
    xx = Concatenate()([clx, tlx, ply])
    xx = Dense(hsz * 2, activation=ELU(), name='conc_dense')(xx)
    xx = Reshape((hsz, 2))(xx)

    # Feature pyramid structure
    x1 = ComplexConv1D(nfilts, hsz // 2 + 1, activation=ELU(), name=f'pyramid0')(xx)
    x2 = ComplexConv1D(nfilts, hsz // 4 + 1, activation=ELU(), name=f'pyramid1')(x1)
    x3 = ComplexConv1D(nfilts, hsz // 8 + 1, activation=ELU(), name=f'pyramid2')(x2)
    f1 = Conv1DTranspose(nfilts * 2, hsz // 8 + 1, activation=ELU(), name=f'feedforward0')(x3)
    f2 = Add()([f1, x2])
    f2 = BatchNormalization()(f2)
    f2 = Conv1DTranspose(nfilts * 2, hsz // 4 + 1, activation=ELU(), name='feedforward1')(f2)
    f3 = Add()([f2, x1])
    f3 = BatchNormalization()(f3)
    f3 = ComplexConv1D(nfilts, hsz // 8 + 1, activation=ELU(), name='feedforward2')(f3)

    wave_layer = Flatten()(f3)
    wave_layer = ComplexDense(binbw * n_ants, activation=ELU(), name=f'wave_dense_1')(wave_layer)
    wave_layer = Reshape((binbw, n_ants * 2))(wave_layer)
    wave_layer = ComplexConv1D(n_ants, 1, padding='same', name=f'wave_conv1d_final')(wave_layer)

    # Add zeros to either end of the spectrum
    x_output = ZeroPadding1D(padding=((fftsz - binbw) // 2, (fftsz - binbw) // 2))(wave_layer)

    return Model((clx, tlx, plx), x_output)


if __name__ == '__main__':
    dec_fftsz = fft_sz // dec_factor
    bin_bw = int(bandwidth // (fs / dec_fftsz))
    bin_bw += 1 if bin_bw % 2 != 0 else 0

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

    # Generate the wavemodel using VAE and other parameters
    mdl = genModel(bin_bw, num_filters, n_sublayers, vae_params['latent_dim'], dec_fftsz)
    mdl.compile(optimizer=Adadelta(learning_rate=.03),
                loss=opt_loss,
                metrics=[clutter_loss, target_loss, autocorr_loss, ortho_loss])

    # with open('./test_target.pic', 'rb') as f:
    #     tpsd_f = pickle.load(f)

    print('Loading parquet files...')
    total_hist = []
    hist_lines = []
    data_mu = 0  # None
    data_std = 1  # None
    # Training phase
    data_test = pq.read_table('./data/test.parquet')
    data_targets = pq.read_table('./data/targets.parquet')

    # Generate the Target PSDs. These are stochastic, so they'll be different for each run
    for run in range(min(n_runs, 10000)):
        targets, target_vae, clutter_vae, data_mu, data_std = \
            compileWaveformData(vae, vae_params['mu'], vae_params['std'],
                                np.random.choice(np.arange(l_sz, data_test.num_rows), l_sz),
                                data_test, data_targets, fft_sz,
                                dec_factor, data_mu, data_std)

        input_psd = np.zeros((targets.shape[0], bin_bw, 2), dtype=np.float64)
        input_psd[:, :bin_bw // 2, :] = targets[:, -bin_bw // 2:, :]
        input_psd[:, -bin_bw // 2:, :] = targets[:, :bin_bw // 2, :]
        # Command to monitor via Tensorboard is
        # tensorboard --logdir ./logs/fit
        # in terminal
        if save_histogram:
            hist_dict = mdl.fit((clutter_vae, target_vae, input_psd), targets, epochs=n_epochs,
                                callbacks=[EarlyStopping(patience=40, monitor='loss', restore_best_weights=True),
                                           TerminateOnNaN(),
                                           TensorBoard(
                                               log_dir=f'./logs/fit/wavemodel_run{run}', histogram_freq=10)])
        else:
            hist_dict = mdl.fit((clutter_vae, target_vae, input_psd), targets, epochs=n_epochs,
                                callbacks=[EarlyStopping(patience=40, monitor='loss', restore_best_weights=True),
                                           TerminateOnNaN()])
        hist_lines.append(len(hist_dict.history['loss']))
        loss_array = np.array([hist_dict.history['loss'], hist_dict.history['clutter_loss'],
                               hist_dict.history['target_loss'], hist_dict.history['autocorr_loss'],
                               hist_dict.history['ortho_loss']])
        if len(total_hist) > 0:
            total_hist = np.concatenate((total_hist, loss_array), axis=1)
        else:
            total_hist = loss_array

    # Testing phase
    print('Generating PSDs for testing...')
    run = np.random.randint(0, n_runs)
    targets, target_vae, clutter_vae, data_mu, data_std = \
        compileWaveformData(vae, vae_params['mu'], vae_params['std'],
                            np.random.choice(np.arange(l_sz, data_test.num_rows), l_sz),
                            data_test, data_targets, fft_sz,
                            dec_factor, data_mu, data_std)

    input_psd = np.zeros((targets.shape[0], bin_bw, 2), dtype=np.float64)
    input_psd[:, :bin_bw // 2, :] = targets[:, -bin_bw // 2:, :]
    input_psd[:, -bin_bw // 2:, :] = targets[:, :bin_bw // 2, :]

    waveforms = getWaveFromData(mdl, target_vae, clutter_vae, input_psd)
    waveform_plot = db(waveforms)

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
