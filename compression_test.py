import os
import sys
from datetime import datetime

import numpy as np
from simulib.simulation_functions import genPulse, findPowerOf2, db, GetAdvMatchedFilter
from tensorflow import keras
from keras.optimizers import Adam, Adadelta
import tensorflow as tf
# from tensorflow.profiler import profile, ProfileOptionBuilder
from keras.callbacks import TensorBoard
from keras.layers import Input, Flatten, Dense, BatchNormalization, \
    Dropout, GaussianNoise, Concatenate, Conv1D, Lambda, MaxPooling1D, ActivityRegularization, \
    LocallyConnected2D, Reshape, LeakyReLU, ZeroPadding1D, Permute, Conv2D, MaxPooling2D, Layer, Conv1DTranspose, ELU, \
    Conv2DTranspose, UpSampling1D
from complexnn.conv import ComplexConv2D, ComplexConv1D
from complexnn.dense import ComplexDense
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
import pyarrow.parquet as pq
import keras_tuner
import pyarrow as pa

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808

latent_dim = 1024
cpi_len = 32
epochs = 15
iterations = 5
batch_sz = 64
save_to_file = True
tensorboard_run = False
tuner_search = False
activation_type = 'elu'
num_filters = 16
learn_rate = .1

sdr_file = ['/data6/SAR_DATA/2023/06072023/SAR_06072023_154802.sar',
            '/data6/SAR_DATA/2023/06062023/SAR_06062023_125944.sar',
            '/data6/SAR_DATA/2023/08092023/SAR_08092023_143927.sar',
            '/data6/SAR_DATA/2023/08092023/SAR_08092023_112016.sar',
            '/data6/SAR_DATA/2023/08092023/SAR_08092023_144437.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_114640.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_144235.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_091003.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_090943.sar',
            '/data6/SAR_DATA/2023/08102023/SAR_08102023_110807.sar',
            '/data6/SAR_DATA/2023/09132023/SAR_09132023_114021.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_115704.sar']


def reconstruct(tri_cov, cov_shape):
    # Getting the upper triangular portion of the covariance matrix,
    # Rebuild it and make it a square covariance matrix
    ret = np.zeros((tri_cov.shape[0], cov_shape, cov_shape, 2))
    ui = np.triu_indices(cov_shape)
    li = np.tril_indices(cov_shape)
    ret[:, ui[0], ui[1], 0] = tri_cov[:, :, 0]
    ret[:, li[0], li[1], 0] = np.transpose(ret, (0, 2, 1, 3))[:, li[0], li[1], 0]

    # The imaginary part of the matrix has the upper part negated
    # since it's Hermitian symmetric
    ret[:, ui[0], ui[1], 1] = tri_cov[:, :, 1]
    ret[:, li[0], li[1], 1] += -np.transpose(ret, (0, 2, 1, 3))[:, li[0], li[1], 1]
    return ret


def genVAETuning(tuner):
    act_type = tuner.Choice('activation_type', ['elu', 'lrelu'])
    nfilts = tuner.Int('num_filters', 3, 10, step=3)
    lrate = tuner.Choice('learn_rate', [.1, .5, .99])
    return genVAE(act_type, nfilts, lrate)


def genVAE(act_type, nfilts, learn_rate, inp_sz=(cpi_len, cpi_len, 2), latent_dim=latent_dim):

    # Encoder
    encoder_inputs = keras.Input(shape=inp_sz)
    x = ComplexConv1D(nfilts, 2048, activation=ELU() if act_type == 'elu' else LeakyReLU(),
               name='encoder_conv0')(encoder_inputs)
    x = MaxPooling1D(4)(x)
    x = ComplexConv1D(nfilts, 1024, activation=ELU() if act_type == 'elu' else LeakyReLU(),
               name='encoder_conv1')(x)
    x = MaxPooling1D(4)(x)
    x = ComplexConv1D(2, 64, activation=ELU() if act_type == 'elu' else LeakyReLU(),
                      name='encoder_conv2')(x)
    x = Dropout(.33)(x)
    x = Flatten()(x)
    encoder_output = Dense(latent_dim, name="encoder_output")(x)
    # encoder = keras.Model(encoder_inputs, encoder_output, name="encoder")

    # Decoder
    # latent_inputs = keras.Input(shape=(latent_dim,))
    x = Dense(latent_dim, activation=LeakyReLU(), name='decoder_dense0')(encoder_output)
    x = Dense(65 * 2)(x)
    x = Reshape((65, 2))(x)
    x = Conv1DTranspose(2, 64, activation=ELU() if act_type == 'elu' else LeakyReLU())(x)
    x = UpSampling1D(4)(x)
    x = Conv1DTranspose(nfilts, 1024, activation=ELU() if act_type == 'elu' else LeakyReLU(),
                        name='decoder_conv0')(x)
    x = UpSampling1D(4)(x)
    x = Conv1DTranspose(nfilts, 2053, activation=ELU() if act_type == 'elu' else LeakyReLU(),
                        name='decoder_conv1')(x)
    decoder_outputs = ComplexConv1D(1, 1024,
                        name='decoder_conv2', padding='same')(x)
    decoder = keras.Model(encoder_inputs, decoder_outputs, name="decoder")
    decoder.compile(optimizer=Adadelta(
        learning_rate=learn_rate), loss='mse')
    return decoder


if __name__ == '__main__':
    total_history = []
    hps = {'activation_type': activation_type, 'num_filters': num_filters, 'learn_rate': learn_rate}
    sdr_f = load(sdr_file[0])
    fft_len = findPowerOf2(sdr_f[0].nsam + sdr_f[0].pulse_length_N)
    dec_fft_len = fft_len // 4
    complex_data = np.fft.fft(sdr_f.getPulses(sdr_f[0].frame_num[:batch_sz], 0), fft_len, axis=0)
    complex_data = complex_data[::4, :]
    ver_data = np.stack([complex_data.real, complex_data.imag]).T
    mu = abs(complex_data).mean()
    std = abs(complex_data).std()
    vae = genVAE(hps['activation_type'], hps['num_filters'], hps['learn_rate'], inp_sz=(dec_fft_len, 2))
    for m in tqdm(range(iterations)):
        # inp_data = np.array([np.array(pq_data[0][n].as_py()) for
        #                      n in np.arange(m * batch_sz, min((m + 1) * batch_sz, pq_data.num_rows))])
        complex_data = np.fft.fft(sdr_f.getPulses(sdr_f[0].frame_num[m * batch_sz:(m + 1) * batch_sz], 0), fft_len, axis=0)
        complex_data = complex_data[::4, :]
        inp_data = (np.stack([complex_data.real, complex_data.imag]).T - mu) / std

        if tensorboard_run:
            train_history = vae.fit(inp_data, inp_data, epochs=epochs,
                                    callbacks=[EarlyStopping(patience=40, monitor='loss',
                                                             restore_best_weights=True),
                                               TerminateOnNaN(),
                                               TensorBoard(log_dir=f'compress_run{m}', histogram_freq=30)])
        else:
            train_history = vae.fit(inp_data, inp_data, epochs=epochs,
                                    callbacks=[EarlyStopping(patience=40, monitor='loss',
                                                             restore_best_weights=True),
                                               TerminateOnNaN()])
        total_history.append(train_history.history)

    outputs = vae.predict(inp_data)

    plt.figure()
    plt.plot(inp_data[0, :, 0])
    plt.plot(outputs[0, :, 0])
    plt.show()