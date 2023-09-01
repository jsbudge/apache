import os
import sys

import numpy as np
from simulib.simulation_functions import genPulse, findPowerOf2, db, GetAdvMatchedFilter
from tensorflow import keras
from keras.optimizers import Adam, Adadelta
import tensorflow as tf
# from tensorflow.profiler import profile, ProfileOptionBuilder
from keras.layers import Input, Flatten, Dense, BatchNormalization, \
    Dropout, GaussianNoise, Concatenate, Conv1D, Lambda, MaxPooling1D, ActivityRegularization, \
    LocallyConnected2D, Reshape, LeakyReLU, ZeroPadding1D, Permute, Conv2D, MaxPooling2D, Layer, Conv2DTranspose
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

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808

latent_dim = 16
cpi_len = 32
epochs = 10000
iterations = 10
batch_sz = 64
save_to_file = True

sdr_file = ['/data6/SAR_DATA/2023/08092023/SAR_08092023_143927.sar',
            '/data6/SAR_DATA/2023/08092023/SAR_08092023_112016.sar',
            '/data6/SAR_DATA/2023/08092023/SAR_08092023_144437.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_114640.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_144235.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_091003.sar',
            '/data6/SAR_DATA/2023/08232023/SAR_08232023_090943.sar',
            '/data6/SAR_DATA/2023/08102023/SAR_08102023_110807.sar']
# sdr_file = ['/data6/SAR_DATA/2023/08092023/SAR_08092023_143927.sar']


def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    return min(init + delta * step / annealing_steps, fin)


class VAE(Model):
    def __init__(self, encoder, decoder, beta=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        if beta is not None:
            self.is_beta = True
            self.beta = beta
        else:
            self.is_beta = False
        self.n_train_steps = 0

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        self.n_train_steps += 1
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data[1], reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            if self.is_beta:
                total_loss = reconstruction_loss + linear_annealing(
                    0, 1, self.n_train_steps, 150) * (self.beta * kl_loss)
            else:
                total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


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


def genVAE(inp_sz, latent_dim):
    osz = cpi_len * cpi_len // 2 + cpi_len // 2

    # Encoder
    encoder_inputs = keras.Input(shape=inp_sz)
    x = Conv2D(30, (10, 10), kernel_regularizer=l1_l2(), bias_regularizer=l1_l2(),
               activity_regularizer=l1_l2())(encoder_inputs)
    x = LeakyReLU()(x)
    x = Conv2D(30, (10, 10), kernel_regularizer=l1_l2(), bias_regularizer=l1_l2(),
               activity_regularizer=l1_l2())(x)
    x = LeakyReLU()(x)
    x = Conv2D(30, (10, 10), kernel_regularizer=l1_l2(), bias_regularizer=l1_l2(),
               activity_regularizer=l1_l2())(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = Dense(1024, kernel_regularizer=l1_l2(), bias_regularizer=l1_l2(),
              activity_regularizer=l1_l2())(latent_inputs)
    x = LeakyReLU()(x)
    x = Dense(inp_sz[0] * inp_sz[1] * inp_sz[2])(x)
    x = Reshape(inp_sz)(x)
    x = Conv2DTranspose(30, (10, 10), kernel_regularizer=l1_l2(), bias_regularizer=l1_l2(),
               activity_regularizer=l1_l2())(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(30, (10, 10), kernel_regularizer=l1_l2(), bias_regularizer=l1_l2(),
                        activity_regularizer=l1_l2())(x)
    x = LeakyReLU()(x)
    x = Conv2DTranspose(30, (10, 10), kernel_regularizer=l1_l2(), bias_regularizer=l1_l2(),
               activity_regularizer=l1_l2())(x)
    x = Flatten()(x)
    x = Dense(osz * 2, kernel_regularizer=l1_l2(), bias_regularizer=l1_l2(),
              activity_regularizer=l1_l2())(x)
    decoder_outputs = Reshape((osz, 2))(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return VAE(encoder, decoder, beta=250)


if __name__ == '__main__':

    vae = genVAE((cpi_len, cpi_len, 2), latent_dim)
    vae.compile(optimizer=Adadelta(learning_rate=.25, clipnorm=5))
    total_history = []

    print('Loading SAR file...')
    cd_mu = None
    cd_std = None
    checkfig = plt.figure()
    ax = checkfig.add_subplot(projection='3d')
    # Training phase
    for fn in sdr_file:
        sdr_f = load(fn)
        fft_len = findPowerOf2(sdr_f[0].nsam + sdr_f[0].pulse_length_N)
        mfilt = GetAdvMatchedFilter(sdr_f[0], fft_len=fft_len)
        print(f'File is {fn}')
        ver_data = []
        for n in range(0, cpi_len // 2 * batch_sz, cpi_len // 2):
            # pulses = np.fft.ifft(np.fft.fft(sdr_f.getPulses(sdr_f[0].frame_num[n:n + cpi_len], 0),
            #                                 fft_len, axis=0) * mfilt[:, None], axis=0)[:, :sdr_f[0].nsam]
            # pulses = np.fft.fft(pulses, axis=1)
            pulses = sdr_f.getPulses(sdr_f[0].frame_num[n:n + cpi_len], 0)
            cov_dt = np.cov(pulses.T)
            ver_data.append(np.stack((cov_dt.real, cov_dt.imag), axis=2))
        ver_data = np.array(ver_data)
        cd_mu = (ver_data[:, :, :, 0].mean(), ver_data[:, :, :, 1].mean())
        cd_std = (ver_data[:, :, :, 0].std(), ver_data[:, :, :, 1].std())
        ver_data[:, :, :, 0] = (ver_data[:, :, :, 0] - cd_mu[0]) / cd_std[0]
        ver_data[:, :, :, 1] = (ver_data[:, :, :, 1] - cd_mu[1]) / cd_std[1]
        ver_data_out = ver_data[:, np.triu_indices(cpi_len)[0], np.triu_indices(cpi_len)[1], :]

        for m in tqdm(range(iterations)):
            inp_data = []
            for n in range(m * cpi_len // 2 * batch_sz, m * cpi_len // 2 * batch_sz + cpi_len // 2 * batch_sz,
                           cpi_len // 2):
                # pulses = np.fft.ifft(np.fft.fft(sdr_f.getPulses(sdr_f[0].frame_num[n:n + cpi_len], 0),
                #                                 fft_len, axis=0) * mfilt[:, None], axis=0)[:, :sdr_f[0].nsam]
                # pulses = np.fft.fft(pulses, axis=1)
                pulses = sdr_f.getPulses(sdr_f[0].frame_num[n:n + cpi_len], 0)
                if pulses.shape[1] < cpi_len:
                    break
                cov_dt = np.cov(pulses.T)
                inp_data.append(np.stack((cov_dt.real, cov_dt.imag), axis=2))
            inp_data = np.array(inp_data)
            inp_data[:, :, :, 0] = (inp_data[:, :, :, 0] - cd_mu[0]) / cd_std[0]
            inp_data[:, :, :, 1] = (inp_data[:, :, :, 1] - cd_mu[1]) / cd_std[1]
            inp_data_out = inp_data[:, np.triu_indices(cpi_len)[0], np.triu_indices(cpi_len)[1], :]

            train_history = vae.fit(inp_data, inp_data_out, epochs=epochs,
                    callbacks=[EarlyStopping(patience=40, monitor='loss', restore_best_weights=True),
                               TerminateOnNaN()])

            vae.n_train_steps += 1
            total_history.append(train_history.history)

            z_mean, z_log_var, encoded_ver = vae.encoder.predict(ver_data)

            ax.scatter(encoded_ver[:, 0], encoded_ver[:, 1], encoded_ver[:, 2])

    print('Plotting reconstruction...')
    plt.figure('Reconstruction')
    mdl_outp = reconstruct(vae.decoder.predict(vae.encoder.predict(ver_data)[2]), cpi_len)
    ver_outp = ver_data
    plt.subplot(2, 2, 1)
    plt.title('Real Original')
    plt.imshow(db(ver_outp[0, :, :, 0]))
    plt.subplot(2, 2, 2)
    plt.title('Imag Original')
    plt.imshow(db(ver_outp[0, :, :, 1]))
    plt.subplot(2, 2, 3)
    plt.title('Real Rec.')
    plt.imshow(db(mdl_outp[0, :, :, 0]))
    plt.subplot(2, 2, 4)
    plt.title('Imag Rec.')
    plt.imshow(db(mdl_outp[0, :, :, 1]))

    print('Plotting loss history...')
    plt.figure('Loss History')
    plt.subplot(3, 1, 1)
    plt.title('Total loss')
    plt.plot(np.concatenate([h['loss'] for h in total_history]))
    plt.subplot(3, 1, 2)
    plt.title('Reconstruction loss')
    plt.plot(np.concatenate([h['reconstruction_loss'] for h in total_history]))
    plt.subplot(3, 1, 3)
    plt.title('KL loss')
    plt.plot(np.concatenate([h['kl_loss'] for h in total_history]))

    if save_to_file:
        # First you access the learnt weights of the encoder and decoder from the VAE model and save them
        vae.get_layer('encoder').save_weights('./model/vae/encoder_weights.h5')
        vae.get_layer('decoder').save_weights('./model/vae/decoder_weights.h5')

        # Since both encoder and decoder are treated as models, you also need to save
        # their architecture defined via instantiated VAE model
        vae.get_layer('encoder').save(
            './model/vae/encoder_arch')
        vae.get_layer('decoder').save('./model/vae/decoder_arch')
        with open('./model/vae/vae_params.pic', 'wb') as f:
            pickle.dump({'cpi_len': cpi_len, 'latent_dim': latent_dim, 'mu': cd_mu, 'std': cd_std}, f)
    print('Showing plots...')
    plt.show()

