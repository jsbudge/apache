import os
import sys
from datetime import datetime

from glob import glob
import numpy as np
from simulib.simulation_functions import genPulse, findPowerOf2, db, GetAdvMatchedFilter
from generate_trainingdata import parse_wrapper, parse_dataset_function
from tensorflow import keras
from keras.optimizers import Adam, Adadelta, Adamax
import tensorflow as tf
import tensorflow_probability as tfp
# from tensorflow.profiler import profile, ProfileOptionBuilder
from keras.callbacks import TensorBoard
from keras.layers import Input, Flatten, Dense, BatchNormalization, \
    Dropout, GaussianNoise, Concatenate, Conv1D, Lambda, MaxPooling1D, ActivityRegularization, \
    LocallyConnected2D, Reshape, LeakyReLU, ZeroPadding1D, Permute, Conv2D, MaxPooling2D, Layer, Conv1DTranspose, ELU, \
    Conv2DTranspose, LayerNormalization
from keras.models import Model, save_model
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l1_l2
from complexnn.conv import ComplexConv2D, ComplexConv1D
from complexnn.dense import ComplexDense
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

latent_dim = 64
cpi_len = 32
epochs = 3000
step_per_epoch = 4
batch_sz = 128
save_to_file = True
tensorboard_run = False
tuner_search = False
activation_type = 'lrelu'
num_filters = 192
learn_rate = 1e-7
model_beta = 250


@tf.function
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    delta = fin - init
    return tf.cast(tf.minimum(init + delta * step / annealing_steps, fin), tf.float32)


class VAE(Model):
    def __init__(self, encoder, decoder, beta=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.n_train_steps = tf.Variable(0, trainable=False, dtype=tf.int64)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker
        ]

    def train_step(self, data):
        self.n_train_steps.assign_add(delta=1)
        with tf.GradientTape() as tape:
            z = self.encoder(data[0])
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data[1], reconstruction), axis=1
                )
            )
            total_loss = reconstruction_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result()
        }


class Triangle(Layer):

    def call(self, inputs):
        x, y = tf.meshgrid(tf.range(0, 32), tf.range(0, 32))
        return tf.boolean_mask(inputs, tf.greater_equal(x, y), axis=1)


class OuterProduct(Layer):

    def call(self, inputs):
        complex_inputs = tf.complex(inputs[:, :, :, 0], inputs[:, :, :, 1])
        sample_cov = tfp.stats.covariance(complex_inputs, complex_inputs, sample_axis=1, event_axis=2)
        return tf.stack([tf.math.real(sample_cov), tf.math.imag(sample_cov)], axis=3)
        # return tfp.stats.covariance(inputs, inputs, sample_axis=1, event_axis=2)


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


def genVAETuning(tuner_input):
    act_type = tuner_input.Choice('activation_type', ['elu', 'lrelu'])
    nfilts = tuner_input.Int('num_filters', 32, 256, step=20)
    lrate = tuner_input.Int('latent_dim', 4, 128, step=8)
    return genVAE(act_type, nfilts, lrate)


def genVAE(act_type, nfilts, prob_latent_dim, inp_sz=(cpi_len, cpi_len, 2)):
    osz = cpi_len * cpi_len // 2 + cpi_len // 2

    # Encoder
    encoder_inputs = keras.Input(shape=inp_sz)
    x = ComplexConv2D(nfilts, (17, 17), activation=ELU() if act_type == 'elu' else LeakyReLU(),
                      name='encoder_conv0', kernel_initializer=tf.keras.initializers.he_normal())(encoder_inputs)
    x = ComplexConv2D(nfilts, (9, 9), activation=ELU() if act_type == 'elu' else LeakyReLU(),
                      name='encoder_conv1', kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = ComplexConv2D(nfilts, (5, 5), activation=ELU() if act_type == 'elu' else LeakyReLU(),
                      name='encoder_conv2', kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = ComplexConv2D(nfilts, (3, 3), activation=ELU() if act_type == 'elu' else LeakyReLU(),
                      name='encoder_conv3', kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = LayerNormalization()(x)
    encoder_outputs = ComplexConv2D(1, (1, 1), activation=ELU() if act_type == 'elu' else LeakyReLU(),
                      name='encoder_output', kernel_initializer=tf.keras.initializers.he_normal())(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs, name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=encoder_outputs.shape[1:])
    x = ComplexConv2D(nfilts, (3, 3), activation=ELU() if act_type == 'elu' else LeakyReLU(),
                      name='decoder_conv0', transposed=True, kernel_initializer=tf.keras.initializers.he_normal())(latent_inputs)
    x = ComplexConv2D(nfilts, (5, 5), activation=ELU() if act_type == 'elu' else LeakyReLU(),
                      name='decoder_conv1', transposed=True, kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = ComplexConv2D(nfilts, (9, 9), activation=ELU() if act_type == 'elu' else LeakyReLU(),
                      name='decoder_conv2', transposed=True, kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = ComplexConv2D(nfilts, (17, 17), activation=ELU() if act_type == 'elu' else LeakyReLU(),
                      name='decoder_conv3', transposed=True,
                      kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = ComplexConv2D(nfilts, (1, 1), activation=ELU() if act_type == 'elu' else LeakyReLU(),
                      name='decoder_conv4', transposed=True,
                      kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = ComplexConv2D(1, (1, 1),
                      name='decoder_finalconv')(x)
    decoder_outputs = Triangle()(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=Adadelta(learning_rate=1e-3))
    return vae


def lr_schedule(ep, lr):
    if ep < 100:
        return .001
    elif 100 <= ep < 200:
        return lr * tf.math.exp(-0.01)
    else:
        return lr * tf.math.exp(-0.05)


if __name__ == '__main__':

    osz = cpi_len * cpi_len // 2 + cpi_len // 2
    # Load up the hyperparameter tuning system
    hp = keras_tuner.HyperParameters()
    total_history = []
    current_run = f'./logs/fit/encoder_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    print('Loading data file...')

    # Get all the separate data files
    clutter_files = glob('./data/clutter_*.tfrecords')
    dataset = tf.data.TFRecordDataset(clutter_files)

    # Map dataset to get inputs and labels
    dataset = dataset.map(
        tf.autograph.experimental.do_not_convert(parse_wrapper(cpi_len, 6554)))

    # Batching and memory optimizations
    dataset = dataset.batch(batch_sz, drop_remainder=True)
    dataset = dataset.shuffle(batch_sz * step_per_epoch)
    dataset = dataset.repeat(step_per_epoch * epochs)
    dataset = dataset.prefetch(batch_sz * step_per_epoch)

    # Training phase
    ver_data = next(dataset.as_numpy_iterator())[0]
    cd_mu = np.mean(ver_data, axis=(1, 2))
    cd_std = np.std(ver_data, axis=(1, 2, 3))
    ver_data = (ver_data - cd_mu[:, None, None, :]) / cd_std[:, None, None, None]
    ver_data_out = ver_data[:, np.triu_indices(cpi_len)[0], np.triu_indices(cpi_len)[1], :]


    def preprocess(covs, specs):
        x, y = tf.meshgrid(tf.range(0, 32), tf.range(0, 32))
        c_mu = tf.reduce_mean(covs, axis=(1, 2))
        c_std = tf.math.reduce_std(covs, axis=(1, 2, 3))
        c_norm = tf.complex((covs[:, :, :, 0] - c_mu[:, None, None, 0]) / c_std[:, None, None],
                            (covs[:, :, :, 1] - c_mu[:, None, None, 1]) / c_std[:, None, None])
        c_comp = tf.boolean_mask(c_norm, tf.greater_equal(x, y), axis=1)
        return (tf.stack([tf.math.real(c_norm), tf.math.imag(c_norm)], axis=3),
                tf.stack([tf.math.real(c_comp), tf.math.imag(c_comp)], axis=2))

    # Finish mapping the dataset and add callbacks
    dataset = dataset.map(tf.autograph.experimental.do_not_convert(preprocess))
    base_callbacks = [EarlyStopping(patience=100, monitor='reconstruction_loss', restore_best_weights=True),
                      TerminateOnNaN(),
                      ReduceLROnPlateau(monitor='reconstruction_loss', patience=40, factor=.1)]

    # Hyperparameters for the tuning search
    hps = {'activation_type': activation_type, 'num_filters': num_filters, 'latent_dim': latent_dim}
    if tuner_search:
        tuner = keras_tuner.RandomSearch(genVAETuning, max_trials=1000, overwrite=True,
                                         objective=keras_tuner.Objective('reconstruction_loss', direction='min'),
                                         directory='./tmp/tb')
        tuner.search(dataset.as_numpy_iterator(), steps_per_epoch=step_per_epoch, epochs=500,
                     callbacks=base_callbacks + [TensorBoard(log_dir=f'{current_run}_tuning')])
        hps = tuner.get_best_hyperparameters()[0]

    # Generate the actual VAE using hyperparameters
    vae = genVAE(hps['activation_type'], hps['num_filters'], hps['latent_dim'])

    if tensorboard_run:
        # This runs the tensorboard callback to view weights, etc.
        # Command to monitor via Tensorboard is
        # tensorboard --logdir ./logs/fit
        # in terminal
        train_history = vae.fit(dataset, steps_per_epoch=step_per_epoch, epochs=epochs,
                                callbacks=base_callbacks + [TensorBoard(log_dir=f'{current_run}', histogram_freq=3)])
    else:
        train_history = vae.fit(dataset, steps_per_epoch=step_per_epoch, epochs=epochs,
                                callbacks=base_callbacks)
    total_history.append(train_history.history)

    print('Plotting reconstruction...')
    plt.figure('Reconstruction')
    mdl_outp = reconstruct(vae.decoder.predict(vae.encoder.predict(ver_data)), cpi_len)
    # mdl_outp = vae.decoder.predict(vae.encoder.predict(ver_data)[2])
    ver_outp = reconstruct(ver_data_out, cpi_len)
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
    plt.subplot(2, 1, 1)
    plt.title('Total loss')
    plt.semilogy(np.concatenate([h['loss'] for h in total_history]))
    plt.subplot(2, 1, 2)
    plt.title('Reconstruction loss')
    plt.semilogy(np.concatenate([h['reconstruction_loss'] for h in total_history]))

    print('Plotting a few generated matrices...')
    generated_covs = reconstruct(vae.decoder.predict(np.random.randint(-4, 4, size=(4, 2, 2, 2))), cpi_len)
    # generated_covs = vae.decoder.predict(np.random.randint(-4, 4, size=(4, latent_dim)))
    plt.figure('Generated')
    for n in range(generated_covs.shape[0]):
        plt.subplot(2, 2, n + 1)
        plt.imshow(db(generated_covs[n, :, :, 0]))

    if save_to_file:
        print('Saving model files...')
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
