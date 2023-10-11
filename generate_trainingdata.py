import os
import sys
from datetime import datetime
from scipy.stats import rayleigh
import numpy as np
from simulib.simulation_functions import genPulse, findPowerOf2, db, GetAdvMatchedFilter
import matplotlib.pyplot as plt
from scipy.signal import welch
from data_converter.SDRParsing import SDRParse, load
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pickle
from jax import jit
from jax.numpy import fft as jaxfft
import tensorflow as tf
import yaml


fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808


# example proto decode
def parse_dataset_function(example_proto, cpi_len, bin_bw):
    keys_to_features = {'cov': tf.io.FixedLenFeature(shape=(cpi_len, cpi_len, 2), dtype=tf.float32),
                        'spectrum': tf.io.FixedLenFeature(shape=(bin_bw,), dtype=tf.float32)}
    parsed_features = tf.io.parse_single_example(example_proto, keys_to_features)
    return parsed_features['cov'], parsed_features['spectrum']


def parse_wrapper(cpi_len, bin_bw):
    return lambda x: parse_dataset_function(x, cpi_len, bin_bw)


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


def genTargetPSDSwerling1(bw, fc, rng_min, rng_max, spec_sz, fs, cpi_len, fft_sz, sz_m=15, alpha=None, chirp=None):
    """
    Generates a target power spectral density using a bunch of random params
    :param sz_m: (float) Radial size of the target in meters.
    :param alpha: (ndarray) Array of shape parameters. Must be 0, .5, or 1 in each element.
    :return: Normalized power spectral density.
    """
    # Generate over a cpi of pulses
    psd = np.zeros((cpi_len, fft_sz), dtype=np.complex128)
    # Number of bins occupied by target
    M = int(2 * sz_m * bw / c0)
    freqs = np.fft.fftfreq(spec_sz, 1 / fs)
    for n in range(cpi_len):
        # Range of target
        rng = np.random.uniform(rng_min, rng_max) + c0 / (2 * bw) * np.arange(M)
        # Shape parameters fo individual scatterers
        alpha = np.random.choice([0, .5, 1], M) if alpha is None else alpha
        # Complex electrical field amplitude
        Am = rayleigh.rvs(size=M) * (np.random.rand(M) + 1j * np.random.rand(M))
        # Get a center frequency for the target response
        t_fc = fc + bw / 2 * np.random.uniform(-1, 1)
        # Overall spectrum of target response given the above parameters
        psd[n, :] = np.sum([Am[n] / rng[n] ** 4 * (1j * freqs / t_fc) ** alpha[n] *
                      np.exp(-1j * 4 * np.pi * freqs / c0 * rng[n]) for n in range(M)], axis=0)
        if chirp is not None:
            psd[n, :] *= chirp
        # Make the spectrum unit energy
        psd[n, :] /= np.sum(abs(psd[n, :]))
    return psd


def formatTargetClutterData(data, bin_bandwidth):
    split = np.zeros((data.shape[0], bin_bandwidth), dtype=np.float64)
    split[:, :bin_bandwidth // 2] = abs(data[:, -bin_bandwidth // 2:])
    split[:, -bin_bandwidth // 2:] = abs(data[:, :bin_bandwidth // 2])
    return split


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
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_115704.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_151902.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_152050.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_152903.sar',
            '/data6/SAR_DATA/2023/09122023/SAR_09122023_153015.sar']

if __name__ == '__main__':
    with open('./vae_config.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    franges = np.linspace(config['perf_params']['vehicle_slant_range_min'],
                          config['perf_params']['vehicle_slant_range_max'], 1000) * 2 / c0
    nrange = franges[0]
    pulse_length = (nrange - 1 / TAC) * config['settings']['plp']
    duty_cycle_time_s = pulse_length + franges
    nr = int(pulse_length * fs)

    if config['generate_data_settings']['run_clutter']:
        print('Running clutter data...')
        writer = None
        # Standardize the FFT length for training purposes (this may cause data loss)
        fft_len = 32768

        for fn in sdr_file:
            sdr_f = load(fn)
            if sdr_f[0].fs != fs:
                continue  # I'll work on this later
            bin_bw = int(config['settings']['bandwidth'] // (sdr_f[0].fs / fft_len))
            bin_bw += 1 if bin_bw % 2 != 0 else 0
            mfilt = GetAdvMatchedFilter(sdr_f[0], fft_len=fft_len)
            rollback = -int(np.round(sdr_f[0].baseband_fc / (sdr_f[0].fs / fft_len)))
            print('Matched filter loaded.')

            print(f'File is {fn}')
            with tf.io.TFRecordWriter(
                    f'./data/clutter_{fn.split("/")[-1].split(".")[0]}.tfrecords') as writer:
                for m in range(config['generate_data_settings']['iterations']):
                    inp_data = []
                    clutter_abs = []
                    for n in tqdm(range(m * config['settings']['cpi_len'] // 2 * config['settings']['batch_sz'], m * config['settings']['cpi_len'] // 2 * config['settings']['batch_sz'] + config['settings']['cpi_len'] // 2 * config['settings']['batch_sz'],
                                   config['settings']['cpi_len'] // 2)):
                        pulse_fft = jaxfft.fft(sdr_f.getPulses(sdr_f[0].frame_num[n:n + config['settings']['cpi_len']], 0),
                                                        fft_len, axis=0) * mfilt[:, None]
                        # If the pulses are offset video, shift to be centered around zero
                        pulse_fft = np.roll(pulse_fft, rollback, axis=0)
                        pulses = jaxfft.ifft(pulse_fft, axis=0)[:sdr_f[0].nsam, :]
                        if n + config['settings']['cpi_len'] > sdr_f[0].frame_num[-1]:
                            break
                        pulses /= np.mean(abs(pulses))  # Make the pulses smaller
                        clutter_abs.append(pulse_fft.mean(axis=1) / np.sum(abs(pulse_fft.mean(axis=1))))
                        cov_dt = np.cov(pulses.T)
                        inp_data.append(np.stack((cov_dt.real, cov_dt.imag), axis=2))
                    if len(inp_data) == 0:
                        break
                    inp_data = np.array(inp_data)
                    clutter_abs = formatTargetClutterData(np.array(clutter_abs), bin_bw)

                    # Get protobuf for the tfrecords, and write to file
                    for c, s in zip(inp_data, clutter_abs):
                        feature = {'cov': tf.train.Feature(float_list=tf.train.FloatList(value=c.flatten())),
                                   'spectrum': tf.train.Feature(float_list=tf.train.FloatList(value=s))}
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())

    if config['generate_data_settings']['run_targets']:
        bin_bw = int(config['settings']['bandwidth'] // (fs / config['generate_data_settings']['fft_sz']))
        bin_bw += 1 if bin_bw % 2 != 0 else 0
        print('Running targets...')
        writer = None
        # Insert metadata so we know what parameters were used to generate these targets
        chirp = np.fft.fft(genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10), nr,
                                    fs, config['settings']['fc'], config['settings']['bandwidth']), config['generate_data_settings']['fft_sz'])
        mfilt = chirp.conj()
        targs = []
        targ_abs = []
        for ntarg in tqdm(range(100)):
            tpsd = genTargetPSDSwerling1(config['settings']['bandwidth'], config['settings']['fc'], config['perf_params']['vehicle_slant_range_min'],
                          config['perf_params']['vehicle_slant_range_max'],
                                         config['generate_data_settings']['fft_sz'], fs, 32, config['generate_data_settings']['fft_sz'])
            targ_abs.append(tpsd.mean(axis=0))
            cov_dt = np.cov(tpsd)
            targs.append(np.stack((cov_dt.real, cov_dt.imag), axis=2))
        targs = np.array(targs)
        targ_abs = formatTargetClutterData(np.array(targ_abs), bin_bw)
        with tf.io.TFRecordWriter(
                f'./data/targets.tfrecords') as writer:
            for c, s in zip(targs, targ_abs):
                feature = {'cov': tf.train.Feature(float_list=tf.train.FloatList(value=c.flatten())),
                           'spectrum': tf.train.Feature(float_list=tf.train.FloatList(value=s))}
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    # Quick check of dataset
    dataset = tf.data.TFRecordDataset('./data/targets.tfrecords')

    # Parse the record into tensors.
    dataset = dataset.map(tf.autograph.experimental.do_not_convert(parse_dataset_function))

    # Generate batches
    dataset = dataset.batch(5)

    for data in dataset:
        print(data[0].numpy())
