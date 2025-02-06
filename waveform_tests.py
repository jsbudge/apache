import sys
import numpy as np
import torch
from numba import cuda
from sdrparse import load
from simulib.backproject_functions import backprojectPulseSet, backprojectPulseStream
from simulib.mesh_objects import Mesh, Scene
from apache_helper import ApachePlatform
from config import get_config
from models import load as loadModel, TargetEmbedding
from simulib.simulation_functions import llh2enu, db, azelToVec, genChirp, genTaylorWindow
from simulib.cuda_kernels import applyRadiationPatternCPU
from simulib.mesh_functions import readCombineMeshFile, getRangeProfileFromScene
from simulib.mimo_functions import genChirpAndMatchedFilters, genChannels
from simulib.grid_helper import MapEnvironment
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import yaml
from scipy.signal import sawtooth
from scipy.linalg import convolution_matrix
from waveform_model import GeneratorModel

pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
GPS_UPDATE_RATE_HZ = 100

if __name__ == '__main__':
    # Load all the settings files
    with open('./wave_simulator.yaml') as y:
        settings = yaml.safe_load(y.read())
    with open('./vae_config.yaml', 'r') as file:
        try:
            wave_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    sim_settings = settings['simulation_params']

    # This is all the constants in the radar equation for this simulation
    fc = settings['fc']
    fft_sz = settings['settings']['fft_sz']
    radar_coeff = (
            c0 ** 2 / fc ** 2 * settings['antenna_params']['transmit_power'][0] * 10 ** (
                (settings['antenna_params']['gain'][0] + 2.15) / 10) * 10 ** (
                        (settings['antenna_params']['gain'][0] + 2.15) / 10) *
            10 ** ((settings['antenna_params']['rec_gain'][0] + 2.15) / 10) / (4 * np.pi) ** 3)
    noise_power = 10 ** (sim_settings['noise_power_db'] / 10)

    print('Setting up embedding model...')
    target_config = get_config('target_exp', './vae_config.yaml')
    embedding = TargetEmbedding.load_from_checkpoint(f'{target_config.weights_path}/{target_config.model_name}.ckpt',
                                                     config=target_config, strict=False)
    print('Setting up wavemodel...')
    model_config = get_config('wave_exp', './vae_config.yaml')
    wave_mdl = GeneratorModel.load_from_checkpoint(f'{model_config.weights_path}/{model_config.model_name}.ckpt',
                                                   config=model_config, embedding=embedding, strict=False)
    # wave_mdl.to(device)
    print('Wavemodel loaded.')
    patterns = torch.tensor(torch.load('/home/jeff/repo/apache/data/target_tensors/target_embedding_means.pt')[2],
                            dtype=torch.float32)

    # Get a sar file and peruse it
    sar_fnme = '/data6/SAR_DATA/2024/07082024/SAR_07082024_121331.sar'

    sdr_f = load(sar_fnme)
    mfilt = np.fft.fft(sdr_f.genMatchedFilter(0, fft_sz), fft_sz)

    test = np.fft.ifft(np.fft.fft(sdr_f.getPulses(np.arange(10), 0)[1], fft_sz, axis=-1) * mfilt)[:, :nsam]

    # Run through various test patterns and see what happens

    wave_mdl.full_forward(test, patterns[0], 3048)


