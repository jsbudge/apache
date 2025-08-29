from config import get_config
from utils import upsample, normalize, fs, narrow_band
import numpy as np
from simulib.simulation_functions import genPulse, db, findPowerOf2
import matplotlib.pyplot as plt
from scipy.signal import stft
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloaders import WaveDataModule
from waveform_model import GeneratorModel


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))



if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    force_cudnn_initialization()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.empty_cache()

    # seed_everything(np.random.randint(1, 2048), workers=True)
    seed_everything(107, workers=True)

    config = get_config('wave_exp', './vae_config.yaml')

    fft_len = config.fft_len
    nr = 5000  # int((config['perf_params']['vehicle_slant_range_min'] * 2 / c0 - 1 / TAC) * fs)
    # Since these are dependent on apache params, we set them up here instead of in the yaml file
    print('Setting up data generator...')
    config.dataset_params['max_pulse_length'] = nr
    config.dataset_params['min_pulse_length'] = 1000

    data = WaveDataModule(device=device, **config.dataset_params)
    data.setup()
    data_iter = iter(data.train_dataloader())
    clutter_mu = []
    target_mu = []
    tvec_mu = []
    dpaths = []

    for clutter_spectrum, target_spectrum, target_vector, pl, bw, dpath in data_iter:
        cs = clutter_spectrum[..., 0, :] + 1j * clutter_spectrum[..., 1, :]
        ts = target_spectrum[..., 0, :] + 1j * target_spectrum[..., 1, :]
        clutter_mu.append([cs.mean(), cs.std()])
        target_mu.append([ts.mean(), ts.std()])
        tvec_mu.append([target_vector.mean(), target_vector.std()])
        dpaths.append(dpath)

    plt.figure('Means')
    plt.subplot(3, 1, 1)
    plt.title('Clutter')
    plt.plot([t[0].real for t in clutter_mu])
    plt.plot([t[0].imag for t in clutter_mu])
    plt.subplot(3, 1, 2)
    plt.title('Target')
    plt.plot([t[0].real for t in target_mu])
    plt.plot([t[0].imag for t in target_mu])
    plt.subplot(3, 1, 3)
    plt.title('Tvec')
    plt.plot([t[0] for t in tvec_mu])

    plt.figure('STDs')
    plt.subplot(3, 1, 1)
    plt.title('Clutter')
    plt.plot([t[1] for t in clutter_mu])
    plt.subplot(3, 1, 2)
    plt.title('Target')
    plt.plot([t[1] for t in target_mu])
    plt.subplot(3, 1, 3)
    plt.title('Tvec')
    plt.plot([t[1] for t in tvec_mu])