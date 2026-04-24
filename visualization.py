import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pickle

from torch import nn
from functools import reduce
from config import get_config
from utils import upsample, fs, narrow_band, getMatchedFilter
from simulib.simulation_functions import genPulse, db, genChirp
from scipy.signal import stft
from scipy.signal.windows import taylor
import torch
from pytorch_lightning import seed_everything
from waveform_model import GeneratorModel




def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def factors(n):
    return list(set(reduce(list.__add__,
                    ([i, n // i] for i in range(1, int(pow(n, 0.5) + 1)) if n % i == 0))))



if __name__ == '__main__':

    files = glob('./data/target_new/*.pic')

    chirp = np.fft.fft(genChirp(1364, 2e9, 16e9, 600e6), 16384)

    for fi in files:
        plt.figure(f'{fi}')
        with open(fi, 'rb') as f:
            data = pickle.load(f)
            cd = np.fft.ifft(data['clutter'][0, :, 0] + 1j * data['clutter'][0, :, 1]).T
            td_prof = data['target'][0, :, 0] + 1j * data['target'][0, :, 1]
            td = np.fft.ifft(td_prof).T
            bt = np.fft.ifft(data['both'][0, :, 0] + 1j * data['both'][0, :, 1]).T
            plt.subplot(3, 1, 1)
            plt.title('sans')
            plt.imshow(np.log10(abs(np.fft.fft(cd, axis=1))))
            plt.axis('tight')
            plt.subplot(3, 1, 2)
            plt.title('with')
            plt.imshow(np.log10(abs(np.fft.fft(bt, axis=1))))
            plt.hlines([data['t_idx'][0, 0]], -.5, cd.shape[1] - .5, linestyle=':')
            plt.axis('tight')
            plt.subplot(3, 1, 3)
            plt.title('just')
            plt.imshow(np.log10(abs(np.fft.fft(td, axis=1))))
            plt.axis('tight')
            '''clutter_data.append(params['clutter'])
            target_data.append(params['target'])
            index_data.append(params['t_idx'])
            nsam.append(params['build']['nsam'])'''

        target_time = np.fft.ifft(td_prof * chirp)
        target_chirp = np.fft.fft(target_time[:, 6274-1378:6274], 16384)

        target_lfm = np.fft.fft(np.fft.ifft(td_prof * chirp * chirp.conj()).T, axis=1)
        target_mod = np.fft.fft(np.fft.ifft(td_prof * target_chirp * target_chirp.conj()).T, axis=1)

        plt.figure('matched')
        plt.subplot(2, 1, 1)
        plt.title('lfm')
        plt.imshow(np.log10(abs(target_lfm)))
        plt.axis('tight')
        plt.subplot(2, 1, 2)
        plt.title('mod')
        plt.imshow(np.log10(abs(target_mod)))
        plt.axis('tight')

    '''for nm, m in wave_mdl.named_modules():
        if isinstance(m, nn.Conv1d):
            wghts = m.weight.data.cpu().numpy()
            if wghts.shape[2] > 1:
                fcs = factors(wghts.shape[1])
                grid_x = fcs[len(fcs) // 2]
                grid_y = wghts.shape[1] // grid_x
                plt.figure(nm)
                for idx in range(wghts.shape[2]):
                    plt.subplot(grid_x, grid_y, idx + 1)
                    plt.plot(wghts[0, idx])'''


