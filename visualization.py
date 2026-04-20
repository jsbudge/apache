import torch
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pickle
from config import get_config
from utils import upsample, fs, narrow_band, getMatchedFilter
from simulib.simulation_functions import genPulse, db
from scipy.signal import stft
from scipy.signal.windows import taylor
import torch
from pytorch_lightning import seed_everything
from waveform_model import GeneratorModel




def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))



if __name__ == '__main__':

    files = glob('./data/target_new/*.pic')

    for fi in files:
        plt.figure(f'{fi}')
        with open(fi, 'rb') as f:
            data = pickle.load(f)
            cd = np.fft.ifft(data['clutter'][0, :, 0] + 1j * data['clutter'][0, :, 1]).T
            td = np.fft.ifft(data['target'][0, :, 0] + 1j * data['target'][0, :, 1]).T
            plt.subplot(2, 1, 1)
            plt.title('sans')
            plt.imshow(np.log10(abs(np.fft.fft(cd, axis=1))))
            plt.axis('tight')
            plt.subplot(2, 1, 2)
            plt.title('with')
            plt.imshow(np.log10(abs(np.fft.fft(td, axis=1))))
            plt.hlines([data['t_idx'][0, 0]], -.5, cd.shape[1] - .5, linestyle=':')
            plt.axis('tight')
            '''clutter_data.append(params['clutter'])
            target_data.append(params['target'])
            index_data.append(params['t_idx'])
            nsam.append(params['build']['nsam'])'''


    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    force_cudnn_initialization()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.empty_cache()

    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(107, workers=True)

    config = get_config('wave_exp', './vae_config.yaml')

    fft_len = config.fft_len
    nr = 5000  # int((config['perf_params']['vehicle_slant_range_min'] * 2 / c0 - 1 / TAC) * fs)
    # Since these are dependent on apache params, we set them up here instead of in the yaml file
    print('Setting up data generator...')
    config.dataset_params['max_pulse_length'] = nr
    config.dataset_params['min_pulse_length'] = 1000
    tbandwidth = .3806

    print('Initializing wavemodel...')
    wave_mdl = GeneratorModel.load_from_checkpoint(f'{config.weights_path}/{config.model_name}.ckpt', config=config, strict=False)

    for fi in files:
        plt.figure(f'{fi}')
        with open(fi, 'rb') as f:
            data = pickle.load(f)
            cd = data['clutter'][0, :, 0] + 1j * data['clutter'][0, :, 1]
            cdp = np.fft.ifft(cd).T
            td = data['target'][0, :, 0] + 1j * data['target'][0, :, 1]
            tdp = np.fft.ifft(td).T
            plt.subplot(2, 1, 1)
            plt.title('sans')
            plt.imshow(np.log10(abs(np.fft.fft(cdp, axis=1))))
            plt.axis('tight')
            plt.subplot(2, 1, 2)
            plt.title('with')
            plt.imshow(np.log10(abs(np.fft.fft(tdp, axis=1))))
            plt.hlines([data['t_idx'][0, 0]], -.5, cdp.shape[1] - .5, linestyle=':')
            plt.axis('tight')
            '''clutter_data.append(params['clutter'])
            target_data.append(params['target'])
            index_data.append(params['t_idx'])
            nsam.append(params['build']['nsam'])'''

            c_inp = torch.tensor(data['clutter'] / config.dataset_params.std, device=wave_mdl.device).unsqueeze(0)
            td_inp = torch.tensor(data['target'] / config.dataset_params.std, device=wave_mdl.device).unsqueeze(0)
            plength = torch.tensor([3178], device=wave_mdl.device).unsqueeze(0)
            bwidth = torch.tensor([tbandwidth], device=wave_mdl.device).unsqueeze(0)

            nn_output = wave_mdl(c_inp, td_inp, plength, bandwidth)
            # nn_numpy = nn_output[0, 0, ...].cpu().data.numpy()

            wave = wave_mdl.getWaveform(nn_output=nn_output).cpu().data.numpy()

            linear = np.fft.fft(genPulse(np.linspace(0, 1, 10),
                                         np.linspace(0, 1, 10), nr, fs, config.fc,
                                         bandwidth[0].cpu().data.numpy() * fs), fft_len)
            linear = linear / np.sqrt(sum(linear * linear.conj()))  # Unit energy

            taytay = np.zeros(fft_len, dtype=np.complex128)
            taytay_len = int(tbandwidth * fft_len) if int(tbandwidth * fft_len) % 2 == 0 else int(
                tbandwidth * fft_len) + 1
            taytay[:taytay_len // 2] = taylor(taytay_len)[-taytay_len // 2:]
            taytay[-taytay_len // 2:] = taylor(taytay_len)[:taytay_len // 2]

            mfiltered_linear = linear * linear.conj() * taytay
            mfiltered_wave0 = wave[tnum, 0] * wave[tnum, 0].conj() * taytay
            linear_corr = np.fft.ifft(td * mfiltered_linear, axis=1)[:nsam]
            wave_corr = np.fft.ifft(td * mfiltered_wave0, axis=1)[:nsam]


