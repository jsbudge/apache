import pickle
import numpy as np
from simulib.simulation_functions import genPulse, db
import matplotlib.pyplot as plt
from scipy.signal import stft
import torch
import yaml
from models import load as loadModel
from models import PulseClassifier
from waveform_model import GeneratorModel
from wave_train import buildModel
from sdrparse import load
from simulib.platform_helper import SDRPlatform
from utils import upsample, normalize, fs

# Good SAR backgrounds
# Airport
# /data6/SAR_DATA/2024/08012024/SAR_08012024_132556.sar
# Fields with a river
# /data6/SAR_DATA/2024/08012024/SAR_08012024_132744.sar
# Outskirts of Spanish Fork
# /data6/SAR_DATA/2024/06212024/SAR_06212024_124710.sar


'''-------------------USER PARAMETERS--------------------------'''
pulse_length = 4.096e-6
pulse_bw = 600e6
target_target = 4
fnme = '/data6/SAR_DATA/2024/08012024/SAR_08012024_132744.sar'
wave_fnme = './data/humvee_field.wave'
save_file = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    with open('./data/target_ids.txt', 'r') as f:
        target_ids = f.readlines()
except:
    print('Error with the target IDs.')

print(f'WAVEFORM GENERATOR-------------------------\n'
      f'Pulse Length {pulse_length * 1e6:.2f}us\n'
      f'Pulse Bandwidth {pulse_bw / 1e6:.2f}MHz\n'
      f'Target Index: {target_target}\n'
      f'Clutter File: {fnme}\n'
      f'Device used is {device}\n'
      f'-------------------------------------------')

sdr = load(fnme, progress_tracker=True)
rp = SDRPlatform(sdr)

with open('./vae_config.yaml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

exp_params = config['wave_exp_params']

fft_len = config['settings']['fft_len']
nr = int(pulse_length * 2e9)

print('Setting up wavemodel...')
wave_mdl = loadModel(GeneratorModel, './model/current_model_params.pic')
'''with open('./model/current_model_params.pic', 'rb') as f:
    generator_params = pickle.load(f)
wave_mdl = GeneratorModel(**generator_params)
wave_mdl.load_state_dict(torch.load(generator_params['state_file']))'''
print('Wavemodel loaded from save state.')

wave_mdl.to(device)
wave_mdl.eval()
pnums = np.arange(1700, 1756)
_, raw_pulse_data = sdr.getPulses(pnums, 0)
mfilt = sdr.genMatchedFilter(0, fft_len=fft_len)
ts = sdr[0].pulse_time[pnums]
pulse_data = np.fft.fft(raw_pulse_data.T, wave_mdl.fft_len, axis=1) * mfilt
patterns = torch.load('/home/jeff/repo/apache/data/target_embedding_means.pt')[0]

waves = wave_mdl.full_forward(pulse_data, patterns[0], nr)
wave_mdl.to('cpu')
print('Waveforms generated.')

'''-------PLOTS AND DEBUG------------'''
classifier = loadModel(PulseClassifier, './model/current_pc_params.pic')
classifier.to(device)

targets_in_data = classifier(torch.tensor(pulse_data, dtype=torch.float32).to(device)).cpu().data.numpy()
clutter = normalize(pulse_data[10, :])
print('Loaded clutter and target data...')

# Run some plots for an idea of what's going on
freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, 1 / fs))
plt.figure('Waveform PSD')
for wave in range(waves.shape[0]):
    plt.plot(freqs, db(np.fft.fftshift(waves[wave])))
plt.plot(freqs, db(np.fft.fftshift(clutter)), linestyle=':', linewidth=.3)
plt.legend([f'Waveform {w}' for w in range(waves.shape[0])] + ['Clutter'])
plt.ylabel('Relative Power (dB)')
plt.xlabel('Freq (Hz)')

clutter_corr = np.fft.ifft(np.sum([clutter * waves[n] * waves[n].conj() for n in range(waves.shape[0])], axis=0))
plt.figure('MIMO Correlations')
plt.plot(db(clutter_corr))
plt.legend(['Clutter'])

# Save the model structure out to a PNG
# plot_model(mdl, to_file='./mdl_plot.png', show_shapes=True)
# waveforms = np.fft.fftshift(waveforms, axes=2)
plt.figure('Autocorrelation')
linear = np.fft.fft(
    genPulse(np.linspace(0, 1, 10),
             np.linspace(0, 1, 10), nr, fs, config['settings']['fc'],
             config['settings']['bandwidth']), fft_len)
linear = linear / sum(linear * linear.conj())  # Unit energy
inp_wave = waves[0] * waves[0].conj()
autocorr1 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
lags = np.arange(len(autocorr1)) - len(autocorr1) // 2
perf_autocorr = np.fft.fftshift(db(np.fft.ifft(upsample(linear * linear.conj()))))
if waves.shape[0] > 1:
    inp_wave = waves[1] * waves[1].conj()
    autocorr2 = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
    plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
             autocorr2[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr2.max())
    inp_wave = waves[0] * waves[1].conj()
    autocorrcr = np.fft.fftshift(db(np.fft.ifft(upsample(inp_wave))))
    plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
             autocorrcr[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr1.max())
plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
         autocorr1[len(lags) // 2 - 200:len(lags) // 2 + 200] - autocorr1.max())
plt.plot(lags[len(lags) // 2 - 200:len(lags) // 2 + 200],
         perf_autocorr[len(lags) // 2 - 200:len(lags) // 2 + 200] - perf_autocorr.max(),
         linestyle='--')
plt.legend(['Waveform 1', 'Waveform 2', 'Cross Correlation', 'Linear Chirp'])
plt.xlabel('Lag')

plt.figure('Time Series')
plot_t = np.arange(nr) / fs
for w in range(waves.shape[0]):
    plt.plot(plot_t, np.fft.ifft(waves[w]).real[:nr])
plt.legend([f'Waveform {w}' for w in range(waves.shape[0])])
plt.xlabel('Time')

wave_t = np.fft.ifft(waves[0])[:nr]
win = torch.windows.hann(256).data.numpy()
freq_stft, t_stft, wave_stft = stft(wave_t, return_onesided=False, window=win, fs=2e9)
plt.figure('Wave STFT')
plt.pcolormesh(t_stft, np.fft.fftshift(freq_stft), np.fft.fftshift(db(wave_stft), axes=0))
plt.ylabel('Freq')
plt.xlabel('Time')
plt.colorbar()

# Get target IDs
with open('./data/target_ids.txt', 'r') as f:
    tnames = [s.strip() for s in f.readlines()]
plt.figure('Targets in Data')
plt.scatter(np.arange(targets_in_data.shape[1]), targets_in_data[0])
plt.xticks(np.arange(targets_in_data.shape[1]), tnames)

# Save out the waveform to a file
print('Plots finished.')
if save_file:
    print(f'Saving to {wave_fnme}')
    upsampled_waves = np.zeros((fft_len * 2,), dtype=np.complex64)
    upsampled_waves[:fft_len // 2] = waves[0, :fft_len // 2]
    upsampled_waves[-fft_len // 2:] = waves[0, -fft_len // 2:]
    new_fc = 250e6 + pulse_bw / 2

    bin_shift = int(np.round(new_fc / (fs / fft_len)))
    upsampled_waves = np.roll(upsampled_waves, bin_shift)

    time_wave = np.fft.ifft(upsampled_waves)[:nr * 2]
    scaling = max(time_wave.real.max(), abs(time_wave.real.min()))
    output_wave = (time_wave.real / scaling).astype(np.float32)
    try:
        with open(wave_fnme, 'wb') as f:
            final = np.concatenate((np.array([new_fc, pulse_bw], dtype=np.float32), output_wave))
            final.tofile(f)
    except IOError as e:
        print(f'Error writing to file. {e}')

    with open(wave_fnme, 'rb') as f:
        check_wave = np.fromfile(f, np.float32)
else:
    print('Wave not saved.')
