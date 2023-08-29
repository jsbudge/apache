import numpy as np
from tensorforce import Environment
from simulation_functions import genPulse
from scipy.signal import welch
from sklearn.feature_selection import mutual_info_regression
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'

c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808


class Wavegen(Environment):
    max_steps = 128

    def __init__(self, sdr, range_min, range_max, fft_sz, bin_bw, n_ants, max_timesteps=128, nr=None, fs=None, fc=None,
                 bandwidth=None, spectrum_sz=512, target_psd=None, noise_sigma=1e-6):
        super().__init__()
        self.sdr = sdr
        self.max_steps = max_timesteps
        self.nr = sdr[0].pulse_length if nr is None else nr
        self.fs = sdr[0].fs if fs is None else fs
        self.fc = sdr[0].fc if fc is None else fc
        self.bw = sdr[0].bw if bandwidth is None else bandwidth

        # Calculate out the bandwidth in bins
        self.bin_bw = bin_bw
        self.amp_scale = 100
        self.bw_binp = (fft_sz // 2 - self.bin_bw // 2, fft_sz // 2 + self.bin_bw // 2)
        self.log = []
        self.spec_sz = spectrum_sz
        self.fft_sz = fft_sz
        self.step = 0
        self.pnum = 0
        self.spec_max = 0
        self.rngs = (range_min, range_max)
        self.target_psd = target_psd if target_psd is not None else self.genTargetPSD()
        self.noise_sigma = noise_sigma
        self.n_ants = n_ants

    def max_episode_timesteps(self):
        return self.max_steps

    def states(self):
        return dict(clutter_psd=dict(type='float', shape=(self.spec_sz, 1), min_value=0, max_value=1.),
                     target_psd=dict(type='float', shape=(self.spec_sz, 1), min_value=0, max_value=1.),
                    waveform_psd=dict(type='float', shape=(self.spec_sz, self.n_ants), min_value=0, max_value=1.))

    def actions(self):
        return dict(mag=dict(type='float', shape=(self.bin_bw, self.n_ants), min_value=0, max_value=1.),
                    phase=dict(type='float', shape=(self.bin_bw, self.n_ants), min_value=-2 * np.pi, max_value=2 * np.pi))

    def execute(self, actions):
        # Generate waveform from phase points and get power spectral density
        mag = actions['mag']
        phase = actions['phase']
        freqs = np.fft.fftfreq(self.fft_sz, d=1/self.fs)[self.bw_binp[0]:self.bw_binp[1]]
        # Get clutter data from input SDR file for this execution
        _, clutter_spectra = welch(self.sdr.getPulse(self.pnum, 0), self.fs, 'flattop', nperseg=self.spec_sz,
                                   return_onesided=False)
        _, next_clutter_spectra = welch(self.sdr.getPulse(self.pnum + 1, 0), self.fs, 'flattop', nperseg=self.spec_sz,
                                        return_onesided=False)
        # Normalize
        clutter_spectra /= np.linalg.norm(clutter_spectra)
        self.spec_max = max(self.spec_max, clutter_spectra.max())
        next_clutter_spectra /= np.linalg.norm(next_clutter_spectra)
        pulses = np.zeros((mag.shape[1], self.nr), dtype=np.complex128)
        for p in range(mag.shape[1]):
            pulse_spectrum = np.zeros((self.fft_sz, ), dtype=np.complex128)
            pulse_spectrum[self.bw_binp[0]:self.bw_binp[1]] = mag[:, p] * np.exp(-1j * 2 * np.pi * freqs + phase[:, p])
            pulses[p, :] = np.fft.ifft(pulse_spectrum)[:self.nr] * self.amp_scale
        scores = np.zeros(4)
        spectra = np.zeros((self.spec_sz, self.n_ants))
        for n in range(pulses.shape[0]):
            psd_freqs, spectra[:, n] = welch(pulses[n, :], self.fs, 'flattop', nperseg=self.spec_sz, return_onesided=False)
            spectra[:, n] /= np.linalg.norm(spectra[:, n])

            # Get scores for difference between wave and clutter/target
            scores[0] += np.trapz(-(clutter_spectra - spectra[:, n])) / pulses.shape[0]
            target_overlap = self.target_psd - spectra[:, n]
            scores[1] += abs(sum(target_overlap[target_overlap < 0])) / pulses.shape[0]
            # Wave score
            # scores[0] += np.linalg.norm(spectra / np.linalg.norm(spectra) - clutter_spectra) / pulses.shape[0]
            # MI Score
            # scores[1] += (psd_freqs[1] - psd_freqs[0]) * sum(
            #     np.log(1 + (self.target_psd * spectra) / (clutter_spectra * spectra + self.noise_sigma)) - \
            #     self.target_psd * spectra / (self.target_psd * spectra + clutter_spectra * spectra + self.noise_sigma)) \
            #              / pulses.shape[0]
            # Correlation Score
            # scores[2] += (abs(np.correlate(spectra / np.linalg.norm(spectra), self.target_psd))[0] -
            #               abs(np.correlate(spectra / np.linalg.norm(spectra), clutter_spectra))[0]) / pulses.shape[0]
        # Cross Correlation Score
        scores[3] = 1 - 2 * np.linalg.norm(np.corrcoef(pulses) - np.eye(pulses.shape[0]))

        # Log everything for plots
        self.log.append([pulses, clutter_spectra, scores[0], scores[1], scores[2], scores[3], sum(scores)])

        # Final bookkeeping
        ret_st = {'clutter_psd': next_clutter_spectra.reshape(self.spec_sz, 1),
                  'target_psd': self.target_psd.reshape(self.spec_sz, 1),
                  'waveform_psd': spectra}
        self.step += 1
        self.pnum += 1
        done = self.step >= self.max_episode_timesteps()
        if self.pnum >= self.sdr[0].nframes:
            done = 2
        return ret_st, done, sum(scores)

    def reset(self, num_parallels=None):
        self.step = 0
        self.log = []
        _, clutter_spectra = welch(self.sdr.getPulse(self.step, 0), self.fs, 'flattop', nperseg=self.spec_sz,
                                   return_onesided=False)
        clutter_spectra /= np.linalg.norm(clutter_spectra)
        return {'clutter_psd': clutter_spectra.reshape(self.spec_sz, 1),
                'target_psd': self.target_psd.reshape(self.spec_sz, 1),
                'waveform_psd': np.zeros((self.spec_sz, self.n_ants))}

    def setParams(self, sdr, nr=None, fs=None, fc=None, bandwidth=None, spectrum_sz=512, target_psd=None):
        """
        Set the parameters of this environment for a new waveform train
        :param sdr: SDRParse object
        :param nr: (int) Number of samples in pulse
        :param fs: (float) Sampling frequency
        :param fc: (float) Center frequency
        :param bandwidth: (float) Bandwidth of signal
        :param spectrum_sz: (int) Size of power spectral density spectrum
        :param target_psd: (ndarray) Target PSD spectrum to train against
        :return: Nuttin'. Just sets stuff.
        """
        self.sdr = sdr
        self.nr = sdr[0].pulse_length if nr is None else nr
        self.fs = sdr[0].fs if fs is None else fs
        self.fc = sdr[0].fc if fc is None else fc
        self.bw = sdr[0].bw if bandwidth is None else bandwidth
        self.log = []
        self.spec_sz = spectrum_sz
        self.step = 0
        self.pnum = 0
        self.target_psd = target_psd if target_psd is not None else self.genTargetPSD()

    def genTargetPSD(self, sz_m=15, alpha=None):
        """
        Generates a target power spectral density using a bunch of random params
        :param sz_m: (float) Radial size of the target in meters.
        :param alpha: (ndarray) Array of shape parameters. Must be 0, .5, or 1 in each element.
        :return: Normalized power spectral density.
        """
        # Number of bins occupied by target
        M = int(2 * sz_m * self.bw / c0)
        freqs = np.fft.fftfreq(self.spec_sz, 1 / self.fs)
        # Range of target
        rng = np.random.uniform(self.rngs[0], self.rngs[1]) + c0 / (2 * self.bw) * np.arange(M)
        # Shape parameters fo individual scatterers
        alpha = np.random.choice([0, .5, 1], M) if alpha is None else alpha
        # Complex electrical field amplitude
        Am = np.random.rand(M) + 1j * np.random.rand(M)
        # Get a center frequency for the target response
        t_fc = self.fc + self.bw / 2 * np.random.uniform(-1, 1)
        # Overall spectrum of target response given the above parameters
        psd = abs(np.sum([Am[n] / rng[n] ** 4 * (1j * freqs / t_fc) ** alpha[n] *
                          np.exp(-1j * 4 * np.pi * freqs / c0 * rng[n]) for n in range(M)], axis=0))
        return psd / np.linalg.norm(psd)
