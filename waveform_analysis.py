import numpy as np
import pandas as pd
from simulib.simulation_functions import db, findPowerOf2, genPulse
from scipy.signal.windows import taylor
import matplotlib.pyplot as plt


c0 = 299792458.0
TAC = 125e6
fs = 2e9
DTR = np.pi / 180


def ambiguity(s1, s2, prf, dopp_bins, a_fs, mag=True, normalize=True, t_cutoff=None):
    fdopp = np.linspace(-prf / 2, prf / 2, dopp_bins)
    fft_sz = findPowerOf2(len(s1))
    s1f = np.tile(np.fft.fft(s1, fft_sz).conj(), (dopp_bins, 1))
    t, f = np.meshgrid(np.arange(len(s2)) / a_fs, np.linspace(-prf / 2, prf / 2, dopp_bins))
    sg = np.exp(2j * np.pi * f * t)
    sg = np.fft.fft(sg * s2, fft_sz, axis=1)
    A = np.fft.fftshift(np.fft.ifft(sg * s1f, axis=1), axes=1)
    if normalize:
        A = A / abs(A).max()
    if t_cutoff is not None:
        A = A[:, A.shape[1] // 2 - t_cutoff:A.shape[1] // 2 + t_cutoff]
    return abs(A) if mag else A, fdopp, (np.arange(fft_sz) - fft_sz // 2) * c0 / a_fs


def autocorrelation(s, s2=None, upsample=8, use_window=False):
    s_fft = np.fft.fft(s, findPowerOf2(len(s)))
    if s2 is None:
        s_corr = s_fft * s_fft.conj()

    else:
        s2_fft = np.fft.fft(s2, findPowerOf2(len(s))).conj()
        s_corr = s_fft * s2_fft
    ret = np.zeros(len(s_fft) * upsample, dtype=np.complex128)
    ret[:len(s_corr) // 2] = s_corr[:len(s_corr) // 2]
    ret[-len(s_corr) // 2:] = s_corr[-len(s_corr) // 2:]
    return np.fft.ifft(ret)


if __name__ == '__main__':
    pulse = genPulse(np.linspace(0, 1, 10), np.linspace(0, 1, 10), 26757, 2e9, 1e10, 445e6)
    amb = ambiguity(pulse, pulse, 500e6, 2048, 2e9)
    plt.figure('unext')
    plt.imshow(db(amb[0]))
    plt.axis('tight')
    fft_len = findPowerOf2(len(pulse)) * 2

    pulse_fft = np.fft.fft(pulse, fft_len)
    pulse_fft_ext = np.insert(pulse_fft, np.arange(0, fft_len, 1), 0)

    plt.figure()
    plt.plot(np.fft.fftfreq(fft_len, 1 / fs), db(pulse_fft))
    plt.plot(np.fft.fftfreq(len(pulse_fft_ext), 1 / fs), db(pulse_fft_ext))

    pulse_ext = np.fft.ifft(pulse_fft)[:len(pulse) * 2]

    plt.figure('ext_auto')
    plt.plot(np.fft.fftshift(db(autocorrelation(pulse))))
    plt.plot(np.arange(0, fft_len * 4, .5), np.fft.fftshift(db(autocorrelation(pulse_ext))))
