import math

import numpy as np
import torch
from torch import nn

fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808


def _xavier_init(model):
    """
    Performs the Xavier weight initialization.
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)
            # nn.init.he_(module.weight)


def upsample(val, fac=8):
    upval = np.zeros(len(val) * fac, dtype=np.complex128)
    upval[:len(val) // 2] = val[:len(val) // 2]
    upval[-len(val) // 2:] = val[-len(val) // 2:]
    return upval


def normalize(data):
    return data / np.expand_dims(np.sqrt(np.sum(data * data.conj(), axis=-1).real), axis=len(data.shape) - 1)


def getRange(alt, theta_el):
    return alt * np.sin(theta_el) * 2 / c0


def scale_normalize(data):
    ndata = data / np.expand_dims(np.sqrt(np.sum(data * data.conj(), axis=-1).real), axis=len(data.shape) - 1)
    mask = ndata != 0
    mus = np.mean(abs(ndata), where=mask, axis=-1)
    sigs = np.std(ndata, where=mask, axis=-1)
    ndata = (ndata - mask * mus[..., None]) / sigs[..., None]
    return ndata


def get_radar_coeff(fc, ant_transmit_power, rx_gain, tx_gain, rec_gain):
    return (
            c0 ** 2 / fc ** 2 * ant_transmit_power * 10 ** ((rx_gain + 2.15) / 10) * 10 ** (
                (tx_gain + 2.15) / 10) *
            10 ** ((rec_gain + 2.15) / 10) / (4 * np.pi) ** 3)


def get_pslr(a):
    """
    Gets Peak Sidelobe Ratio for a signal.
    :param a: NxM tensor, where N is the batch size and M is the number of samples. Expects them to be real.
    :return: Nx1 tensor of PSLR values.
    """
    gpu_temp1 = a[:, 1:-1] - a[:, :-2]
    gpu_temp2 = a[:, 1:-1] - a[:, 2:]

    # and checking where both shifts are positive;
    out1 = torch.where(gpu_temp1 > 0, gpu_temp1 * 0 + 1, gpu_temp1 * 0)
    out2 = torch.where(gpu_temp2 > 0, out1, gpu_temp2 * 0)

    # argrelmax containing all peaks
    argrelmax_gpu = torch.nonzero(out2, out=None)
    peaks = [a[argrelmax_gpu[argrelmax_gpu[:, 0] == n, 0], argrelmax_gpu[argrelmax_gpu[:, 0] == n, 1]] for n in
             range(a.shape[0])]
    return torch.stack([abs(torch.topk(p, 2).values.diff()) for p in peaks])


def narrow_band(signal, lag=None, n_fbins=None):
    """Narrow band ambiguity function.

    :param signal: Signal to be analyzed.
    :param lag: vector of lag values.
    :param n_fbins: number of frequency bins
    :type signal: array-like
    :type lag: array-like
    :type n_fbins: int
    :return: Doppler lag representation
    :rtype: array-like
    """

    n = signal.shape[0]
    if lag is None:
        if n % 2 == 0:
            tau_start, tau_end = -n / 2 + 1, n / 2
        else:
            tau_start, tau_end = -(n - 1) / 2, (n + 1) / 2
        lag = np.arange(tau_start, tau_end)
    taucol = lag.shape[0]

    if n_fbins is None:
        n_fbins = signal.shape[0]

    naf = np.zeros((n_fbins, taucol), dtype=complex)
    for icol in range(taucol):
        taui = int(lag[icol])
        t = np.arange(abs(taui), n - abs(taui)).astype(int)
        naf[t, icol] = signal[t + taui] * np.conj(signal[t - taui])
    naf = np.fft.fft(naf, axis=0)

    _ix1 = np.arange((n_fbins + (n_fbins % 2)) // 2, n_fbins)
    _ix2 = np.arange((n_fbins + (n_fbins % 2)) // 2)

    _xi1 = -(n_fbins - (n_fbins % 2)) // 2
    _xi2 = ((n_fbins + (n_fbins % 2)) // 2 - 1)
    xi = np.arange(_xi1, _xi2 + 1, dtype=float) / n_fbins
    naf = naf[np.hstack((_ix1, _ix2)), :]
    return naf, lag, xi