import numpy as np


fs = 2e9
c0 = 299792458.0
TAC = 125e6
DTR = np.pi / 180
inch_to_m = .0254
m_to_ft = 3.2808


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
    ndata[ndata != 0] = (ndata[ndata != 0] - abs(ndata[ndata != 0]).mean()) / ndata[ndata != 0].std()
    return ndata


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