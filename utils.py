import math
import matplotlib.pyplot as plt
from typing import Union, List
from matplotlib.lines import Line2D
from scipy.signal.windows import taylor
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
            nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
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


def getMatchedFilter(chirp, bw, fs, fc, fft_len):
    twin = taylor(int(np.round(bw / fs * fft_len)))
    taytay = np.zeros(fft_len, dtype=np.complex128)
    winloc = int((fc % fs) * fft_len / fs) - len(twin) // 2
    if winloc + len(twin) > fft_len:
        taytay[winloc:fft_len] += twin[:fft_len - winloc]
        taytay[:len(twin) - (fft_len - winloc)] += twin[fft_len - winloc:]
    else:
        taytay[winloc:winloc + len(twin)] += twin

    return taytay / (np.fft.fft(chirp, fft_len) + 1e-12)


class GrowingCosine(nn.Module):
    def forward(self, x):
        return x * torch.cos(x)


class ELiSH(nn.Module):
    def forward(self, x):
        return torch.where(x > 0, x * torch.sigmoid(x), (torch.exp(x) - 1) * torch.sigmoid(x))


class SinLU(nn.Module):
    def forward(self, x):
        return (x + torch.sin(x)) * torch.sigmoid(x)


class ParameterSinLU(nn.Module):
    def __init__(self):
        super(ParameterSinLU,self).__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))
    def forward(self,x):
        return torch.sigmoid(x)*(x+self.a*torch.sin(self.b*x))


nonlinearities = {'silu': nn.SiLU(), 'gelu': nn.GELU(), 'selu': nn.SELU(), 'leaky': nn.LeakyReLU(),
                  'grow': GrowingCosine(), 'elish': ELiSH(), 'sinlu': SinLU(), 'psinlu': ParameterSinLU(),
                  'mish': nn.Mish()}


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = {}
    max_grads = {}
    layers = {}
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            module_name = n.split('.')[0]
            if module_name in layers:
                layers[module_name].append(n)
                ave_grads[module_name].append(p.grad.abs().mean().cpu().numpy())
                max_grads[module_name].append(p.grad.abs().max().cpu().numpy())
            else:
                layers[module_name] = [n]
                ave_grads[module_name] = [p.grad.abs().mean().cpu().numpy()]
                max_grads[module_name] = [p.grad.abs().max().cpu().numpy()]
    grid_sz = int(np.sqrt(len(layers)) + 1)
    for idx, mn in enumerate(layers.keys()):
        plt.subplot(grid_sz, grid_sz, idx + 1)
        plt.title(mn)
        plt.bar(np.arange(len(max_grads[mn])), max_grads[mn], alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads[mn])), ave_grads[mn], alpha=0.1, lw=1, color="b")
        '''plt.hlines(0, 0, len(ave_grads[mn]) + 1, lw=2, color="k")
        plt.xticks(range(len(ave_grads[mn])), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads[mn]))
        # plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(False)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])'''
    plt.show()


class FAMO:
    """
    Fast Adaptive Multitask Optimization.
    """
    prev_loss: float = 0.

    def __init__(
            self,
            n_tasks: int,
            device: torch.device,
            gamma: float = 0.01,  # the regularization coefficient
            w_lr: float = 0.025,  # the learning rate of the task logits
            max_norm: float = 1.0,  # the maximum gradient norm
    ):
        self.min_losses = torch.zeros(n_tasks).to(device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.device = device

    def set_min_losses(self, losses):
        self.min_losses = losses

    def get_weighted_loss(self, losses):
        self.prev_loss = losses
        z = nn_func.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        return (D.log() * z / c).sum()

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(nn_func.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()

    def backward(
            self,
            losses: torch.Tensor,
            shared_parameters: Union[
                List[torch.nn.parameter.Parameter], torch.Tensor
            ] = None,
    ) -> Union[torch.Tensor, None]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        Returns
        -------
        Loss, extra outputs
        """
        loss = self.get_weighted_loss(losses=losses)
        loss.backward()
        if self.max_norm > 0 and shared_parameters is not None:
            torch.nn.utils.clip_grad_norm_(shared_parameters, self.max_norm)
        return loss