from typing import List, Any, TypeVar
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.nn import functional as F

Tensor = TypeVar('torch.tensor')


class RichConv1d(LightningModule):
    def __init__(self, in_channels, out_channels, out_layer_sz, activation='leaky'):
        super(RichConv1d, self).__init__()
        self.branch0 = nn.Conv1d(in_channels, out_channels, 4, 2, 1)
        self.branch1 = nn.Conv1d(in_channels, out_channels, out_layer_sz + 3, 1, 1)
        self.branch2 = nn.Conv1d(in_channels, out_channels, out_layer_sz + 1, 1, 0)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        if self.activation == 'elu':
            x = torch.add(torch.add(F.elu(self.branch0(x)), F.elu(self.branch1(x))),
                          F.elu(self.branch2(x)))
        elif self.activation == 'tanh':
            x = torch.add(torch.add(F.tanh(self.branch0(x)), F.tanh(self.branch1(x))),
                          F.tanh(self.branch2(x)))
        else:
            x = torch.add(torch.add(F.leaky_relu(self.branch0(x)), F.leaky_relu(self.branch1(x))),
                          F.leaky_relu(self.branch2(x)))
        return self.batch_norm(x)


class RichConvTranspose1d(LightningModule):
    def __init__(self, in_channels, out_channels, in_layer_sz, activation='leaky'):
        super(RichConvTranspose1d, self).__init__()
        self.branch0 = nn.ConvTranspose1d(in_channels, out_channels, 4, 2, 1)
        self.branch1 = nn.ConvTranspose1d(in_channels, out_channels, in_layer_sz + 3, 1, 1)
        self.branch2 = nn.ConvTranspose1d(in_channels, out_channels, in_layer_sz + 1, 1, 0)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = activation

    def forward(self, x):
        if self.activation == 'elu':
            x = torch.add(torch.add(F.elu(self.branch0(x)), F.elu(self.branch1(x))),
                          F.elu(self.branch2(x)))
        elif self.activation == 'tanh':
            x = torch.add(torch.add(F.tanh(self.branch0(x)), F.tanh(self.branch1(x))),
                          F.tanh(self.branch2(x)))
        else:
            x = torch.add(torch.add(F.leaky_relu(self.branch0(x)), F.leaky_relu(self.branch1(x))),
                          F.leaky_relu(self.branch2(x)))
        return self.batch_norm(x)


class RichConvTranspose2d(LightningModule):
    def __init__(self, in_channels, out_channels, in_layer_sz, activation='leaky'):
        super(RichConvTranspose2d, self).__init__()
        self.branch0 = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        self.branch1 = nn.ConvTranspose2d(in_channels, out_channels, in_layer_sz + 3, 1, 1)
        self.branch2 = nn.ConvTranspose2d(in_channels, out_channels, in_layer_sz + 1, 1, 0)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.linear_final = Linear2d(in_layer_sz * 2, in_layer_sz * 2, out_channels)

    def forward(self, x):
        if self.activation == 'elu':
            x = torch.add(torch.add(F.elu(self.branch0(x)), F.elu(self.branch1(x))),
                          F.elu(self.branch2(x)))
        elif self.activation == 'tanh':
            x = torch.add(torch.add(F.tanh(self.branch0(x)), F.tanh(self.branch1(x))),
                          F.tanh(self.branch2(x)))
        else:
            x = torch.add(torch.add(F.leaky_relu(self.branch0(x)), F.leaky_relu(self.branch1(x))),
                          F.leaky_relu(self.branch2(x)))
        x = self.linear_final(x)
        return self.batch_norm(x)


class RichConv2d(LightningModule):
    def __init__(self, in_channels, out_channels, out_layer_sz, activation='leaky', maintain_shape=False):
        super(RichConv2d, self).__init__()
        if maintain_shape:
            # Assumes an even shape
            b0_sz = out_layer_sz // 2 + 1
            b1_sz = out_layer_sz // 4 + 1
            self.branch0 = nn.Conv2d(in_channels, out_channels, b0_sz, 1, b0_sz // 2)
            self.branch1 = nn.Conv2d(in_channels, out_channels, b1_sz, 1, b1_sz // 2)
            self.branch2 = nn.Conv2d(in_channels, out_channels, 5, 1, 2)
        else:
            self.branch0 = nn.Conv2d(in_channels, out_channels, 4, 2, 1)
            self.branch1 = nn.Conv2d(in_channels, out_channels, out_layer_sz + 3, 1, 1)
            self.branch2 = nn.Conv2d(in_channels, out_channels, out_layer_sz + 1, 1, 0)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.linear_final = Linear2d(out_layer_sz, out_layer_sz, out_channels)
        self.activation = activation

    def forward(self, x):
        if self.activation == 'elu':
            x = torch.add(torch.add(F.elu(self.branch0(x)), F.elu(self.branch1(x))),
                          F.elu(self.branch2(x)))
        elif self.activation == 'tanh':
            x = torch.add(torch.add(F.tanh(self.branch0(x)), F.tanh(self.branch1(x))),
                          F.tanh(self.branch2(x)))
        else:
            x = torch.add(torch.add(F.leaky_relu(self.branch0(x)), F.leaky_relu(self.branch1(x))),
                          F.leaky_relu(self.branch2(x)))
        x = self.linear_final(x)
        return self.batch_norm(x)


class Linear2d(LightningModule):
    def __init__(self, width, height, nchan):
        super(Linear2d, self).__init__()
        self.width = width
        self.height = height
        self.nchan = nchan
        self.total_weights = width * height * nchan
        self.linear = nn.Linear(self.total_weights, self.total_weights)

    def forward(self, x):
        x = x.view(-1, self.total_weights)
        x = self.linear(x)
        x = F.leaky_relu(x)
        x = x.view(-1, self.nchan, self.width, self.height)
        return x


class SelfAttention(LightningModule):
    '''
    This is taken from a Medium article that references Self-Attention Generative Adversarial Networks:
    https://arxiv.org/pdf/1805.08318.pdf
    '''

    def __init__(self, n_channels, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.query = nn.Conv1d(n_channels, n_channels // 8, kernel_size=1, bias=False)
        self.key = nn.Conv1d(n_channels, n_channels // 8, kernel_size=1, bias=False)
        self.value = nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        # Notation from paper.
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class STFTAttention(LightningModule):
    '''
    This is taken from a Medium article that references Self-Attention Generative Adversarial Networks:
    https://arxiv.org/pdf/1805.08318.pdf
    '''

    def __init__(self, n_channels, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.query = nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)
        self.key = nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(1, n_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        f, g = self.query(x[:, :, 0, :]), self.key(x[:, :, :, 0])
        beta = self.value(F.softmax(torch.bmm(g.transpose(1, 2), f), dim=1).unsqueeze(1))
        o = self.gamma * beta + x
        return o.contiguous()


class LSTMAttention(nn.Module):
    def __init__(self):
        super(LSTMAttention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch_size, seq_len, hidden_dim)
        # decoder_hidden: (batch_size, hidden_dim)

        # Calculate the attention scores.
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)

        attn_weights = F.softmax(scores, dim=1)  # (batch_size, seq_len)

        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim)

        return context_vector, attn_weights


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, \
            "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out


class BandwidthAttention(LightningModule):

    def __init__(self):
        super(BandwidthAttention, self).__init__()

    def forward(self, x, bin_bw):
        window = torch.zeros((x.shape[0], x.shape[2]), device=self.device)
        window[:, :bin_bw // 2] = 1.
        window[:, -bin_bw // 2:] = 1.
        return window


class BandwidthEncoder(LightningModule):
    def __init__(self, fs, nbins):
        super(BandwidthEncoder, self).__init__()
        self.fs = fs
        self.nbins = nbins

    def forward(self, batch, bandwidth):
        encoder = torch.zeros((batch, self.nbins), device=self.device)
        encoder[torch.arange(0, batch, 1, device=self.device),
        torch.floor((bandwidth / self.fs * self.nbins)).type(torch.int)] = 1.
        return encoder


class MMExpand(LightningModule):

    def __init__(self, inner_channels, outer_channels, step_sz, win_sz):
        super(MMExpand, self).__init__()
        self.channels = nn.Parameter(torch.ones(1, inner_channels, device=self.device))
        self.inner_channels = inner_channels
        self.step_sz = step_sz
        self.win_sz = win_sz
        self.norm = nn.BatchNorm2d(1)

    def forward(self, x, nframes):
        pos_encoding = torch.linspace(0, nframes * self.step_sz, nframes, device=self.device).view(
            1, nframes, 1) / (nframes * self.step_sz + self.win_sz)
        y = torch.tile(pos_encoding * self.channels, (x.shape[0], 1, 1))
        return self.norm(torch.bmm(y, x).unsqueeze(1))


class Windower(LightningModule):

    def __init__(self, win_sz, fs, win_type='hann'):
        super(Windower, self).__init__()
        self.win_sz = win_sz
        self.fs = fs

    def forward(self, bwidth):
        bin_bw = int((bwidth / self.fs) * self.win_sz)
        bin_bw += 1 if bin_bw % 2 != 0 else 0
        win = torch.zeros((1, 1, self.win_sz, 1), device=self.device)
        w = torch.windows.hann(bin_bw, device=self.device)
        win[0, 0, :bin_bw // 2, 0] = w[-bin_bw // 2:]
        win[0, 0, -bin_bw // 2:, 0] = w[:bin_bw // 2]
        return win


class ISTFT(LightningModule):
    def __init__(self, win_sz, fft_sz=2048):
        super(ISTFT, self).__init__()
        self.M = win_sz
        self.N = fft_sz
        self.h = torch.windows.hann(fft_sz, device=self.device)
        self.step_size = fft_sz - win_sz - 1

    def forward(self, x, nr):
        pos = 0
        y = torch.zeros((x.shape[0], nr), dtype=torch.complex64, device=self.device)
        while pos + self.N <= x.shape[2]:
            yt = torch.fft.ifft()
