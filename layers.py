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