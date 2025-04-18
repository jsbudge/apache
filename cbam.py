from math import log2
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F

"""
Classes in this file taken from 
https://bobmcdear.github.io/posts/attention-in-vision/#convolutional-block-attention-module
where they are broken down in detail.
"""


class CBAMChannelAttention(nn.Module):
    """
    CBAM's channel attention module.

    Args:
        in_dim (int): Number of input channels.
        reduction_factor (int): Reduction factor for the 
        bottleneck layer.
        Default is 16.
    """

    def __init__(
            self,
            in_dim: int,
            reduction_factor: int = 16,
    ) -> None:
        super().__init__()

        bottleneck_dim = in_dim // reduction_factor
        self.mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=bottleneck_dim,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=bottleneck_dim,
                out_channels=in_dim,
                kernel_size=1,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        avg_pooled = F.adaptive_avg_pool2d(input, 1)
        max_pooled = F.adaptive_avg_pool2d(input, 1)

        avg_attention = self.mlp(avg_pooled)
        max_attention = self.mlp(max_pooled)

        attention = avg_attention + max_attention
        attention = F.sigmoid(attention)

        return attention * input


def channel_avg_pool(input: torch.Tensor) -> torch.Tensor:
    """									
    Average pool along the channel axis.

    Args:
        input (torch.Tensor): Input to average pool.

    Returns (torch.Tensor): Input average pooled over the channel axis.
    """
    return input.mean(dim=1, keepdim=True)


def channel_max_pool(input: torch.Tensor) -> torch.Tensor:
    """
    Max pool along the channel axis.

    Args:
        input (torch.Tensor): Input to max pool.

    Returns (torch.Tensor): Input max pooled over the channel axis.
    """
    return input.max(dim=1, keepdim=True).values


class CBAMSpatialAttention(nn.Module):
    """
    CBAM's spatial attention.

    Args:
        kernel_size (int): Kernel size of the convolution.
        Default is 7.
    """

    def __init__(
            self,
            kernel_size: int = 7,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        avg_pooled = channel_avg_pool(input)
        max_pooled = channel_max_pool(input)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)

        attention = self.conv(pooled)
        attention = F.sigmoid(attention)

        return attention * input


class CBAM(nn.Module):
    """
    Convolutional block attention module.

    Args:
        in_dim (int): Number of input channels.
        reduction_factor (int): Reduction factor for the
        bottleneck layer of the channel attention module.
        Default is 16.
        kernel_size (int): Kernel size for the convolution
        of the spatial attention module.
        Default is 7.
    """

    def __init__(
            self,
            in_dim: int,
            reduction_factor: int = 16,
            kernel_size: int = 7,
    ) -> None:
        super().__init__()

        self.channel_attention = CBAMChannelAttention(
            in_dim=in_dim,
            reduction_factor=reduction_factor,
        )
        self.spatial_attention = CBAMSpatialAttention(kernel_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.channel_attention(input)
        output = self.spatial_attention(output)
        return output


class GETheta(nn.Module):
    """
    Gather-excite with parameters, including for the excite unit.

    Args:
        in_dim (int): Number of input channels.
        extent (int): Extent. 0 for a global
        extent.
        reduction_factor (int): Reduction factor for the 
        bottleneck layer of the excite module.
        Default is 16.
        spatial_dim (Optional[Union[Tuple[int, int], int]]):
        Spatial dimension of the input, required for a global 
        extent.
        Default is None.
    """

    def __init__(
            self,
            in_dim: int,
            extent: int,
            reduction_factor: int = 16,
            spatial_dim: Optional[Union[Tuple[int, int], int]] = None,
    ) -> None:
        super().__init__()

        if extent == 0:
            self.gather = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_dim,
                    out_channels=in_dim,
                    kernel_size=spatial_dim,
                    groups=in_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(in_dim),
            )

        else:
            n_layers = int(log2(extent))
            layers = n_layers * [
                nn.Conv2d(
                    in_channels=in_dim,
                    out_channels=in_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(),
            ]
            layers = layers[:-1]
            self.gather = nn.Sequential(*layers)

        bottleneck_dim = in_dim // reduction_factor
        self.mlp = nn.Sequential(
            nn.Conv2d(
                in_channels=in_dim,
                out_channels=bottleneck_dim,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=bottleneck_dim,
                out_channels=in_dim,
                kernel_size=1,
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        gathered = self.gather(input)
        attention = self.mlp(gathered)
        attention = F.interpolate(
            input=attention,
            size=input.shape[-2:],
        )
        attention = F.sigmoid(attention)

        return attention * input