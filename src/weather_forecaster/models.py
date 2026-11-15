"""Neural network architectures."""

from __future__ import annotations

import itertools

import torch
from torch import nn


class GlobeConv(nn.Module):
    """A convolution that knows the grid wraps around the globe.

    The column east of the last longitude is the first longitude, so the input is padded
    with its own opposite edge in that direction. Nothing lies beyond the poles, so the
    latitude direction is padded with zeros.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(self.pad, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x[..., -self.pad :], x, x[..., : self.pad]], dim=-1)
        return self.conv(x)


class SimpleCNN(nn.Module):
    """The WeatherBench baseline network: five 5x5 convolutions, 64 channels wide.

    Maps the fields at one time to the same fields at a later time, on the same grid.
    """

    def __init__(self, channels: int, width: int = 64, depth: int = 5, kernel_size: int = 5):
        super().__init__()
        sizes = [channels] + [width] * (depth - 1) + [channels]
        layers: list[nn.Module] = []
        for i, (n_in, n_out) in enumerate(itertools.pairwise(sizes)):
            layers.append(GlobeConv(n_in, n_out, kernel_size))
            if i < depth - 1:
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    """Two convolutions whose result is added to the input, so the block learns a correction."""

    def __init__(self, width: int, kernel_size: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.conv1 = GlobeConv(width, width, kernel_size)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv2 = GlobeConv(width, width, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(nn.functional.silu(self.norm1(x)))
        y = self.conv2(nn.functional.silu(self.norm2(y)))
        return x + y


class ResNet(nn.Module):
    """A deeper network for the stepping model: a stack of residual blocks at one width.

    The input can have more channels than the output, because the network is also told the
    time of day, the season and the map but only predicts the weather. The last layer starts
    at zero, so an untrained network predicts no change at all. That is the persistence
    forecast, a sensible place to start learning from.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 128,
        blocks: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.stem = GlobeConv(in_channels, width, kernel_size)
        self.blocks = nn.Sequential(*(ResBlock(width, kernel_size) for _ in range(blocks)))
        self.norm = nn.GroupNorm(8, width)
        self.head = GlobeConv(width, out_channels, kernel_size)
        nn.init.zeros_(self.head.conv.weight)
        nn.init.zeros_(self.head.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        return self.head(nn.functional.silu(self.norm(x)))
