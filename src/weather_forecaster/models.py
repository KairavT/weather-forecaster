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
