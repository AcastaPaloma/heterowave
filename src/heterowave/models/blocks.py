"""Shared neural blocks for acquisition-aware reconstruction."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


def group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(group_count(output_channels), output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(output_channels), output_channels)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == 1
            else nn.Conv2d(input_channels, output_channels, 1, stride=stride)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.silu(self.norm1(self.conv1(inputs)))
        return F.silu(self.norm2(self.conv2(hidden)) + self.skip(inputs))
