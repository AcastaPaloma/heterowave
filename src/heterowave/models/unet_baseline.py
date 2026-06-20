"""Small fixed-input FBP + U-Net baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.convolution1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.normalization1 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.convolution2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.normalization2 = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.skip = nn.Identity() if input_channels == output_channels else nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = F.silu(self.normalization1(self.convolution1(inputs)))
        hidden = self.normalization2(self.convolution2(hidden))
        return F.silu(hidden + self.skip(inputs))


class FBPUNet(nn.Module):
    """U-Net mapping three fixed FBP/coverage channels to normalized speed."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        channels: tuple[int, ...] | list[int] = (16, 32, 64, 96),
        residual_output: bool = True,
    ) -> None:
        super().__init__()
        widths = tuple(channels)
        if len(widths) < 2 or any(width < 1 for width in widths):
            raise ValueError("channels must contain at least two positive widths")
        self.residual_output = residual_output
        self.encoders = nn.ModuleList()
        previous = input_channels
        for width in widths:
            self.encoders.append(ResidualBlock(previous, width))
            previous = width
        self.decoders = nn.ModuleList(
            ResidualBlock(widths[index] + widths[index - 1], widths[index - 1])
            for index in range(len(widths) - 1, 0, -1)
        )
        self.output = nn.Conv2d(widths[0], 1, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("FBPUNet inputs must have shape [B,3,H,W]")
        skips: list[Tensor] = []
        hidden = inputs
        for index, encoder in enumerate(self.encoders):
            hidden = encoder(hidden)
            skips.append(hidden)
            if index + 1 < len(self.encoders):
                hidden = F.max_pool2d(hidden, 2)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            hidden = decoder(torch.cat((hidden, skip), dim=1))
        prediction = self.output(hidden)
        return prediction + inputs[:, :1] if self.residual_output else prediction
