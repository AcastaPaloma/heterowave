"""FBPConvNet-style sparse-view artifact-correction baseline."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FBPConvNet(nn.Module):
    """Residual U-Net post-processor for masked/sparse FBP images.

    This is intentionally simpler than the existing fair ``masked_fbp_unet``:
    it receives only the sparse-view FBP image, matching the classic
    FBPConvNet-style "analytic inverse then image-domain artifact removal"
    protocol. Coverage/observed-fraction information is withheld so this is a
    clean external-style baseline rather than another mask-aware HeteroWave
    variant.
    """

    def __init__(
        self,
        *,
        channels: tuple[int, ...] | list[int] = (16, 32, 64, 96),
        residual_output: bool = True,
    ) -> None:
        super().__init__()
        widths = tuple(channels)
        if len(widths) < 2 or min(widths) < 1:
            raise ValueError("channels must contain at least two positive widths")
        self.residual_output = residual_output
        self.encoders = nn.ModuleList()
        previous = 1
        for width in widths:
            self.encoders.append(_ConvBlock(previous, width))
            previous = width
        self.decoders = nn.ModuleList(
            _ConvBlock(widths[index] + widths[index - 1], widths[index - 1])
            for index in range(len(widths) - 1, 0, -1)
        )
        self.output = nn.Conv2d(widths[0], 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, fbp: Tensor) -> Tensor:
        if fbp.ndim != 4 or fbp.shape[1] != 1:
            raise ValueError("FBPConvNet expects a [B,1,H,W] FBP image")
        skips = []
        hidden = fbp
        for index, encoder in enumerate(self.encoders):
            hidden = encoder(hidden)
            skips.append(hidden)
            if index + 1 < len(self.encoders):
                hidden = F.max_pool2d(hidden, 2)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            hidden = decoder(torch.cat((hidden, skip), dim=1))
        residual = self.output(hidden)
        return residual + fbp if self.residual_output else residual
