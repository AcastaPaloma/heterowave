"""Acquisition-aware permutation-invariant HeteroWave model."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from heterowave.data.masks import validate_sector_mask
from heterowave.physics import unfiltered_backprojection

from .blocks import ResidualBlock
from .set_stats import AggregationMode, MaskedSetAggregator


def sector_geometry(
    num_angles: int,
    num_sectors: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if num_angles % num_sectors != 0:
        raise ValueError("num_angles must be divisible by num_sectors")
    angles_per_sector = num_angles // num_sectors
    centers = (torch.arange(num_sectors, device=device, dtype=dtype) + 0.5) * (
        torch.pi / num_sectors
    )
    widths = torch.full_like(centers, angles_per_sector / num_angles)
    return torch.stack((centers.sin(), centers.cos(), widths), dim=1)


def sector_backprojections(
    sinogram: Tensor,
    sector_mask: Tensor,
    *,
    num_sectors: int = 16,
    output_size: int | None = None,
    align_corners: bool = False,
) -> Tensor:
    """Backproject each sector independently to ``[B,S,1,H,W]``."""
    if sinogram.ndim != 3 or not sinogram.is_floating_point():
        raise ValueError("sinogram must be a floating-point [B,A,D] tensor")
    batch, num_angles, detector_bins = sinogram.shape
    mask = validate_sector_mask(sector_mask, num_sectors=num_sectors).to(sinogram.device)
    if mask.shape[0] != batch or num_angles % num_sectors != 0:
        raise ValueError("Batch sizes must match and angles must divide evenly into sectors")
    angles_per_sector = num_angles // num_sectors
    # Grid sampling is kept in FP32; neural encoding may still use autocast.
    with torch.autocast(device_type=sinogram.device.type, enabled=False):
        physics_sinogram = sinogram.float()
        all_angles = torch.arange(num_angles, device=sinogram.device, dtype=torch.float32) * (
            torch.pi / num_angles
        )
        partials = []
        for sector in range(num_sectors):
            start, stop = sector * angles_per_sector, (sector + 1) * angles_per_sector
            partial = unfiltered_backprojection(
                physics_sinogram[:, start:stop],
                angles=all_angles[start:stop],
                output_size=output_size or detector_bins,
                align_corners=align_corners,
            )
            # Each sector spans only 1/S of the full angular integral.
            partial = partial * (angles_per_sector / num_angles)
            partials.append(partial)
    stacked = torch.stack(partials, dim=1)
    return stacked * mask[:, :, None, None, None].to(stacked.dtype)


class HeteroWave(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 128,
        num_angles: int = 64,
        num_sectors: int = 16,
        channels: tuple[int, ...] | list[int] = (16, 32, 64, 96),
        aggregation: AggregationMode = "mean_var_count",
        geometry_channels: bool = True,
        uncertainty: bool = False,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        widths = tuple(channels)
        if len(widths) < 2 or min(widths) < 1:
            raise ValueError("channels must contain at least two positive widths")
        if num_angles % num_sectors != 0:
            raise ValueError("num_angles must be divisible by num_sectors")
        self.image_size = image_size
        self.num_angles = num_angles
        self.num_sectors = num_sectors
        self.geometry_channels = geometry_channels
        self.uncertainty = uncertainty
        self.align_corners = align_corners
        input_channels = 4 if geometry_channels else 1
        self.encoders = nn.ModuleList()
        previous = input_channels
        for index, width in enumerate(widths):
            self.encoders.append(ResidualBlock(previous, width, stride=1 if index == 0 else 2))
            previous = width
        self.aggregators = nn.ModuleList(MaskedSetAggregator(width, aggregation) for width in widths)
        self.decoders = nn.ModuleList(
            ResidualBlock(widths[index] + widths[index - 1], widths[index - 1])
            for index in range(len(widths) - 1, 0, -1)
        )
        self.output = nn.Conv2d(widths[0], 2 if uncertainty else 1, 1)

    def _default_geometry(self, reference: Tensor) -> Tensor:
        return sector_geometry(
            self.num_angles,
            self.num_sectors,
            device=reference.device,
            dtype=reference.dtype,
        )

    def forward_partials(
        self,
        partials: Tensor,
        sector_mask: Tensor,
        *,
        geometry: Tensor | None = None,
    ) -> Tensor | dict[str, Tensor]:
        if partials.ndim != 5 or partials.shape[1:3] != (self.num_sectors, 1):
            raise ValueError("partials must have shape [B,S,1,H,W]")
        mask = validate_sector_mask(sector_mask, num_sectors=self.num_sectors).to(partials.device)
        if mask.shape[0] != partials.shape[0]:
            raise ValueError("partials and sector_mask batch sizes must match")
        if geometry is None:
            geometry = self._default_geometry(partials)
        if geometry.shape != (self.num_sectors, 3):
            raise ValueError("geometry must have shape [S,3]")
        if self.geometry_channels:
            geometry_images = geometry.to(partials).view(1, self.num_sectors, 3, 1, 1).expand(
                partials.shape[0], -1, -1, partials.shape[-2], partials.shape[-1]
            )
            hidden = torch.cat((partials, geometry_images), dim=2)
        else:
            hidden = partials

        aggregated_scales = []
        batch, sectors = partials.shape[:2]
        for encoder, aggregator in zip(self.encoders, self.aggregators):
            flat = hidden.reshape(batch * sectors, hidden.shape[2], hidden.shape[3], hidden.shape[4])
            flat = encoder(flat)
            hidden = flat.reshape(batch, sectors, flat.shape[1], flat.shape[2], flat.shape[3])
            aggregated_scales.append(aggregator(hidden, mask))

        decoded = aggregated_scales[-1]
        for decoder, skip in zip(self.decoders, reversed(aggregated_scales[:-1])):
            decoded = F.interpolate(decoded, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            decoded = decoder(torch.cat((decoded, skip), dim=1))
        output = self.output(decoded)
        if not self.uncertainty:
            return output
        return {"mean": output[:, :1], "log_variance": output[:, 1:2].clamp(-8.0, 3.0)}

    def forward(self, sinogram: Tensor, sector_mask: Tensor) -> Tensor | dict[str, Tensor]:
        if sinogram.shape[1] != self.num_angles:
            raise ValueError(f"Expected {self.num_angles} projection angles")
        partials = sector_backprojections(
            sinogram,
            sector_mask,
            num_sectors=self.num_sectors,
            output_size=self.image_size,
            align_corners=self.align_corners,
        )
        return self.forward_partials(partials, sector_mask)
