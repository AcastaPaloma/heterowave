"""FBP reconstruction and fixed U-Net feature construction."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .physics import filtered_backprojection, unfiltered_backprojection


def _angle_mask(mask: Tensor | None, sinogram: Tensor) -> Tensor:
    batch, angles, _ = sinogram.shape
    if mask is None:
        return torch.ones((batch, angles), device=sinogram.device, dtype=sinogram.dtype)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(batch, -1)
    if mask.shape != (batch, angles):
        raise ValueError(f"angle_mask must have shape [{batch},{angles}]")
    return mask.to(device=sinogram.device, dtype=sinogram.dtype)


def fbp_normalized_speed(
    sinogram: Tensor,
    metadata: dict[str, Any],
    *,
    angle_mask: Tensor | None = None,
) -> Tensor:
    """Reconstruct cached slowness sinograms as normalized speed maps."""
    mask = _angle_mask(angle_mask, sinogram)
    delta_slowness = filtered_backprojection(
        sinogram * mask.unsqueeze(-1),
        output_size=int(metadata["image_size"]),
        align_corners=bool(metadata.get("align_corners", False)),
    )
    water_speed = float(metadata["water_speed"])
    slowness = (delta_slowness + (1.0 / water_speed)).clamp(min=1.0 / 3000.0, max=1.0 / 750.0)
    speed = slowness.reciprocal()
    return (speed - float(metadata["speed_mean"])) / float(metadata["speed_std"])


def fbp_unet_features(
    sinogram: Tensor,
    metadata: dict[str, Any],
    *,
    angle_mask: Tensor | None = None,
) -> Tensor:
    """Build FBP, angular coverage, and observed-count input channels."""
    mask = _angle_mask(angle_mask, sinogram)
    fbp = fbp_normalized_speed(sinogram, metadata, angle_mask=mask)
    coverage_sinogram = mask.unsqueeze(-1).expand_as(sinogram)
    coverage = unfiltered_backprojection(
        coverage_sinogram,
        output_size=int(metadata["image_size"]),
        align_corners=bool(metadata.get("align_corners", False)),
    )
    coverage = coverage / coverage.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    observed_fraction = mask.mean(dim=1).view(-1, 1, 1, 1).expand_as(fbp)
    return torch.cat((fbp, coverage, observed_fraction), dim=1)
