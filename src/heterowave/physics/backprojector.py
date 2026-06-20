"""Differentiable unfiltered and filtered parallel-beam backprojection."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from ._geometry import rotation_grids, validate_angles
from .filters import ramp_filter


def unfiltered_backprojection(
    sinogram: Tensor,
    *,
    angles: Tensor | None = None,
    output_size: int | None = None,
    align_corners: bool = False,
) -> Tensor:
    """Backproject ``[B, A, D]`` into normalized images ``[B, 1, H, W]``."""
    if sinogram.ndim != 3 or not sinogram.is_floating_point():
        raise ValueError("sinogram must be a floating-point [B, A, D] tensor")
    batch, num_angles, detector_bins = sinogram.shape
    if num_angles < 1:
        raise ValueError("sinogram must contain at least one angle")
    size = output_size or detector_bins
    if size < 2:
        raise ValueError("output_size must be >= 2")
    angles = validate_angles(angles, num_angles, sinogram)
    if angles.numel() != num_angles:
        raise ValueError("angles length must match the sinogram angle dimension")
    rays = sinogram.reshape(batch * num_angles, 1, 1, detector_bins)
    if detector_bins != size:
        rays = F.interpolate(rays, size=(1, size), mode="bilinear", align_corners=False) * (detector_bins / size)
    slabs = rays.expand(-1, -1, size, -1)
    grids = rotation_grids(size, angles, inverse=True, align_corners=align_corners)
    expanded_grids = grids[None].expand(batch, -1, -1, -1, -1).reshape(-1, size, size, 2)
    rotated = F.grid_sample(slabs, expanded_grids, mode="bilinear", padding_mode="zeros", align_corners=align_corners)
    return rotated.reshape(batch, num_angles, 1, size, size).sum(dim=1) * (torch.pi / (2.0 * num_angles))


def filtered_backprojection(
    sinogram: Tensor,
    *,
    angles: Tensor | None = None,
    output_size: int | None = None,
    align_corners: bool = False,
) -> Tensor:
    """Apply a Ram-Lak ramp filter followed by normalized backprojection."""
    return unfiltered_backprojection(
        ramp_filter(sinogram), angles=angles, output_size=output_size, align_corners=align_corners
    )

