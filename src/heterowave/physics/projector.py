"""Differentiable parallel-beam forward projector."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from ._geometry import rotation_grids, validate_angles


def parallel_beam_project(
    images: Tensor,
    *,
    angles: Tensor | None = None,
    num_angles: int | None = None,
    detector_bins: int | None = None,
    align_corners: bool = False,
) -> Tensor:
    """Project ``[B, 1, H, W]`` square images to sinograms ``[B, A, D]``."""
    if images.ndim != 4 or images.shape[1] != 1 or images.shape[2] != images.shape[3]:
        raise ValueError("images must have shape [B, 1, H, W] with square spatial dimensions")
    if not images.is_floating_point():
        raise TypeError("images must use a floating-point dtype")
    batch, _, size, _ = images.shape
    angles = validate_angles(angles, num_angles, images)
    grids = rotation_grids(size, angles, inverse=False, align_corners=align_corners)
    expanded_images = images[:, None].expand(batch, angles.numel(), 1, size, size).reshape(-1, 1, size, size)
    expanded_grids = grids[None].expand(batch, -1, -1, -1, -1).reshape(-1, size, size, 2)
    rotated = F.grid_sample(expanded_images, expanded_grids, mode="bilinear", padding_mode="zeros", align_corners=align_corners)
    projections = rotated.sum(dim=2).reshape(batch, angles.numel(), size)
    bins = detector_bins or size
    if bins != size:
        projections = F.interpolate(projections, size=bins, mode="linear", align_corners=False) * (size / bins)
    return projections

