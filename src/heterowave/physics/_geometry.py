"""Shared angle validation and affine-grid caching."""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import functional as F


def default_angles(num_angles: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.arange(num_angles, device=device, dtype=dtype) * (torch.pi / num_angles)


def validate_angles(angles: Tensor | None, num_angles: int | None, reference: Tensor) -> Tensor:
    if angles is None:
        if num_angles is None or num_angles < 1:
            raise ValueError("num_angles must be positive when angles are omitted")
        return default_angles(num_angles, device=reference.device, dtype=reference.dtype)
    if angles.ndim != 1 or angles.numel() < 1:
        raise ValueError("angles must be a nonempty 1D tensor")
    return angles.to(device=reference.device, dtype=reference.dtype)


@lru_cache(maxsize=64)
def _cached_grids(size: int, angle_values: tuple[float, ...], device_type: str, device_index: int | None, dtype: torch.dtype, inverse: bool, align_corners: bool) -> Tensor:
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    angles = torch.tensor(angle_values, device=device, dtype=dtype)
    if inverse:
        angles = -angles
    cosine, sine = torch.cos(angles), torch.sin(angles)
    theta = torch.zeros((angles.numel(), 2, 3), device=device, dtype=dtype)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    return F.affine_grid(theta, (angles.numel(), 1, size, size), align_corners=align_corners)


def rotation_grids(size: int, angles: Tensor, *, inverse: bool, align_corners: bool) -> Tensor:
    values = tuple(float(value) for value in angles.detach().cpu().tolist())
    return _cached_grids(size, values, angles.device.type, angles.device.index, angles.dtype, inverse, align_corners)

