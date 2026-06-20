"""Detector-axis filters used by filtered backprojection."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def ramp_filter(sinogram: Tensor) -> Tensor:
    """Apply a zero-padded Ram-Lak ramp along the last sinogram dimension."""
    if sinogram.ndim != 3 or not sinogram.is_floating_point():
        raise ValueError("sinogram must be a floating-point [B, A, D] tensor")
    detector_bins = sinogram.shape[-1]
    padded_size = max(64, 2 ** ((2 * detector_bins - 1).bit_length()))
    pad_total = padded_size - detector_bins
    left = pad_total // 2
    padded = F.pad(sinogram, (left, pad_total - left))
    frequencies = torch.fft.rfftfreq(padded_size, d=1.0, device=sinogram.device).to(sinogram.dtype)
    response = 2.0 * frequencies.abs()
    filtered = torch.fft.irfft(torch.fft.rfft(padded, dim=-1) * response, n=padded_size, dim=-1)
    return filtered[..., left : left + detector_bins]

