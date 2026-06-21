"""Acquisition perturbations for water-coupled ultrasonic CT training."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import AcquisitionConfig


def _randn(
    shape: tuple[int, ...],
    reference: Tensor,
    generator: torch.Generator | None,
) -> Tensor:
    values = torch.randn(shape, generator=generator, dtype=torch.float32)
    return values.to(device=reference.device, dtype=reference.dtype)


def _shift_detectors(sinogram: Tensor, shifts: Tensor) -> Tensor:
    """Apply differentiable fractional shifts along detector bins."""
    batch, angles, detector_bins = sinogram.shape
    if detector_bins < 2:
        return sinogram
    y = torch.linspace(-1.0, 1.0, angles, device=sinogram.device, dtype=sinogram.dtype)
    x = torch.linspace(-1.0, 1.0, detector_bins, device=sinogram.device, dtype=sinogram.dtype)
    yy = y.view(1, angles, 1).expand(batch, -1, detector_bins)
    xx = x.view(1, 1, detector_bins).expand(batch, angles, -1)
    shift = (2.0 * shifts.to(sinogram) / float(detector_bins - 1)).unsqueeze(-1)
    grid = torch.stack((xx - shift, yy), dim=-1)
    shifted = F.grid_sample(
        sinogram.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return shifted.squeeze(1)


def augment_sinogram(
    sinogram: Tensor,
    config: AcquisitionConfig,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Apply small, explicit measurement perturbations during training only.

    These are not "water noise" assumptions. They model realistic acquisition
    nuisances around a water-coupled ring scanner: channel gain variation,
    low-level additive electronics/calibration bias, stochastic measurement
    noise, and tiny detector-axis timing/bin shifts.
    """
    if not config.enabled:
        return sinogram
    if min(config.noise_std, config.gain_std, config.bias_std, config.detector_shift_std) < 0:
        raise ValueError("acquisition perturbation standard deviations must be nonnegative")
    if config.noise_std == config.gain_std == config.bias_std == config.detector_shift_std == 0:
        return sinogram
    if sinogram.ndim != 3 or not sinogram.is_floating_point():
        raise ValueError("sinogram must be a floating-point [B,A,D] tensor")

    augmented = sinogram
    batch, angles, detector_bins = sinogram.shape
    scale = sinogram.detach().float().abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6).to(sinogram)

    if config.gain_std > 0:
        angle_gain = _randn((batch, angles, 1), sinogram, generator)
        detector_gain = _randn((batch, 1, detector_bins), sinogram, generator)
        gain = 1.0 + config.gain_std * 0.5 * (angle_gain + detector_gain)
        augmented = augmented * gain
    if config.bias_std > 0:
        angle_bias = _randn((batch, angles, 1), sinogram, generator)
        detector_bias = _randn((batch, 1, detector_bins), sinogram, generator)
        augmented = augmented + config.bias_std * scale * 0.5 * (angle_bias + detector_bias)
    if config.noise_std > 0:
        augmented = augmented + config.noise_std * scale * _randn(tuple(sinogram.shape), sinogram, generator)
    if config.detector_shift_std > 0:
        shifts = config.detector_shift_std * _randn((batch, angles), sinogram, generator)
        augmented = _shift_detectors(augmented, shifts)
    return augmented
