"""Phase 4 reconstruction losses."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .data.masks import sector_mask_to_angle_mask
from .physics import parallel_beam_project


def gradient_loss(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must be matching [B,C,H,W] tensors")
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    true_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    true_y = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(pred_x, true_x) + F.l1_loss(pred_y, true_y)


def reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    image_weight: float = 1.0,
    gradient_weight: float = 0.1,
    log_variance: Tensor | None = None,
    uncertainty_weight: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    image = F.smooth_l1_loss(prediction, target)
    gradient = gradient_loss(prediction, target)
    total = image_weight * image + gradient_weight * gradient
    parts = {"image_loss": image.detach(), "gradient_loss": gradient.detach()}
    if log_variance is not None:
        if log_variance.shape != target.shape:
            raise ValueError("log_variance and target shapes must match")
        uncertainty = 0.5 * (torch.exp(-log_variance) * (prediction - target).square() + log_variance)
        uncertainty = uncertainty.mean()
        total = total + uncertainty_weight * uncertainty
        parts["uncertainty_nll"] = uncertainty.detach()
    elif uncertainty_weight > 0:
        raise ValueError("log_variance is required when uncertainty_weight is positive")
    parts["loss"] = total.detach()
    return total, parts


def observed_data_consistency_loss(
    prediction: Tensor,
    sinogram: Tensor,
    sector_mask: Tensor,
    metadata: dict[str, Any],
    *,
    angle_fraction: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """L1 projection residual over a sampled subset of observed angles."""
    if prediction.ndim != 4 or prediction.shape[1] != 1 or sinogram.ndim != 3:
        raise ValueError("Expected prediction [B,1,H,W] and sinogram [B,A,D]")
    if not 0 < angle_fraction <= 1:
        raise ValueError("angle_fraction must be within (0,1]")
    angle_mask = sector_mask_to_angle_mask(sector_mask, sinogram.shape[1]).to(sinogram.device)
    if angle_fraction < 1:
        sampled = torch.zeros_like(angle_mask)
        for batch_index, observed in enumerate(angle_mask.cpu()):
            indices = observed.nonzero(as_tuple=False).flatten()
            count = max(1, round(len(indices) * angle_fraction))
            selected = indices[torch.randperm(len(indices), generator=generator)[:count]]
            sampled[batch_index, selected.to(sampled.device)] = True
        angle_mask = sampled
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        speed = prediction.float() * float(metadata["speed_std"]) + float(metadata["speed_mean"])
        slowness_contrast = speed.clamp_min(1.0).reciprocal() - (1.0 / float(metadata["water_speed"]))
        reprojection = parallel_beam_project(
            slowness_contrast,
            num_angles=int(metadata["num_angles"]),
            detector_bins=int(metadata["detector_bins"]),
            align_corners=bool(metadata.get("align_corners", False)),
        )
        weights = angle_mask.unsqueeze(-1).expand_as(sinogram).to(reprojection.dtype)
        count = weights.sum().clamp_min(1)
        residual = ((reprojection - sinogram.float()).abs() * weights).sum() / count
        # Slowness line integrals are O(1e-4); normalize so the configured
        # Phase-7 weight has a meaningful scale while preserving the same optimum.
        target_scale = (sinogram.float().abs() * weights).sum() / count
        return residual / target_scale.clamp_min(1e-8)
