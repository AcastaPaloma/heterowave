"""Core reconstruction metrics used by Phase 4 baselines."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def structural_similarity(prediction: Tensor, target: Tensor) -> Tensor:
    """Return a compact local-window SSIM estimate for image batches."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must be matching [B,C,H,W] tensors")
    window = min(11, prediction.shape[-2], prediction.shape[-1])
    if window % 2 == 0:
        window -= 1
    padding = window // 2
    mean_x = F.avg_pool2d(prediction.float(), window, stride=1, padding=padding)
    mean_y = F.avg_pool2d(target.float(), window, stride=1, padding=padding)
    variance_x = F.avg_pool2d(prediction.float().square(), window, 1, padding) - mean_x.square()
    variance_y = F.avg_pool2d(target.float().square(), window, 1, padding) - mean_y.square()
    covariance = F.avg_pool2d(prediction.float() * target.float(), window, 1, padding) - mean_x * mean_y
    data_range = (target.float().amax() - target.float().amin()).clamp_min(1e-4)
    c1, c2 = (0.01 * data_range).square(), (0.03 * data_range).square()
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2).clamp_min(1e-12)
    )
    return score.mean()


class ReconstructionMetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.target_square = 0.0
        self.target_min = float("inf")
        self.target_max = float("-inf")
        self.ssim_sum = 0.0
        self.samples = 0

    def update(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        prediction, target = prediction.float(), target.float()
        error = prediction - target
        self.count += target.numel()
        self.absolute_error += float(error.abs().sum())
        self.squared_error += float(error.square().sum())
        self.target_square += float(target.square().sum())
        self.target_min = min(self.target_min, float(target.amin()))
        self.target_max = max(self.target_max, float(target.amax()))
        self.ssim_sum += float(structural_similarity(prediction, target)) * target.shape[0]
        self.samples += target.shape[0]

    def compute(self) -> dict[str, float]:
        if not self.count:
            raise ValueError("No samples were accumulated")
        mae = self.absolute_error / self.count
        rmse = math.sqrt(self.squared_error / self.count)
        target_rms = max(math.sqrt(self.target_square / self.count), 1e-8)
        data_range = max(self.target_max - self.target_min, 1e-8)
        psnr = 20.0 * math.log10(data_range / max(rmse, 1e-8))
        return {
            "mae": mae,
            "rmse": rmse,
            "nrmse": rmse / target_rms,
            "psnr": psnr,
            "ssim": self.ssim_sum / self.samples,
        }


def reconstruction_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    error = prediction.float() - target.float()
    mae = error.abs().mean()
    rmse = error.square().mean().sqrt()
    target_rms = target.float().square().mean().sqrt().clamp_min(1e-8)
    data_range = (target.float().amax() - target.float().amin()).clamp_min(1e-8)
    psnr = 20.0 * torch.log10(data_range / rmse.clamp_min(1e-8))
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "nrmse": float(rmse / target_rms),
        "psnr": float(psnr) if math.isfinite(float(psnr)) else float("inf"),
        "ssim": float(structural_similarity(prediction, target)),
    }
