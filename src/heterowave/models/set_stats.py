"""Permutation-invariant masked HeMIS statistics."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

AggregationMode = Literal["mean", "mean_var", "mean_var_count"]


def masked_set_statistics(features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Aggregate ``[B,S,C,H,W]`` features with population variance."""
    if features.ndim != 5 or mask.shape != features.shape[:2]:
        raise ValueError("features and mask must have shapes [B,S,C,H,W] and [B,S]")
    weights = mask.to(device=features.device, dtype=features.dtype).view(*mask.shape, 1, 1, 1)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (features * weights).sum(dim=1) / count
    variance = ((features - mean.unsqueeze(1)).square() * weights).sum(dim=1) / count
    variance = variance.clamp_min(0.0)
    normalized_count = (count / features.shape[1]).expand(-1, 1, features.shape[-2], features.shape[-1])
    return mean, variance, normalized_count


class MaskedSetAggregator(nn.Module):
    def __init__(self, channels: int, mode: AggregationMode = "mean_var_count") -> None:
        super().__init__()
        if mode not in {"mean", "mean_var", "mean_var_count"}:
            raise ValueError(f"Unsupported aggregation mode: {mode}")
        self.mode = mode
        multiplier = 1 if mode == "mean" else 2
        input_channels = channels * multiplier + (1 if mode == "mean_var_count" else 0)
        self.reduce = nn.Conv2d(input_channels, channels, 1)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        mean, variance, count = masked_set_statistics(features, mask)
        values = [mean]
        if self.mode != "mean":
            values.append(variance)
        if self.mode == "mean_var_count":
            values.append(count)
        return self.reduce(torch.cat(values, dim=1))
