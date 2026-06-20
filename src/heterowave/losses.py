"""Phase 4 reconstruction losses."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


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
) -> tuple[Tensor, dict[str, Tensor]]:
    image = F.smooth_l1_loss(prediction, target)
    gradient = gradient_loss(prediction, target)
    total = image_weight * image + gradient_weight * gradient
    return total, {"loss": total.detach(), "image_loss": image.detach(), "gradient_loss": gradient.detach()}
