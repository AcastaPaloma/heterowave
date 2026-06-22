"""Small learned primal-dual style baseline using the repo physics operators."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from heterowave.baselines import fbp_normalized_speed, fbp_unet_features
from heterowave.data.masks import sector_mask_to_angle_mask, validate_sector_mask
from heterowave.physics import parallel_beam_project, unfiltered_backprojection

from .unet_baseline import ResidualBlock


class _UpdateNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(in_channels, hidden_channels),
            ResidualBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class LearnedPrimalDual(nn.Module):
    """A lightweight unrolled data-consistency refiner.

    This is not a full Learned Primal-Dual reproduction. It is the small,
    cache-compatible baseline we can train fairly here:

    1. initialize the primal image with masked FBP;
    2. project the current image through the same PyTorch forward operator;
    3. backproject the observed projection residual;
    4. use a learned CNN update to refine the image;
    5. repeat for a small number of iterations.

    The output layer of each update is zero-initialized, so the model starts as
    masked FBP and learns only residual corrections.
    """

    def __init__(
        self,
        *,
        image_size: int = 128,
        num_angles: int = 64,
        num_sectors: int = 16,
        iterations: int = 3,
        hidden_channels: int = 16,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        if min(image_size, num_angles, num_sectors, iterations, hidden_channels) < 1:
            raise ValueError("LearnedPrimalDual parameters must be positive")
        if num_angles % num_sectors != 0:
            raise ValueError("num_angles must be divisible by num_sectors")
        self.image_size = image_size
        self.num_angles = num_angles
        self.num_sectors = num_sectors
        self.iterations = iterations
        self.align_corners = align_corners
        # Channels: current x, initial FBP, coverage, observed fraction, residual backprojection.
        self.updates = nn.ModuleList(_UpdateNet(5, hidden_channels) for _ in range(iterations))
        self.update_gates = nn.ParameterList(nn.Parameter(torch.tensor(1.0)) for _ in range(iterations))

    def _projection_residual_backprojection(
        self,
        x: Tensor,
        sinogram: Tensor,
        angle_mask: Tensor,
        metadata: dict[str, Any],
    ) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            angles = torch.arange(
                int(metadata["num_angles"]),
                device=x.device,
                dtype=torch.float32,
            ) * (torch.pi / int(metadata["num_angles"]))
            speed = x.float() * float(metadata["speed_std"]) + float(metadata["speed_mean"])
            slowness_contrast = speed.clamp_min(1.0).reciprocal() - (1.0 / float(metadata["water_speed"]))
            projection = parallel_beam_project(
                slowness_contrast,
                angles=angles,
                detector_bins=int(metadata["detector_bins"]),
                align_corners=bool(metadata.get("align_corners", False)),
            )
            residual = (projection - sinogram.float()) * angle_mask.unsqueeze(-1).to(dtype=torch.float32)
            residual_bp = unfiltered_backprojection(
                residual.float(),
                angles=angles,
                output_size=self.image_size,
                align_corners=self.align_corners,
            )
            scale = residual_bp.flatten(1).abs().mean(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
            return (residual_bp / scale).to(dtype=x.dtype)

    def forward(
        self,
        sinogram: Tensor,
        sector_mask: Tensor,
        metadata: dict[str, Any],
    ) -> Tensor:
        if sinogram.ndim != 3:
            raise ValueError("sinogram must have shape [B,A,D]")
        sector_mask = validate_sector_mask(sector_mask, num_sectors=self.num_sectors).to(sinogram.device)
        angle_mask = sector_mask_to_angle_mask(sector_mask, sinogram.shape[1]).to(sinogram.device)
        with torch.no_grad(), torch.autocast(device_type=sinogram.device.type, enabled=False):
            fbp = fbp_normalized_speed(sinogram.float(), metadata, angle_mask=angle_mask)
            features = fbp_unet_features(sinogram.float(), metadata, angle_mask=angle_mask)
        coverage = features[:, 1:2]
        observed_fraction = features[:, 2:3]
        x = fbp
        for gate, update in zip(self.update_gates, self.updates):
            residual_bp = self._projection_residual_backprojection(x, sinogram, angle_mask, metadata)
            update_input = torch.cat((x, fbp, coverage, observed_fraction, residual_bp), dim=1)
            x = x + torch.tanh(gate) * update(update_input)
        return x
