"""Global-FBP U-Net with optional gated permutation-invariant sector features."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from heterowave.data.masks import validate_sector_mask

from .heterowave import sector_backprojections, sector_geometry
from .set_stats import AggregationMode, MaskedSetAggregator
from .unet_baseline import ResidualBlock


class HeteroWaveV2(nn.Module):
    """A strong masked-FBP U-Net trunk augmented by zero-gated sector features.

    With gates initialized to zero and matching trunk weights, the initial mean
    prediction is exactly the masked FBP U-Net prediction. This makes every
    benefit from the sector path directly attributable and safely ablatable.
    """

    def __init__(
        self,
        *,
        image_size: int = 128,
        num_angles: int = 64,
        num_sectors: int = 16,
        channels: tuple[int, ...] | list[int] = (16, 32, 64, 96),
        aggregation: AggregationMode = "mean_var_count",
        geometry_channels: bool = True,
        residual_output: bool = True,
        uncertainty: bool = False,
        sector_fusion: bool = True,
        fusion_gate_init: float = 0.0,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        widths = tuple(channels)
        if len(widths) < 2 or min(widths) < 1:
            raise ValueError("channels must contain at least two positive widths")
        if num_angles % num_sectors != 0:
            raise ValueError("num_angles must be divisible by num_sectors")
        self.image_size = image_size
        self.num_angles = num_angles
        self.num_sectors = num_sectors
        self.geometry_channels = geometry_channels
        self.residual_output = residual_output
        self.uncertainty = uncertainty
        self.sector_fusion = sector_fusion
        self.align_corners = align_corners

        # Names intentionally match FBPUNet so its trained trunk loads directly.
        self.encoders = nn.ModuleList()
        previous = 3
        for width in widths:
            self.encoders.append(ResidualBlock(previous, width))
            previous = width
        self.decoders = nn.ModuleList(
            ResidualBlock(widths[index] + widths[index - 1], widths[index - 1])
            for index in range(len(widths) - 1, 0, -1)
        )
        self.output = nn.Conv2d(widths[0], 2 if uncertainty else 1, 1)

        sector_input_channels = 4 if geometry_channels else 1
        self.sector_encoders = nn.ModuleList()
        previous = sector_input_channels
        for width in widths:
            self.sector_encoders.append(ResidualBlock(previous, width))
            previous = width
        self.aggregators = nn.ModuleList(MaskedSetAggregator(width, aggregation) for width in widths)
        self.fusions = nn.ModuleList(nn.Conv2d(2 * width, width, 1) for width in widths)
        self.fusion_gates = nn.ParameterList(
            nn.Parameter(torch.tensor(float(fusion_gate_init))) for _ in widths
        )

    def _sector_scales(
        self,
        partials: Tensor,
        sector_mask: Tensor,
        geometry: Tensor | None = None,
    ) -> list[Tensor]:
        if partials.ndim != 5 or partials.shape[1:3] != (self.num_sectors, 1):
            raise ValueError("partials must have shape [B,S,1,H,W]")
        mask = validate_sector_mask(sector_mask, num_sectors=self.num_sectors).to(partials.device)
        if geometry is None:
            geometry = sector_geometry(
                self.num_angles,
                self.num_sectors,
                device=partials.device,
                dtype=partials.dtype,
            )
        if geometry.shape != (self.num_sectors, 3):
            raise ValueError("geometry must have shape [S,3]")
        if self.geometry_channels:
            geometry_images = geometry.to(partials).view(1, self.num_sectors, 3, 1, 1).expand(
                partials.shape[0], -1, -1, partials.shape[-2], partials.shape[-1]
            )
            hidden = torch.cat((partials, geometry_images), dim=2)
        else:
            hidden = partials
        scales = []
        batch, sectors = partials.shape[:2]
        for index, (encoder, aggregator) in enumerate(zip(self.sector_encoders, self.aggregators)):
            flat = hidden.reshape(batch * sectors, hidden.shape[2], hidden.shape[3], hidden.shape[4])
            flat = encoder(flat)
            hidden = flat.reshape(batch, sectors, flat.shape[1], flat.shape[2], flat.shape[3])
            scales.append(aggregator(hidden, mask))
            if index + 1 < len(self.sector_encoders):
                pooled = F.max_pool2d(hidden.flatten(0, 1), 2)
                hidden = pooled.reshape(batch, sectors, pooled.shape[1], pooled.shape[2], pooled.shape[3])
        return scales

    def forward_partials(
        self,
        global_features: Tensor,
        partials: Tensor,
        sector_mask: Tensor,
        *,
        geometry: Tensor | None = None,
    ) -> Tensor | dict[str, Tensor]:
        if global_features.ndim != 4 or global_features.shape[1] != 3:
            raise ValueError("global_features must have shape [B,3,H,W]")
        sector_scales = self._sector_scales(partials, sector_mask, geometry) if self.sector_fusion else None
        skips = []
        hidden = global_features
        for index, encoder in enumerate(self.encoders):
            hidden = encoder(hidden)
            if sector_scales is not None:
                update = self.fusions[index](torch.cat((hidden, sector_scales[index]), dim=1))
                hidden = hidden + torch.tanh(self.fusion_gates[index]) * update
            skips.append(hidden)
            if index + 1 < len(self.encoders):
                hidden = F.max_pool2d(hidden, 2)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            hidden = decoder(torch.cat((hidden, skip), dim=1))
        output = self.output(hidden)
        mean = output[:, :1]
        if self.residual_output:
            mean = mean + global_features[:, :1]
        if not self.uncertainty:
            return mean
        return {"mean": mean, "log_variance": output[:, 1:2].clamp(-8.0, 3.0)}

    def forward(
        self,
        sinogram: Tensor,
        sector_mask: Tensor,
        global_features: Tensor,
    ) -> Tensor | dict[str, Tensor]:
        partials = sector_backprojections(
            sinogram,
            sector_mask,
            num_sectors=self.num_sectors,
            output_size=self.image_size,
            align_corners=self.align_corners,
        )
        return self.forward_partials(global_features, partials, sector_mask)
