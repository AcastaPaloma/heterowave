"""Reliability-attention HeteroWave model for sparse-view ultrasonic CT."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from heterowave.data.masks import validate_sector_mask

from .heterowave import sector_backprojections, sector_geometry
from .set_stats import AggregationMode, MaskedSetAggregator
from .unet_baseline import ResidualBlock


def expand_sector_geometry(geometry: Tensor, angle_fourier_bands: int = 0) -> Tensor:
    """Append higher-frequency angular Fourier features to ``[sin θ, cos θ, width]``.

    The base ``sector_geometry`` encoding already contains first-harmonic angle
    features, so ``angle_fourier_bands=2`` appends sin/cos at 2θ and 4θ.
    """
    if geometry.ndim != 2 or geometry.shape[1] < 3:
        raise ValueError("geometry must have shape [S,3+] with sin, cos, width first")
    if angle_fourier_bands < 0:
        raise ValueError("angle_fourier_bands must be nonnegative")
    if angle_fourier_bands == 0:
        return geometry
    angle = torch.atan2(geometry[:, 0], geometry[:, 1])
    features = [geometry]
    for band in range(angle_fourier_bands):
        frequency = float(2 ** (band + 1))
        features.append(torch.sin(frequency * angle).unsqueeze(1))
        features.append(torch.cos(frequency * angle).unsqueeze(1))
    return torch.cat(features, dim=1)


class MaskGeometryEncoder(nn.Module):
    """Encode global angular mask shape for decoder FiLM conditioning."""

    def __init__(
        self,
        *,
        num_sectors: int,
        fourier_bands: int = 4,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        if num_sectors < 1 or fourier_bands < 0 or embedding_dim < 1:
            raise ValueError("mask geometry parameters must be positive")
        self.num_sectors = num_sectors
        self.fourier_bands = fourier_bands
        centers = (torch.arange(num_sectors, dtype=torch.float32) + 0.5) * (torch.pi / num_sectors)
        self.register_buffer("centers", centers, persistent=False)
        input_dim = num_sectors + 1 + 2 * fourier_bands
        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )

    def forward(self, sector_mask: Tensor) -> Tensor:
        mask = validate_sector_mask(sector_mask, num_sectors=self.num_sectors)
        values = mask.to(dtype=self.centers.dtype, device=self.centers.device)
        features = [values, values.mean(dim=1, keepdim=True)]
        for band in range(1, self.fourier_bands + 1):
            angles = self.centers * float(band)
            features.append((values * angles.sin()).mean(dim=1, keepdim=True))
            features.append((values * angles.cos()).mean(dim=1, keepdim=True))
        return self.net(torch.cat(features, dim=1))


class MaskedSectorAttention(nn.Module):
    """Permutation-invariant sector fusion with tempered spatial precision maps."""

    def __init__(
        self,
        channels: int,
        *,
        aggregation: AggregationMode = "mean_var_count",
        include_statistics: bool = True,
        precision_temperature: float = 0.5,
        precision_max: float = 10.0,
    ) -> None:
        super().__init__()
        if precision_temperature <= 0 or precision_max <= 0:
            raise ValueError("precision_temperature and precision_max must be positive")
        hidden = max(channels // 4, 1)
        self.include_statistics = include_statistics
        self.min_precision = 1e-4
        self.precision_temperature = float(precision_temperature)
        self.precision_max = float(precision_max)
        self.precision = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        self.value = nn.Conv2d(channels, channels, 1)
        extra_channels = channels + 3 if include_statistics else 0
        self.reduce = nn.Conv2d(channels + extra_channels, channels, 1)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        if features.ndim != 5 or mask.shape != features.shape[:2]:
            raise ValueError("features and mask must have shapes [B,S,C,H,W] and [B,S]")
        mask = validate_sector_mask(mask, num_sectors=features.shape[1]).to(features.device)
        batch, sectors, channels, height, width = features.shape
        flat = features.reshape(batch * sectors, channels, height, width)
        logits = self.precision(flat).reshape(batch, sectors, 1, height, width)
        precision = F.softplus(logits) + self.min_precision
        precision = precision.clamp(max=self.precision_max).pow(self.precision_temperature)
        precision = precision * mask.view(batch, sectors, 1, 1, 1).to(precision)
        total_precision = precision.sum(dim=1).clamp_min(self.min_precision)
        weights = precision / total_precision.unsqueeze(1)
        values = self.value(flat).reshape(batch, sectors, channels, height, width)
        attended = (weights * values).sum(dim=1)
        if self.include_statistics:
            variance = (weights * (values - attended.unsqueeze(1)).square()).sum(dim=1)
            effective_count = total_precision.square() / precision.square().sum(dim=1).clamp_min(
                self.min_precision
            )
            effective_count = (effective_count / float(sectors)).clamp(0.0, 1.0)
            observed_fraction = (
                mask.to(attended).mean(dim=1).view(batch, 1, 1, 1).expand(-1, 1, height, width)
            )
            log_precision = torch.log1p(total_precision)
            attended = torch.cat((attended, variance, effective_count, observed_fraction, log_precision), dim=1)
        return self.reduce(attended)


class HeteroWaveV3(nn.Module):
    """Masked-FBP trunk with reliability-attentive sector evidence.

    The global trunk intentionally keeps the same parameter names as
    ``FBPUNet``/``HeteroWaveV2``. With zero fusion gates and matching trunk
    weights, the model starts exactly as the masked FBP U-Net; all v3 behavior
    then enters through learnable, ablatable sector gates.
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
        attention_fusion: bool = True,
        sector_statistics: bool = True,
        angle_fourier_bands: int = 0,
        mask_film: bool = True,
        mask_fourier_bands: int = 4,
        mask_embedding_dim: int = 64,
        precision_temperature: float = 0.5,
        precision_max: float = 10.0,
        fusion_gate_init: float = 0.0,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        widths = tuple(channels)
        if len(widths) < 2 or min(widths) < 1:
            raise ValueError("channels must contain at least two positive widths")
        if num_angles % num_sectors != 0:
            raise ValueError("num_angles must be divisible by num_sectors")
        if angle_fourier_bands < 0:
            raise ValueError("angle_fourier_bands must be nonnegative")
        self.image_size = image_size
        self.num_angles = num_angles
        self.num_sectors = num_sectors
        self.geometry_channels = geometry_channels
        self.residual_output = residual_output
        self.uncertainty = uncertainty
        self.sector_fusion = sector_fusion
        self.attention_fusion = attention_fusion
        self.sector_statistics = sector_statistics
        self.angle_fourier_bands = angle_fourier_bands
        self.mask_film = mask_film
        self.align_corners = align_corners

        # Names intentionally match FBPUNet for direct warm starts.
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

        geometry_width = 3 + 2 * angle_fourier_bands
        sector_input_channels = 1 + geometry_width if geometry_channels else 1
        self.sector_encoders = nn.ModuleList()
        previous = sector_input_channels
        for width in widths:
            self.sector_encoders.append(ResidualBlock(previous, width))
            previous = width
        if attention_fusion:
            self.aggregators = nn.ModuleList(
                MaskedSectorAttention(
                    width,
                    aggregation=aggregation,
                    include_statistics=sector_statistics,
                    precision_temperature=precision_temperature,
                    precision_max=precision_max,
                )
                for width in widths
            )
        else:
            self.aggregators = nn.ModuleList(MaskedSetAggregator(width, aggregation) for width in widths)
        self.fusions = nn.ModuleList(nn.Conv2d(2 * width, width, 1) for width in widths)
        self.fusion_gates = nn.ParameterList(
            nn.Parameter(torch.tensor(float(fusion_gate_init))) for _ in widths
        )
        self.mask_encoder = (
            MaskGeometryEncoder(
                num_sectors=num_sectors,
                fourier_bands=mask_fourier_bands,
                embedding_dim=mask_embedding_dim,
            )
            if mask_film
            else None
        )
        decoder_widths = tuple(widths[index - 1] for index in range(len(widths) - 1, 0, -1))
        self.decoder_films = nn.ModuleList(nn.Linear(mask_embedding_dim, 2 * width) for width in decoder_widths)
        for film in self.decoder_films:
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)

    def _default_geometry(self, reference: Tensor) -> Tensor:
        geometry = sector_geometry(
            self.num_angles,
            self.num_sectors,
            device=reference.device,
            dtype=reference.dtype,
        )
        return expand_sector_geometry(geometry, self.angle_fourier_bands)

    def _normalize_geometry(self, geometry: Tensor, reference: Tensor) -> Tensor:
        geometry = geometry.to(reference)
        expected_width = 3 + 2 * self.angle_fourier_bands
        if geometry.shape == (self.num_sectors, 3) and self.angle_fourier_bands:
            geometry = expand_sector_geometry(geometry, self.angle_fourier_bands)
        if geometry.shape != (self.num_sectors, expected_width):
            raise ValueError(f"geometry must have shape [{self.num_sectors},{expected_width}]")
        return geometry

    def _sector_scales(
        self,
        partials: Tensor,
        sector_mask: Tensor,
        geometry: Tensor | None = None,
    ) -> list[Tensor]:
        if partials.ndim != 5 or partials.shape[1:3] != (self.num_sectors, 1):
            raise ValueError("partials must have shape [B,S,1,H,W]")
        mask = validate_sector_mask(sector_mask, num_sectors=self.num_sectors).to(partials.device)
        if mask.shape[0] != partials.shape[0]:
            raise ValueError("partials and sector_mask batch sizes must match")
        if geometry is None:
            geometry = self._default_geometry(partials)
        else:
            geometry = self._normalize_geometry(geometry, partials)
        if self.geometry_channels:
            geometry_images = geometry.view(1, self.num_sectors, geometry.shape[1], 1, 1).expand(
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
        mask_code = self.mask_encoder(sector_mask) if self.mask_encoder is not None else None
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
        for decoder_index, (decoder, skip) in enumerate(zip(self.decoders, reversed(skips[:-1]))):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            hidden = decoder(torch.cat((hidden, skip), dim=1))
            if mask_code is not None:
                gamma, beta = self.decoder_films[decoder_index](mask_code).view(
                    hidden.shape[0], 2, hidden.shape[1], 1, 1
                ).unbind(dim=1)
                hidden = hidden * (1.0 + gamma) + beta
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
