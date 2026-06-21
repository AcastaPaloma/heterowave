"""Shared baseline and HeteroWave training helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .baselines import fbp_unet_features
from .config import ProjectConfig
from .data import (
    CachedHeteroWaveDataset,
    SyntheticReconstructionDataset,
    generate_fixed_validation_masks,
    load_validation_masks,
    sample_sector_masks,
)
from .metrics import ReconstructionMetricAccumulator
from .models import FBPUNet, HeteroWave


def resolve_device(requested: str) -> torch.device:
    if requested != "cuda":
        return torch.device(requested)
    if not torch.cuda.is_available():
        print("warning=CUDA requested but torch.cuda.is_available() is false; using CPU")
        return torch.device("cpu")
    try:
        probe = torch.ones(1, device="cuda")
        probe.add_(1.0)
        torch.cuda.synchronize()
        return torch.device("cuda")
    except (RuntimeError, AssertionError) as error:
        print(f"warning=CUDA execution probe failed ({error}); using CPU")
    return torch.device("cpu")


def create_datasets(config: ProjectConfig):
    if config.data.dataset == "cached":
        train = CachedHeteroWaveDataset(config.data.root, "train")
        validation = CachedHeteroWaveDataset(config.data.root, "val")
    else:
        train = SyntheticReconstructionDataset(
            config.data.train_samples, config.data.image_size, config.physics.num_angles, seed=config.seed
        )
        validation = SyntheticReconstructionDataset(
            config.data.val_samples, config.data.image_size, config.physics.num_angles, seed=config.seed + 1
        )
    if train.metadata["image_size"] != config.data.image_size:
        raise ValueError("Configured image size does not match the dataset cache")
    if train.metadata["num_angles"] != config.physics.num_angles:
        raise ValueError("Configured angle count does not match the dataset cache")
    configured_bins = config.physics.detector_bins or config.data.image_size
    if train.metadata["detector_bins"] != configured_bins:
        raise ValueError("Configured detector-bin count does not match the dataset cache")
    cached_sectors = int(train.metadata.get("num_sectors", config.physics.num_sectors))
    if cached_sectors != config.physics.num_sectors:
        raise ValueError("Configured sector count does not match the dataset cache")
    return train, validation


def create_loaders(config: ProjectConfig):
    train, validation = create_datasets(config)
    common = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
        "persistent_workers": config.data.persistent_workers,
    }
    generator = torch.Generator().manual_seed(config.seed)
    return (
        DataLoader(train, shuffle=True, generator=generator, **common),
        DataLoader(validation, shuffle=False, **common),
        train.metadata,
    )


def build_model(config: ProjectConfig) -> nn.Module:
    if config.model.name == "fbp_unet":
        return FBPUNet(channels=config.model.channels, residual_output=config.model.residual_output)
    return HeteroWave(
        image_size=config.data.image_size,
        num_angles=config.physics.num_angles,
        num_sectors=config.physics.num_sectors,
        channels=config.model.channels,
        aggregation=config.model.aggregation,
        geometry_channels=config.model.geometry_channels,
        uncertainty=config.model.uncertainty,
        align_corners=config.physics.align_corners,
    )


def training_prediction(
    model: nn.Module,
    sinogram: torch.Tensor,
    metadata: dict[str, Any],
    config: ProjectConfig,
    mask_generator: torch.Generator,
) -> torch.Tensor:
    if config.model.name == "fbp_unet":
        with torch.no_grad(), torch.autocast(device_type=sinogram.device.type, enabled=False):
            features = fbp_unet_features(sinogram.float(), metadata)
        return model(features)
    sector_mask = sample_sector_masks(
        len(sinogram),
        num_sectors=config.physics.num_sectors,
        minimum_sectors=config.masking.minimum_sectors,
        random_probability=config.masking.random_probability,
        wedge_probability=config.masking.wedge_probability,
        periodic_probability=config.masking.periodic_probability,
        generator=mask_generator,
    ).to(sinogram.device)
    output = model(sinogram, sector_mask)
    return output["mean"] if isinstance(output, dict) else output


def _validation_masks(config: ProjectConfig) -> list[torch.Tensor]:
    masks = (
        load_validation_masks(config.masking.validation_masks)
        if config.masking.validation_masks
        else generate_fixed_validation_masks(num_sectors=config.physics.num_sectors, seed=config.seed)
    )
    return list(masks.values())


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    metadata: dict[str, Any],
    device: torch.device,
    config: ProjectConfig,
) -> dict[str, float]:
    model.eval()
    metrics = ReconstructionMetricAccumulator()
    fixed_masks = _validation_masks(config) if config.model.name == "heterowave" else []
    sample_offset = 0
    for batch in loader:
        target = batch["target"].to(device, non_blocking=True)
        sinogram = batch["sinogram"].to(device, non_blocking=True)
        if config.model.name == "fbp_unet":
            prediction = model(fbp_unet_features(sinogram, metadata))
        else:
            sector_mask = torch.stack(
                [fixed_masks[(sample_offset + index) % len(fixed_masks)] for index in range(len(sinogram))]
            ).to(device)
            output = model(sinogram, sector_mask)
            prediction = output["mean"] if isinstance(output, dict) else output
            sample_offset += len(sinogram)
        metrics.update(prediction, target)
    return metrics.compute()
