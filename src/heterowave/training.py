"""Phase 4 baseline data, device, and validation helpers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .baselines import fbp_unet_features
from .config import ProjectConfig
from .data import CachedHeteroWaveDataset, SyntheticReconstructionDataset
from .metrics import ReconstructionMetricAccumulator


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


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    metadata: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics = ReconstructionMetricAccumulator()
    for batch in loader:
        target = batch["target"].to(device, non_blocking=True)
        sinogram = batch["sinogram"].to(device, non_blocking=True)
        prediction = model(fbp_unet_features(sinogram, metadata))
        metrics.update(prediction, target)
    return metrics.compute()
