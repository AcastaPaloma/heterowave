"""Memory-mapped Phase 3 cache datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from heterowave.physics import parallel_beam_project

from .phantoms import make_random_phantoms, speed_to_slowness_contrast


class CachedHeteroWaveDataset(Dataset[dict[str, torch.Tensor]]):
    """Read cached samples lazily without copying the complete arrays into RAM."""

    def __init__(self, root: str | Path, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.root = Path(root)
        self.split = split
        with (self.root / "metadata.json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.targets = np.load(self.root / f"{split}_targets.npy", mmap_mode="r")
        self.sinograms = np.load(self.root / f"{split}_sinograms.npy", mmap_mode="r")
        if len(self.targets) != len(self.sinograms):
            raise ValueError(f"Mismatched {split} target and sinogram counts")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Copies are per-sample and avoid PyTorch warnings about read-only memmaps.
        target = torch.from_numpy(np.array(self.targets[index], dtype=np.float32, copy=True)).unsqueeze(0)
        sinogram = torch.from_numpy(np.array(self.sinograms[index], dtype=np.float32, copy=True))
        return {"target": target, "sinogram": sinogram}


class SyntheticReconstructionDataset(Dataset[dict[str, torch.Tensor]]):
    """Tiny deterministic reconstruction dataset for local smoke tests."""

    def __init__(self, count: int, image_size: int, num_angles: int, *, seed: int = 1337) -> None:
        if min(count, image_size, num_angles) < 1:
            raise ValueError("count, image_size, and num_angles must be positive")
        speed = make_random_phantoms(count, image_size, seed=seed)
        self.targets = (speed - 1500.0) / 50.0
        self.sinograms = parallel_beam_project(
            speed_to_slowness_contrast(speed), num_angles=num_angles, detector_bins=image_size
        )
        self.metadata = {
            "image_size": image_size,
            "num_angles": num_angles,
            "detector_bins": image_size,
            "align_corners": False,
            "speed_mean": 1500.0,
            "speed_std": 50.0,
            "water_speed": 1500.0,
        }

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"target": self.targets[index].clone(), "sinogram": self.sinograms[index].clone()}
