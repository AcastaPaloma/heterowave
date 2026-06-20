"""Memory-mapped Phase 3 cache datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


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
