"""Sector-level acquisition masks for sparse-view tomography."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor


def validate_sector_mask(mask: Tensor, *, num_sectors: int | None = None) -> Tensor:
    if mask.ndim != 2:
        raise ValueError("sector_mask must have shape [B,S]")
    if num_sectors is not None and mask.shape[1] != num_sectors:
        raise ValueError(f"Expected {num_sectors} sectors, got {mask.shape[1]}")
    mask = mask.to(dtype=torch.bool)
    if torch.any(mask.sum(dim=1) < 1):
        raise ValueError("Every sample must retain at least one sector")
    return mask


def sector_mask_to_angle_mask(sector_mask: Tensor, num_angles: int) -> Tensor:
    sector_mask = validate_sector_mask(sector_mask)
    if num_angles % sector_mask.shape[1] != 0:
        raise ValueError("num_angles must be divisible by num_sectors")
    return sector_mask.repeat_interleave(num_angles // sector_mask.shape[1], dim=1)


def _random_mask(num_sectors: int, observed: int, generator: torch.Generator) -> Tensor:
    indices = torch.randperm(num_sectors, generator=generator)[:observed]
    mask = torch.zeros(num_sectors, dtype=torch.bool)
    mask[indices] = True
    return mask


def _contiguous_mask(num_sectors: int, observed: int, generator: torch.Generator) -> Tensor:
    start = int(torch.randint(num_sectors, (), generator=generator))
    mask = torch.zeros(num_sectors, dtype=torch.bool)
    mask[(torch.arange(observed) + start) % num_sectors] = True
    return mask


def _periodic_mask(num_sectors: int, minimum: int, generator: torch.Generator) -> Tensor:
    strides = [stride for stride in (2, 3, 4) if (num_sectors + stride - 1) // stride >= minimum]
    stride = strides[int(torch.randint(len(strides), (), generator=generator))]
    offset = int(torch.randint(stride, (), generator=generator))
    mask = torch.arange(num_sectors).remainder(stride) == offset
    if int(mask.sum()) < minimum:
        mask |= _random_mask(num_sectors, minimum, generator)
    return mask


def sample_sector_masks(
    batch_size: int,
    *,
    num_sectors: int = 16,
    minimum_sectors: int = 2,
    random_probability: float = 0.50,
    wedge_probability: float = 0.35,
    periodic_probability: float = 0.15,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample mixed random, contiguous, and periodic masks on CPU."""
    if not 1 <= minimum_sectors <= num_sectors:
        raise ValueError("minimum_sectors must be within [1,num_sectors]")
    probabilities = torch.tensor([random_probability, wedge_probability, periodic_probability])
    if torch.any(probabilities < 0) or not torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-6):
        raise ValueError("mask probabilities must be nonnegative and sum to one")
    generator = generator or torch.Generator()
    masks = []
    for _ in range(batch_size):
        kind = int(torch.multinomial(probabilities, 1, generator=generator))
        observed = int(torch.randint(minimum_sectors, num_sectors + 1, (), generator=generator))
        if kind == 0:
            mask = _random_mask(num_sectors, observed, generator)
        elif kind == 1:
            mask = _contiguous_mask(num_sectors, observed, generator)
        else:
            mask = _periodic_mask(num_sectors, minimum_sectors, generator)
        masks.append(mask)
    return torch.stack(masks)


def generate_fixed_validation_masks(*, num_sectors: int = 16, seed: int = 1337) -> dict[str, Tensor]:
    if num_sectors != 16:
        raise ValueError("The fixed Phase 5 scenarios currently require 16 sectors")
    generator = torch.Generator().manual_seed(seed)
    scenarios: dict[str, Tensor] = {
        "all_16": torch.ones(num_sectors, dtype=torch.bool),
        "observed_12": _random_mask(num_sectors, 12, generator),
    }
    for observed in (8, 4, 2):
        scenarios[f"random_{observed}"] = _random_mask(num_sectors, observed, generator)
        scenarios[f"contiguous_{observed}"] = _contiguous_mask(num_sectors, observed, generator)
    return scenarios


def save_validation_masks(path: str | Path, *, num_sectors: int = 16, seed: int = 1337) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = generate_fixed_validation_masks(num_sectors=num_sectors, seed=seed)
    payload = {"seed": seed, "num_sectors": num_sectors, "masks": {k: v.int().tolist() for k, v in masks.items()}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def load_validation_masks(path: str | Path) -> dict[str, Tensor]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    masks = {name: torch.tensor(values, dtype=torch.bool) for name, values in payload["masks"].items()}
    for mask in masks.values():
        validate_sector_mask(mask.unsqueeze(0), num_sectors=int(payload["num_sectors"]))
    return masks
