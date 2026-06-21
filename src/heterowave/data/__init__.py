"""Synthetic and cached dataset utilities."""

from .dataset import CachedHeteroWaveDataset, SyntheticReconstructionDataset
from .mat_loader import ArrayCandidate, inspect_mat, load_speed_maps
from .masks import (
    generate_fixed_validation_masks,
    load_validation_masks,
    sample_sector_masks,
    save_validation_masks,
    sector_mask_to_angle_mask,
)
from .phantoms import PhantomRegion, make_disk_phantom, make_random_phantoms, speed_to_slowness_contrast

__all__ = [
    "ArrayCandidate",
    "CachedHeteroWaveDataset",
    "PhantomRegion",
    "SyntheticReconstructionDataset",
    "inspect_mat",
    "generate_fixed_validation_masks",
    "load_validation_masks",
    "load_speed_maps",
    "make_disk_phantom",
    "make_random_phantoms",
    "sample_sector_masks",
    "save_validation_masks",
    "sector_mask_to_angle_mask",
    "speed_to_slowness_contrast",
]
