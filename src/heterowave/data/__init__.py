"""Synthetic and cached dataset utilities."""

from .dataset import CachedHeteroWaveDataset, SyntheticReconstructionDataset
from .mat_loader import ArrayCandidate, inspect_mat, load_speed_maps
from .phantoms import PhantomRegion, make_disk_phantom, make_random_phantoms, speed_to_slowness_contrast

__all__ = [
    "ArrayCandidate",
    "CachedHeteroWaveDataset",
    "PhantomRegion",
    "SyntheticReconstructionDataset",
    "inspect_mat",
    "load_speed_maps",
    "make_disk_phantom",
    "make_random_phantoms",
    "speed_to_slowness_contrast",
]
