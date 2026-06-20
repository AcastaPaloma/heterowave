"""Robust loading and inspection of classic and v7.3 MATLAB arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class ArrayCandidate:
    key: str
    shape: tuple[int, ...]
    dtype: str


def _classic_arrays(value: object, prefix: str) -> Iterator[tuple[str, np.ndarray]]:
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        yield prefix, value
    elif isinstance(value, np.ndarray) and value.dtype == object:
        for index, item in np.ndenumerate(value):
            label = ",".join(str(part) for part in index)
            yield from _classic_arrays(item, f"{prefix}[{label}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _classic_arrays(item, f"{prefix}.{key}" if prefix else str(key))
    elif hasattr(value, "_fieldnames"):
        for field in value._fieldnames:
            yield from _classic_arrays(getattr(value, field), f"{prefix}.{field}")


def _hdf5_arrays(group: h5py.Group) -> Iterator[tuple[str, h5py.Dataset]]:
    for key, value in group.items():
        if isinstance(value, h5py.Dataset) and np.issubdtype(value.dtype, np.number):
            yield value.name.lstrip("/"), value
        elif isinstance(value, h5py.Group):
            yield from _hdf5_arrays(value)


def _classic_array_map(contents: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        nested_key: array
        for key, value in contents.items()
        if not key.startswith("__")
        for nested_key, array in _classic_arrays(value, key)
    }


def _is_hdf5(path: Path) -> bool:
    return h5py.is_hdf5(path)


def inspect_mat(path: str | Path) -> list[ArrayCandidate]:
    """Return and print every numeric array found in a MATLAB file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    candidates: list[ArrayCandidate] = []
    if _is_hdf5(path):
        with h5py.File(path, "r") as handle:
            candidates = [
                ArrayCandidate(key, tuple(int(v) for v in value.shape), str(value.dtype))
                for key, value in _hdf5_arrays(handle)
            ]
    else:
        contents = loadmat(path, squeeze_me=False, struct_as_record=False)
        candidates = [
            ArrayCandidate(key, tuple(int(v) for v in value.shape), str(value.dtype))
            for key, value in _classic_array_map(contents).items()
        ]
    print(f"MATLAB file: {path}")
    print(f"Format: {'v7.3/HDF5' if _is_hdf5(path) else 'classic'}")
    if not candidates:
        print("  (no numeric arrays found)")
    for candidate in candidates:
        print(f"  {candidate.key}: shape={candidate.shape}, dtype={candidate.dtype}")
    return candidates


def _read_array(path: Path, key: str) -> np.ndarray:
    if _is_hdf5(path):
        with h5py.File(path, "r") as handle:
            if key not in handle:
                raise KeyError(f"Array {key!r} not found in {path}")
            return np.asarray(handle[key])
    arrays = _classic_array_map(loadmat(path, squeeze_me=False, struct_as_record=False))
    if key not in arrays:
        raise KeyError(f"Array {key!r} not found in {path}")
    return arrays[key]


def _squeezed_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(size for size in shape if size != 1)


def _plausible_map_shape(shape: tuple[int, ...]) -> bool:
    shape = _squeezed_shape(shape)
    if len(shape) not in {2, 3}:
        return False
    spatial = shape if len(shape) == 2 else sorted(shape, reverse=True)[:2]
    return min(spatial) >= 8


def _select_key(candidates: list[ArrayCandidate], requested_key: str | None) -> str:
    if requested_key is not None:
        matches = [candidate for candidate in candidates if candidate.key == requested_key]
        if not matches:
            names = ", ".join(candidate.key for candidate in candidates)
            raise KeyError(f"Requested array {requested_key!r} was not found; candidates: {names}")
        if not _plausible_map_shape(matches[0].shape):
            raise ValueError(f"Requested array {requested_key!r} has implausible map shape {matches[0].shape}")
        return requested_key
    plausible = [candidate for candidate in candidates if _plausible_map_shape(candidate.shape)]
    if len(plausible) != 1:
        names = ", ".join(f"{item.key}{item.shape}" for item in plausible) or "none"
        raise ValueError(
            "Could not select a unique spatial-map array automatically. "
            f"Plausible candidates: {names}. Pass an explicit array key."
        )
    return plausible[0].key


def _to_nhw(array: np.ndarray, sample_axis: int | None) -> np.ndarray:
    array = np.squeeze(array)
    if array.ndim == 2:
        array = array[None]
    elif array.ndim == 3:
        if sample_axis is None:
            equal_pairs = [
                (left, right)
                for left in range(3)
                for right in range(left + 1, 3)
                if array.shape[left] == array.shape[right] and array.shape[left] >= 8
            ]
            if len(equal_pairs) == 1:
                sample_axis = ({0, 1, 2} - set(equal_pairs[0])).pop()
            else:
                raise ValueError(
                    f"Cannot infer a unique sample axis for shape {array.shape}; "
                    "pass --sample-axis explicitly"
                )
        sample_axis %= 3
        array = np.moveaxis(array, sample_axis, 0)
    else:
        raise ValueError(f"Expected a 2D or 3D map array after squeezing, got shape {array.shape}")
    if array.shape[1] < 8 or array.shape[2] < 8:
        raise ValueError(f"Spatial dimensions are implausibly small: {array.shape}")
    return np.asarray(array)


def load_speed_maps(
    path: str | Path,
    *,
    key: str | None = None,
    sample_axis: int | None = None,
) -> np.ndarray:
    """Load a MATLAB speed-map collection as finite positive ``[N,H,W]`` maps.

    Invalid samples are removed as whole samples rather than silently filling
    corrupt pixels.
    """
    path = Path(path)
    candidates = inspect_mat(path)
    selected = _select_key(candidates, key)
    maps = _to_nhw(_read_array(path, selected), sample_axis).astype(np.float32, copy=False)
    valid = np.isfinite(maps).all(axis=(1, 2)) & (maps > 0).all(axis=(1, 2))
    removed = int((~valid).sum())
    print(f"Selected array: {selected}; interpreted shape [N,H,W]={maps.shape}")
    if removed:
        print(f"Removed {removed} samples containing NaN, infinity, or non-positive speed")
    maps = maps[valid]
    if not len(maps):
        raise ValueError(f"No valid positive speed maps remain in {path}")
    return np.ascontiguousarray(maps)
