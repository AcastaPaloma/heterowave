"""Restart-safe construction of memory-mapped Phase 3 caches."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.nn import functional as F

from heterowave.physics import parallel_beam_project

from .mat_loader import load_speed_maps

PROJECTOR_VERSION = "parallel_beam_grid_sample_v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _resize(maps: np.ndarray, size: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(maps)).to(torch.float32).unsqueeze(1)
    return F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False).squeeze(1)


def _training_statistics(maps: np.ndarray, image_size: int, batch_size: int) -> tuple[float, float]:
    total = 0.0
    total_square = 0.0
    count = 0
    for start in range(0, len(maps), batch_size):
        resized = _resize(maps[start : start + batch_size], image_size).to(torch.float64)
        total += float(resized.sum())
        total_square += float(resized.square().sum())
        count += resized.numel()
    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    std = variance**0.5
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Training speed maps have zero or invalid standard deviation")
    return mean, std


def deterministic_split(count: int, *, seed: int, validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("At least two official training samples are required for a train/validation split")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    val_count = max(1, min(count - 1, int(round(count * validation_fraction))))
    order = np.random.default_rng(seed).permutation(count)
    return order[val_count:], order[:val_count]


def _source_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _signature(settings: dict[str, Any]) -> str:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _open_output(path: Path, *, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.memmap:
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"Partial cache has unexpected shape or dtype: {path}")
        return array
    return open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _validate_output(path: Path, *, shape: tuple[int, ...], dtype: np.dtype[Any]) -> None:
    try:
        array = np.load(path, mmap_mode="r")
    except Exception as error:
        raise ValueError(f"Existing cache file cannot be read: {path}") from error
    if array.shape != shape or array.dtype != dtype:
        raise ValueError(
            f"Existing cache file has shape {array.shape} and dtype {array.dtype}, "
            f"expected {shape} and {dtype}: {path}"
        )


def _cache_split(
    name: str,
    maps: np.ndarray,
    *,
    output_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    image_size: int,
    num_angles: int,
    detector_bins: int,
    batch_size: int,
    device: torch.device,
    speed_mean: float,
    speed_std: float,
    water_speed: float,
    align_corners: bool,
) -> None:
    target_final = output_dir / f"{name}_targets.npy"
    sino_final = output_dir / f"{name}_sinograms.npy"
    if target_final.exists() and sino_final.exists():
        _validate_output(
            target_final,
            shape=(len(maps), image_size, image_size),
            dtype=np.dtype(np.float32),
        )
        _validate_output(
            sino_final,
            shape=(len(maps), num_angles, detector_bins),
            dtype=np.dtype(np.float16),
        )
        print(f"{name}: complete cache already exists; skipping")
        return

    target_partial = output_dir / f"{name}_targets.npy.partial"
    sino_partial = output_dir / f"{name}_sinograms.npy.partial"
    offset = int(state["offsets"].get(name, 0))
    if offset == len(maps) and target_partial.exists() and sino_partial.exists():
        os.replace(target_partial, target_final)
        os.replace(sino_partial, sino_final)
        print(f"{name}: finalized completed partial cache")
        return

    targets = _open_output(
        target_final if target_final.exists() else target_partial,
        shape=(len(maps), image_size, image_size),
        dtype=np.dtype(np.float32),
    )
    sinograms = _open_output(
        sino_final if sino_final.exists() else sino_partial,
        shape=(len(maps), num_angles, detector_bins),
        dtype=np.dtype(np.float16),
    )
    for start in range(offset, len(maps), batch_size):
        stop = min(start + batch_size, len(maps))
        speed = _resize(maps[start:stop], image_size)
        normalized = (speed - speed_mean) / speed_std
        slowness = speed.reciprocal() - (1.0 / water_speed)
        with torch.inference_mode():
            projected = parallel_beam_project(
                slowness.unsqueeze(1).to(device),
                num_angles=num_angles,
                detector_bins=detector_bins,
                align_corners=align_corners,
            ).cpu()
        if not torch.isfinite(normalized).all() or not torch.isfinite(projected).all():
            raise ValueError(f"Non-finite value generated in {name} samples {start}:{stop}")
        targets[start:stop] = normalized.numpy().astype(np.float32, copy=False)
        sinograms[start:stop] = projected.numpy().astype(np.float16, copy=False)
        targets.flush()
        sinograms.flush()
        state["offsets"][name] = stop
        _atomic_json(state_path, state)
        print(f"{name}: {stop}/{len(maps)}")
    del targets, sinograms
    if not target_final.exists():
        os.replace(target_partial, target_final)
    if not sino_final.exists():
        os.replace(sino_partial, sino_final)


def prepare_cache(
    *,
    train_mat: str | Path,
    test_mat: str | Path,
    output_dir: str | Path,
    train_key: str | None = None,
    test_key: str | None = None,
    train_sample_axis: int | None = None,
    test_sample_axis: int | None = None,
    image_size: int = 128,
    num_angles: int = 64,
    detector_bins: int | None = None,
    num_sectors: int = 16,
    batch_size: int = 32,
    device: str = "cuda",
    seed: int = 1337,
    validation_fraction: float = 0.1,
    water_speed: float = 1500.0,
    align_corners: bool = False,
) -> Path:
    """Build all cache arrays and return the final metadata path."""
    train_path, test_path = Path(train_mat), Path(test_mat)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detector_bins = detector_bins or image_size
    if min(image_size, num_angles, detector_bins, num_sectors, batch_size) < 1:
        raise ValueError("Image, acquisition, sector, and batch sizes must be positive")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preprocessing was requested, but torch.cuda.is_available() is false")

    settings: dict[str, Any] = {
        "train_source": _source_identity(train_path),
        "test_source": _source_identity(test_path),
        "train_key": train_key,
        "test_key": test_key,
        "train_sample_axis": train_sample_axis,
        "test_sample_axis": test_sample_axis,
        "image_size": image_size,
        "num_angles": num_angles,
        "detector_bins": detector_bins,
        "num_sectors": num_sectors,
        "preprocessing_device": str(torch_device),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "water_speed": water_speed,
        "align_corners": align_corners,
        "projector_version": PROJECTOR_VERSION,
    }
    signature = _signature(settings)
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)
        if existing_metadata.get("cache_signature") != signature:
            raise RuntimeError(
                f"Existing completed cache in {output} was made with different settings; "
                "use a new output directory"
            )
    state_path = output / ".cache-progress.json"
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("signature") != signature:
            raise RuntimeError(
                f"Existing partial cache in {output} was made with different settings; "
                "use a new output directory or remove the partial cache"
            )
    else:
        state = {"signature": signature, "offsets": {}}
        _atomic_json(state_path, state)

    official_train = load_speed_maps(train_path, key=train_key, sample_axis=train_sample_axis)
    official_test = load_speed_maps(test_path, key=test_key, sample_axis=test_sample_axis)
    train_indices, val_indices = deterministic_split(
        len(official_train), seed=seed, validation_fraction=validation_fraction
    )
    split_maps = {
        "train": official_train[train_indices],
        "val": official_train[val_indices],
        "test": official_test,
    }
    speed_mean, speed_std = _training_statistics(split_maps["train"], image_size, batch_size)
    print(f"Training speed mean={speed_mean:.8g}, std={speed_std:.8g}")
    for name, maps in split_maps.items():
        _cache_split(
            name,
            maps,
            output_dir=output,
            state=state,
            state_path=state_path,
            image_size=image_size,
            num_angles=num_angles,
            detector_bins=detector_bins,
            batch_size=batch_size,
            device=torch_device,
            speed_mean=speed_mean,
            speed_std=speed_std,
            water_speed=water_speed,
            align_corners=align_corners,
        )

    metadata = {
        **settings,
        "cache_signature": signature,
        "format_version": 1,
        "target_representation": "normalized_speed",
        "sinogram_source": "slowness_contrast",
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "split_seed": seed,
        "split_counts": {name: len(maps) for name, maps in split_maps.items()},
        "train_indices": train_indices.tolist(),
        "validation_indices": val_indices.tolist(),
        "target_dtype": "float32",
        "sinogram_dtype": "float16",
        "normalization": {
            "representation": "normalized_speed",
            "speed_mean": speed_mean,
            "speed_std": speed_std,
            "water_speed": water_speed,
        },
        "projector": {
            "version": PROJECTOR_VERSION,
            "implementation": "pure_pytorch",
            "num_angles": num_angles,
            "detector_bins": detector_bins,
            "align_corners": align_corners,
            "input_representation": "slowness_contrast",
        },
    }
    _atomic_json(metadata_path, metadata)
    state_path.unlink(missing_ok=True)
    print(f"Cache complete: {metadata_path}")
    return metadata_path
