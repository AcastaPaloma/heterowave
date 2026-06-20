from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.io import savemat

from heterowave.data.cache import deterministic_split, prepare_cache
from heterowave.data.dataset import CachedHeteroWaveDataset
from heterowave.data.visualize import main as visualize_main


def _write_fixture(path, count, offset=0.0):
    yy, xx = np.mgrid[:10, :12]
    maps = []
    for index in range(count):
        maps.append(1480.0 + offset + index * 4.0 + xx + yy * 0.5)
    savemat(path, {"speed": np.stack(maps).astype(np.float32)})


def test_deterministic_split_is_disjoint_and_reproducible():
    first_train, first_val = deterministic_split(20, seed=1337, validation_fraction=0.2)
    second_train, second_val = deterministic_split(20, seed=1337, validation_fraction=0.2)
    np.testing.assert_array_equal(first_train, second_train)
    np.testing.assert_array_equal(first_val, second_val)
    assert len(first_val) == 4
    assert set(first_train).isdisjoint(first_val)


def test_prepare_cache_and_memory_mapped_dataset(tmp_path):
    train_mat, test_mat = tmp_path / "train.mat", tmp_path / "test.mat"
    cache_dir = tmp_path / "cache"
    _write_fixture(train_mat, 6)
    _write_fixture(test_mat, 2, offset=12.0)

    metadata_path = prepare_cache(
        train_mat=train_mat,
        test_mat=test_mat,
        output_dir=cache_dir,
        train_key="speed",
        test_key="speed",
        train_sample_axis=0,
        test_sample_axis=0,
        image_size=8,
        num_angles=5,
        batch_size=2,
        validation_fraction=1 / 3,
        device="cpu",
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["split_seed"] == 1337
    assert metadata["target_representation"] == "normalized_speed"
    assert metadata["sinogram_source"] == "slowness_contrast"
    assert metadata["split_counts"] == {"test": 2, "train": 4, "val": 2}
    assert metadata["speed_std"] > 0
    assert not (cache_dir / ".cache-progress.json").exists()

    targets = np.load(cache_dir / "train_targets.npy", mmap_mode="r")
    sinograms = np.load(cache_dir / "train_sinograms.npy", mmap_mode="r")
    assert isinstance(targets, np.memmap)
    assert targets.shape == (4, 8, 8)
    assert targets.dtype == np.float32
    assert sinograms.shape == (4, 5, 8)
    assert sinograms.dtype == np.float16
    assert np.isfinite(targets).all() and np.isfinite(sinograms).all()

    dataset = CachedHeteroWaveDataset(cache_dir, "train")
    sample = dataset[0]
    assert sample["target"].shape == (1, 8, 8)
    assert sample["sinogram"].shape == (5, 8)

    figure = tmp_path / "sample.png"
    assert visualize_main(["--cache-dir", str(cache_dir), "--output", str(figure)]) == figure
    assert figure.stat().st_size > 0

    # A completed cache can be checked and reused without rewriting its arrays.
    before = (cache_dir / "train_targets.npy").stat().st_mtime_ns
    prepare_cache(
        train_mat=train_mat,
        test_mat=test_mat,
        output_dir=cache_dir,
        train_key="speed",
        test_key="speed",
        train_sample_axis=0,
        test_sample_axis=0,
        image_size=8,
        num_angles=5,
        batch_size=2,
        validation_fraction=1 / 3,
        device="cpu",
    )
    assert (cache_dir / "train_targets.npy").stat().st_mtime_ns == before


def test_cache_resumes_after_interrupted_batch(tmp_path, monkeypatch):
    import heterowave.data.cache as cache_module

    train_mat, test_mat = tmp_path / "train.mat", tmp_path / "test.mat"
    cache_dir = tmp_path / "cache"
    _write_fixture(train_mat, 5)
    _write_fixture(test_mat, 2)
    real_projector = cache_module.parallel_beam_project
    calls = 0

    def interrupt_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return real_projector(*args, **kwargs)

    arguments = dict(
        train_mat=train_mat,
        test_mat=test_mat,
        output_dir=cache_dir,
        train_key="speed",
        test_key="speed",
        train_sample_axis=0,
        test_sample_axis=0,
        image_size=8,
        num_angles=4,
        batch_size=2,
        validation_fraction=0.2,
        device="cpu",
    )
    monkeypatch.setattr(cache_module, "parallel_beam_project", interrupt_once)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        prepare_cache(**arguments)
    progress = json.loads((cache_dir / ".cache-progress.json").read_text(encoding="utf-8"))
    assert progress["offsets"]["train"] == 2

    monkeypatch.setattr(cache_module, "parallel_beam_project", real_projector)
    prepare_cache(**arguments)
    assert np.load(cache_dir / "train_targets.npy").shape[0] == 4
    assert not list(cache_dir.glob("*.partial"))
