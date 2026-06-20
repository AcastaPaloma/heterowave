from __future__ import annotations

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from heterowave.data.mat_loader import inspect_mat, load_speed_maps


def test_classic_mat_inspection_and_hwn_layout(tmp_path, capsys):
    path = tmp_path / "classic.mat"
    maps = np.arange(3 * 8 * 8, dtype=np.float32).reshape(8, 8, 3) + 1400
    savemat(path, {"speed_maps": maps, "scalar": np.array([[2]])})

    candidates = inspect_mat(path)
    loaded = load_speed_maps(path, key="speed_maps")

    assert {candidate.key for candidate in candidates} == {"speed_maps", "scalar"}
    assert loaded.shape == (3, 8, 8)
    np.testing.assert_array_equal(loaded[1], maps[:, :, 1])
    assert "speed_maps: shape=(8, 8, 3)" in capsys.readouterr().out


def test_hdf5_mat_loading_and_invalid_sample_removal(tmp_path):
    path = tmp_path / "v73.mat"
    maps = np.full((4, 8, 8), 1500.0, dtype=np.float32)
    maps[2, 1, 1] = np.nan
    with h5py.File(path, "w") as handle:
        handle.create_dataset("collection/speed", data=maps)

    loaded = load_speed_maps(path, key="collection/speed")

    assert loaded.shape == (3, 8, 8)
    assert np.isfinite(loaded).all()


def test_ambiguous_array_selection_requires_explicit_key(tmp_path):
    path = tmp_path / "ambiguous.mat"
    savemat(
        path,
        {
            "first": np.ones((2, 8, 8), dtype=np.float32),
            "second": np.ones((2, 8, 8), dtype=np.float32),
        },
    )
    with pytest.raises(ValueError, match="Pass an explicit array key"):
        load_speed_maps(path)


def test_ambiguous_sample_axis_requires_override(tmp_path):
    path = tmp_path / "cube.mat"
    savemat(path, {"speed": np.ones((8, 8, 8), dtype=np.float32)})
    with pytest.raises(ValueError, match="sample axis"):
        load_speed_maps(path, key="speed")
    assert load_speed_maps(path, key="speed", sample_axis=0).shape == (8, 8, 8)
