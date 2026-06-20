import torch

from heterowave.data.phantoms import make_disk_phantom, make_random_phantoms, speed_to_slowness_contrast


def test_synthetic_phantom_shapes_and_reproducibility():
    first = make_random_phantoms(3, 24, seed=7)
    second = make_random_phantoms(3, 24, seed=7)
    assert first.shape == (3, 1, 24, 24)
    torch.testing.assert_close(first, second)
    assert torch.unique(first).numel() > 2


def test_disk_and_slowness_contrast():
    disk = make_disk_phantom(32)
    contrast = speed_to_slowness_contrast(disk)
    assert disk.shape == (1, 32, 32)
    assert contrast.min() < 0
    assert contrast.max() == 0

