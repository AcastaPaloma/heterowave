import torch

from heterowave.data.masks import (
    generate_fixed_validation_masks,
    sample_sector_masks,
    sector_mask_to_angle_mask,
)


def test_mixed_masks_are_deterministic_valid_and_arbitrary():
    first = sample_sector_masks(64, generator=torch.Generator().manual_seed(22))
    second = sample_sector_masks(64, generator=torch.Generator().manual_seed(22))
    torch.testing.assert_close(first, second)
    counts = first.sum(dim=1)
    assert first.shape == (64, 16)
    assert int(counts.min()) >= 2 and int(counts.max()) <= 16
    assert len(torch.unique(counts)) > 3


def test_sector_mask_expands_to_angles():
    sectors = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
    angles = sector_mask_to_angle_mask(sectors, 12)
    assert angles.tolist() == [[True, True, True, False, False, False, True, True, True, False, False, False]]


def test_fixed_validation_scenarios_have_expected_counts():
    masks = generate_fixed_validation_masks(seed=1337)
    assert set(masks) == {
        "all_16",
        "observed_12",
        "random_8",
        "contiguous_8",
        "random_4",
        "contiguous_4",
        "random_2",
        "contiguous_2",
    }
    for name, mask in masks.items():
        expected = int(name.split("_")[-1])
        assert int(mask.sum()) == expected
