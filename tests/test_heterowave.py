import pytest
import torch

from heterowave.data import SyntheticReconstructionDataset
from heterowave.losses import reconstruction_loss
from heterowave.models import HeteroWave, sector_backprojections, sector_geometry


def test_sector_backprojections_respect_mask():
    dataset = SyntheticReconstructionDataset(2, 12, 8, seed=4)
    sinogram = torch.stack([dataset[index]["sinogram"] for index in range(2)])
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=torch.bool)
    partials = sector_backprojections(sinogram, mask, num_sectors=4, output_size=12)
    assert partials.shape == (2, 4, 1, 12, 12)
    assert torch.count_nonzero(partials[0, 1]) == 0
    assert torch.count_nonzero(partials[1, 3]) == 0
    assert torch.isfinite(partials).all()


def test_sector_backprojections_stay_fp32_under_autocast():
    sinogram = torch.randn(1, 8, 8)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        partials = sector_backprojections(sinogram, mask, num_sectors=4, output_size=8)
    assert partials.dtype == torch.float32
    assert sector_backprojections(sinogram, mask, num_sectors=4, output_size=8).dtype == torch.float32


@pytest.mark.parametrize("aggregation", ["mean", "mean_var", "mean_var_count"])
def test_heterowave_aggregation_modes_and_output_shape(aggregation):
    model = HeteroWave(
        image_size=8,
        num_angles=8,
        num_sectors=4,
        channels=[4, 8],
        aggregation=aggregation,
    )
    output = model(torch.randn(2, 8, 8), torch.tensor([[1, 1, 0, 0], [1, 0, 1, 1]]))
    assert output.shape == (2, 1, 8, 8)


def test_uncertainty_mode_is_bounded():
    model = HeteroWave(
        image_size=8, num_angles=8, num_sectors=4, channels=[4, 8], uncertainty=True
    )
    output = model(torch.randn(1, 8, 8), torch.ones(1, 4))
    assert output["mean"].shape == output["log_variance"].shape == (1, 1, 8, 8)
    assert float(output["log_variance"].min()) >= -8
    assert float(output["log_variance"].max()) <= 3


def test_model_is_invariant_to_sector_order_when_geometry_moves_with_data():
    torch.manual_seed(5)
    model = HeteroWave(image_size=8, num_angles=8, num_sectors=4, channels=[4, 8])
    partials = torch.randn(2, 4, 1, 8, 8)
    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 1, 0]], dtype=torch.bool)
    geometry = sector_geometry(8, 4, device=partials.device, dtype=partials.dtype)
    permutation = torch.tensor([2, 0, 3, 1])
    expected = model.forward_partials(partials, mask, geometry=geometry)
    actual = model.forward_partials(
        partials[:, permutation], mask[:, permutation], geometry=geometry[permutation]
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_tiny_heterowave_overfits_eight_examples():
    torch.manual_seed(9)
    dataset = SyntheticReconstructionDataset(8, 16, 8, seed=9)
    target = torch.stack([dataset[index]["target"] for index in range(8)])
    sinogram = torch.stack([dataset[index]["sinogram"] for index in range(8)])
    mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )
    partials = sector_backprojections(sinogram, mask, num_sectors=4, output_size=16)
    model = HeteroWave(
        image_size=16, num_angles=8, num_sectors=4, channels=[8, 12], aggregation="mean_var_count"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    with torch.no_grad():
        initial, _ = reconstruction_loss(model.forward_partials(partials, mask), target)
    for _ in range(120):
        prediction = model.forward_partials(partials, mask)
        loss, _ = reconstruction_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final, _ = reconstruction_loss(model.forward_partials(partials, mask), target)
    assert final < initial * 0.30
