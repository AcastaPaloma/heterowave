import torch

from heterowave.config import load_config
from heterowave.data import SyntheticReconstructionDataset
from heterowave.models import LearnedPrimalDual


def test_learned_primal_dual_preserves_target_shape_on_synthetic_fixture():
    dataset = SyntheticReconstructionDataset(count=2, image_size=32, num_angles=16, seed=1337)
    batch = [dataset[index] for index in range(2)]
    sinogram = torch.stack([sample["sinogram"] for sample in batch])
    target = torch.stack([sample["target"] for sample in batch])
    sector_mask = torch.tensor(
        [
            [True, True, False, True],
            [True, False, True, True],
        ]
    )
    model = LearnedPrimalDual(
        image_size=32,
        num_angles=16,
        num_sectors=4,
        iterations=1,
        hidden_channels=4,
    )
    output = model(sinogram, sector_mask, dataset.metadata)
    assert output.shape == target.shape
    assert torch.isfinite(output).all()


def test_learned_primal_dual_handles_bfloat16_autocast_physics_path():
    dataset = SyntheticReconstructionDataset(count=1, image_size=32, num_angles=16, seed=1337)
    sample = dataset[0]
    sinogram = sample["sinogram"].unsqueeze(0)
    sector_mask = torch.tensor([[True, True, False, True]])
    model = LearnedPrimalDual(
        image_size=32,
        num_angles=16,
        num_sectors=4,
        iterations=1,
        hidden_channels=4,
    )
    with torch.autocast(device_type=sinogram.device.type, dtype=torch.bfloat16):
        output = model(sinogram, sector_mask, dataset.metadata)
    assert output.shape == sample["target"].unsqueeze(0).shape
    assert torch.isfinite(output).all()


def test_learned_primal_dual_smoke_config_loads():
    primal_dual = load_config("configs/local_learned_primal_dual_smoke.yaml")
    assert primal_dual.model.name == "learned_primal_dual"
    assert primal_dual.model.primal_dual_iterations == 1
