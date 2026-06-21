import torch

from heterowave.config import load_config
from heterowave.models import FBPUNet, HeteroWaveV2
from heterowave.training import build_model, create_datasets, training_prediction


def _load_matching_trunk(source, target):
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name] = value
    target.load_state_dict(target_state)


def test_zero_gated_v2_starts_exactly_as_masked_unet():
    torch.manual_seed(4)
    baseline = FBPUNet(channels=[4, 8], residual_output=True)
    model = HeteroWaveV2(
        image_size=16,
        num_angles=8,
        num_sectors=4,
        channels=[4, 8],
        residual_output=True,
        fusion_gate_init=0.0,
    )
    _load_matching_trunk(baseline, model)
    global_features = torch.randn(2, 3, 16, 16)
    partials = torch.randn(2, 4, 1, 16, 16)
    mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.bool)
    torch.testing.assert_close(
        model.forward_partials(global_features, partials, mask),
        baseline(global_features),
        rtol=0,
        atol=0,
    )


def test_v2_sector_fusion_is_permutation_invariant_and_gates_learn():
    torch.manual_seed(5)
    model = HeteroWaveV2(
        image_size=16,
        num_angles=8,
        num_sectors=4,
        channels=[4, 8],
        fusion_gate_init=0.2,
    )
    global_features = torch.randn(2, 3, 16, 16)
    partials = torch.randn(2, 4, 1, 16, 16)
    geometry = torch.randn(4, 3)
    mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.bool)
    permutation = torch.tensor([2, 0, 3, 1])
    expected = model.forward_partials(global_features, partials, mask, geometry=geometry)
    actual = model.forward_partials(
        global_features,
        partials[:, permutation],
        mask[:, permutation],
        geometry=geometry[permutation],
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    expected.square().mean().backward()
    assert all(gate.grad is not None and torch.isfinite(gate.grad) for gate in model.fusion_gates)


def test_v2_config_freezes_only_global_trunk_and_trains_end_to_end():
    config = load_config("configs/local_heterowave_v2_smoke.yaml", ["device=cpu", "model.freeze_global_trunk=true"])
    model = build_model(config)
    assert not any(parameter.requires_grad for parameter in model.encoders.parameters())
    assert not any(parameter.requires_grad for parameter in model.decoders.parameters())
    assert not any(parameter.requires_grad for parameter in model.output.parameters())
    assert any(parameter.requires_grad for parameter in model.sector_encoders.parameters())
    train, _ = create_datasets(config)
    sinogram = torch.stack([train[index]["sinogram"] for index in range(2)])
    prediction = training_prediction(
        model,
        sinogram,
        train.metadata,
        config,
        torch.Generator().manual_seed(1337),
    )
    assert prediction.shape == (2, 1, 32, 32)
    assert torch.isfinite(prediction).all()
