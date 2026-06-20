import torch

from heterowave.baselines import fbp_normalized_speed, fbp_unet_features
from heterowave.data import SyntheticReconstructionDataset
from heterowave.losses import reconstruction_loss
from heterowave.models import FBPUNet
from heterowave.training import resolve_device


def _batch(count=4, size=16, angles=8):
    dataset = SyntheticReconstructionDataset(count, size, angles, seed=1337)
    target = torch.stack([dataset[index]["target"] for index in range(count)])
    sinogram = torch.stack([dataset[index]["sinogram"] for index in range(count)])
    return target, sinogram, dataset.metadata


def test_fbp_baseline_and_fixed_features_are_finite():
    target, sinogram, metadata = _batch()
    fbp = fbp_normalized_speed(sinogram, metadata)
    features = fbp_unet_features(sinogram, metadata)
    assert fbp.shape == target.shape
    assert features.shape == (4, 3, 16, 16)
    assert torch.isfinite(fbp).all() and torch.isfinite(features).all()
    torch.testing.assert_close(features[:, 2], torch.ones_like(features[:, 2]))


def test_baseline_training_step_updates_parameters():
    torch.manual_seed(1)
    target, sinogram, metadata = _batch()
    features = fbp_unet_features(sinogram, metadata)
    model = FBPUNet(channels=[4, 8])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.output.weight.detach().clone()
    prediction = model(features)
    loss, parts = reconstruction_loss(prediction, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    assert not torch.equal(before, model.output.weight)


def test_tiny_unet_overfits_eight_examples():
    torch.manual_seed(7)
    target, sinogram, metadata = _batch(count=8)
    features = fbp_unet_features(sinogram, metadata)
    model = FBPUNet(channels=[4, 8], residual_output=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    with torch.no_grad():
        initial, _ = reconstruction_loss(model(features), target)
    for _ in range(80):
        prediction = model(features)
        loss, _ = reconstruction_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final, _ = reconstruction_loss(model(features), target)
    assert final < initial * 0.35


def test_resolve_device_uses_available_cuda_without_arch_list_filter(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    real_ones = torch.ones

    def fake_ones(*args, **kwargs):
        kwargs.pop("device", None)
        return real_ones(*args, **kwargs)

    monkeypatch.setattr(torch, "ones", fake_ones)
    assert resolve_device("cuda").type == "cuda"
