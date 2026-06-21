import torch

from heterowave.config import load_config
from heterowave.models import FBPConvNet


def test_fbpconvnet_preserves_image_shape_and_starts_as_fbp():
    model = FBPConvNet(channels=[4, 8], residual_output=True)
    fbp = torch.randn(2, 1, 32, 32)
    output = model(fbp)
    assert output.shape == fbp.shape
    assert torch.allclose(output, fbp)
    assert torch.isfinite(output).all()


def test_fbpconvnet_smoke_config_loads():
    fbpconvnet = load_config("configs/local_fbpconvnet_smoke.yaml")
    assert fbpconvnet.model.name == "fbpconvnet"
