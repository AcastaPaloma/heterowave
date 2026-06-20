import pytest
import torch

from heterowave.models import FBPUNet


@pytest.mark.parametrize("size", [16, 31, 32])
def test_fbp_unet_preserves_image_shape(size):
    model = FBPUNet(channels=[4, 8, 12])
    inputs = torch.randn(2, 3, size, size)
    output = model(inputs)
    assert output.shape == (2, 1, size, size)
    assert torch.isfinite(output).all()


def test_fbp_unet_rejects_wrong_input_channels():
    model = FBPUNet(channels=[4, 8])
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(1, 2, 16, 16))
