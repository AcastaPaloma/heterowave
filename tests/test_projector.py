import pytest
import torch

from heterowave.data.phantoms import make_disk_phantom
from heterowave.physics import parallel_beam_project

CUDA_CAN_RUN = torch.cuda.is_available() and (
    f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}" in torch.cuda.get_arch_list()
)


def test_constant_image_has_smooth_expected_projections():
    image = torch.ones(2, 1, 24, 24)
    sinogram = parallel_beam_project(image, num_angles=16)
    assert sinogram.shape == (2, 16, 24)
    assert torch.isfinite(sinogram).all()
    assert sinogram[:, 0].std() < 1e-6
    assert sinogram.max() <= 24.0 + 1e-5


def test_centered_disk_sinogram_is_symmetric():
    disk = (make_disk_phantom(40, background_speed=0.0, disk_speed=1.0)).unsqueeze(0)
    sinogram = parallel_beam_project(disk, num_angles=32)
    torch.testing.assert_close(sinogram, sinogram.flip(-1), atol=0.16, rtol=0.03)
    # Rasterization makes a small angle-dependent edge variation unavoidable.
    assert sinogram.std(dim=1).mean() < 0.18


def test_projector_propagates_gradients():
    image = torch.randn(1, 1, 16, 16, requires_grad=True)
    parallel_beam_project(image, num_angles=8).square().mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0


@pytest.mark.skipif(not CUDA_CAN_RUN, reason="CUDA is unavailable or this PyTorch build lacks the GPU architecture")
def test_projector_cpu_gpu_close():
    image = torch.randn(1, 1, 12, 12)
    cpu = parallel_beam_project(image, num_angles=7)
    gpu = parallel_beam_project(image.cuda(), num_angles=7).cpu()
    torch.testing.assert_close(cpu, gpu, atol=2e-4, rtol=2e-4)
