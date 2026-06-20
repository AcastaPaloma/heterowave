import pytest
import torch

from heterowave.data.phantoms import make_disk_phantom
from heterowave.physics import filtered_backprojection, parallel_beam_project, unfiltered_backprojection

CUDA_CAN_RUN = torch.cuda.is_available() and (
    f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}" in torch.cuda.get_arch_list()
)


def test_backprojection_shape_and_gradients():
    sinogram = torch.randn(2, 11, 20, requires_grad=True)
    reconstruction = unfiltered_backprojection(sinogram, output_size=24)
    assert reconstruction.shape == (2, 1, 24, 24)
    reconstruction.square().mean().backward()
    assert sinogram.grad is not None
    assert torch.isfinite(sinogram.grad).all()
    assert sinogram.grad.abs().sum() > 0


def test_fbp_roughly_reconstructs_disk():
    target = make_disk_phantom(48, background_speed=0.0, disk_speed=1.0).unsqueeze(0)
    sinogram = parallel_beam_project(target, num_angles=64)
    reconstruction = filtered_backprojection(sinogram)
    center = reconstruction[0, 0, 24, 24]
    corner = reconstruction[0, 0, 2, 2]
    assert center > corner + 0.35
    assert torch.mean((reconstruction - target).square()).sqrt() < 0.32


def test_gradients_pass_through_projector_and_fbp():
    image = torch.randn(1, 1, 16, 16, requires_grad=True)
    reconstruction = filtered_backprojection(parallel_beam_project(image, num_angles=12))
    reconstruction.abs().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


@pytest.mark.skipif(not CUDA_CAN_RUN, reason="CUDA is unavailable or this PyTorch build lacks the GPU architecture")
def test_backprojector_cpu_gpu_close():
    sinogram = torch.randn(1, 7, 12)
    cpu = filtered_backprojection(sinogram)
    gpu = filtered_backprojection(sinogram.cuda()).cpu()
    torch.testing.assert_close(cpu, gpu, atol=2e-4, rtol=2e-4)
