"""Verify the tiny Phase 1-2 runtime without downloads or preprocessing."""

from __future__ import annotations

import argparse
import platform
import sys

import torch

from heterowave.config import load_config
from heterowave.data.phantoms import make_random_phantoms
from heterowave.physics import filtered_backprojection, parallel_beam_project


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_smoke.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Python 3.11 is required; found {platform.python_version()}")
    if config.precision != "fp32" or config.compile or config.data.num_workers != 0:
        raise RuntimeError("Local config must use FP32, compile=false, and num_workers=0")
    if config.data.preprocess or config.data.dataset != "synthetic":
        raise RuntimeError("Local verification must use in-memory synthetic data without preprocessing")
    if not torch.cuda.is_available():
        raise RuntimeError("Local config requests CUDA, but torch.cuda.is_available() is false")
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    if architecture not in torch.cuda.get_arch_list():
        raise RuntimeError(
            f"The installed PyTorch wheel does not include {architecture}; "
            "install the pinned CUDA 12.6 wheel from the README"
        )
    device = torch.device("cuda")
    images = make_random_phantoms(config.data.batch_size, config.data.image_size, seed=config.seed, device=device)
    contrast = ((images - 1500.0) / 100.0).requires_grad_()
    sinogram = parallel_beam_project(contrast, num_angles=config.physics.num_angles, detector_bins=config.physics.detector_bins)
    reconstruction = filtered_backprojection(sinogram, output_size=config.data.image_size)
    reconstruction.square().mean().backward()
    if contrast.grad is None or not torch.isfinite(contrast.grad).all():
        raise RuntimeError("Differentiable physics smoke test produced invalid gradients")
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(0)} capability={capability}")
    print(f"config=fp32 compile={config.compile} num_workers={config.data.num_workers}")
    print(f"smoke_shapes=images{tuple(images.shape)} sinogram{tuple(sinogram.shape)} reconstruction{tuple(reconstruction.shape)}")
    print("environment_verification=PASS")


if __name__ == "__main__":
    main()
