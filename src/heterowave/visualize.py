"""Generate a synthetic phantom, sinogram, UBP, and FBP sanity-check figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from .config import load_config
from .data.phantoms import PhantomRegion, render_phantom
from .physics import filtered_backprojection, parallel_beam_project, unfiltered_backprojection


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_smoke.yaml")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    requested = config.device
    cuda_usable = False
    if requested == "cuda" and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        cuda_usable = f"sm_{capability[0]}{capability[1]}" in torch.cuda.get_arch_list()
    device = torch.device("cuda" if cuda_usable else "cpu")
    if requested == "cuda" and not cuda_usable:
        print("warning=CUDA requested but the installed PyTorch wheel cannot execute on this GPU; using CPU")
    size = config.data.image_size
    speed = render_phantom(size, [
        PhantomRegion("circle", (0.0, 0.0), (0.48, 0.48), 1540.0),
        PhantomRegion("ellipse", (-0.16, 0.10), (0.13, 0.22), 1470.0, 25.0),
        PhantomRegion("rectangle", (0.22, -0.16), (0.11, 0.08), 1580.0, -15.0),
    ], device=device).unsqueeze(0)
    image = (speed - 1500.0) / 100.0
    sino = parallel_beam_project(image, num_angles=config.physics.num_angles, detector_bins=config.physics.detector_bins, align_corners=config.physics.align_corners)
    ubp = unfiltered_backprojection(sino, output_size=size, align_corners=config.physics.align_corners)
    fbp = filtered_backprojection(sino, output_size=size, align_corners=config.physics.align_corners)

    arrays = [image[0, 0], sino[0], ubp[0, 0], fbp[0, 0]]
    titles = ["Speed contrast", "Sinogram", "Unfiltered BP", "Filtered BP"]
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.2), constrained_layout=True)
    for axis, array, title in zip(axes, arrays, titles):
        axis.imshow(array.detach().cpu(), cmap="gray", aspect="auto")
        axis.set_title(title)
        axis.axis("off")
    output = Path(config.visualization.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"device={device}")
    print(f"sinogram_shape={tuple(sino.shape)}")
    print(f"saved={output.resolve()}")
    return output


if __name__ == "__main__":
    main()
