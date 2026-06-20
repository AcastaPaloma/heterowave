"""Visualize a cached target, complete sinogram, and reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from heterowave.physics import filtered_backprojection

from .dataset import CachedHeteroWaveDataset


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args(argv)

    dataset = CachedHeteroWaveDataset(args.cache_dir, args.split)
    if not -len(dataset) <= args.index < len(dataset):
        raise IndexError(f"index {args.index} is outside a dataset of length {len(dataset)}")
    sample = dataset[args.index]
    target = sample["target"].squeeze(0)
    sinogram = sample["sinogram"]
    reconstruction = filtered_backprojection(
        sinogram.unsqueeze(0), output_size=int(dataset.metadata["image_size"])
    )[0, 0]

    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(target.numpy(), cmap="viridis")
    axes[0].set_title("Normalized speed target")
    axes[1].imshow(sinogram.numpy(), cmap="magma", aspect="auto")
    axes[1].set_title("Complete sinogram")
    axes[2].imshow(reconstruction.numpy(), cmap="gray")
    axes[2].set_title("FBP of slowness contrast")
    for axis in axes:
        axis.set_axis_off()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(output)
    return output


if __name__ == "__main__":
    main()
