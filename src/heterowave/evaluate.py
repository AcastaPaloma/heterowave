"""Evaluate Phase 4 FBP and FBP + U-Net baselines."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .baselines import fbp_normalized_speed, fbp_unet_features
from .config import load_config
from .data import CachedHeteroWaveDataset, SyntheticReconstructionDataset
from .metrics import ReconstructionMetricAccumulator
from .models import FBPUNet
from .physics import parallel_beam_project
from .training import resolve_device


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_smoke.yaml")
    parser.add_argument("--baseline", choices=("fbp", "unet"), default="fbp")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    device = resolve_device(config.device)
    if config.data.dataset == "cached":
        dataset = CachedHeteroWaveDataset(config.data.root, args.split)
    else:
        count = config.data.train_samples if args.split == "train" else config.data.val_samples
        dataset = SyntheticReconstructionDataset(
            count, config.data.image_size, config.physics.num_angles, seed=config.seed + 1
        )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
    )
    model = None
    if args.baseline == "unet":
        if not args.checkpoint:
            parser.error("--checkpoint is required for the unet baseline")
        model = FBPUNet(
            channels=config.model.channels, residual_output=config.model.residual_output
        ).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()

    metrics = ReconstructionMetricAccumulator()
    residual_sum, residual_count, inference_seconds = 0.0, 0, 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch in loader:
            target = batch["target"].to(device, non_blocking=True)
            sinogram = batch["sinogram"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            if model is None:
                prediction = fbp_normalized_speed(sinogram, dataset.metadata)
            else:
                prediction = model(fbp_unet_features(sinogram, dataset.metadata))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            metrics.update(prediction, target)
            speed = prediction * float(dataset.metadata["speed_std"]) + float(dataset.metadata["speed_mean"])
            speed = speed.clamp_min(1.0)
            slowness_contrast = speed.reciprocal() - (1.0 / float(dataset.metadata["water_speed"]))
            reprojection = parallel_beam_project(
                slowness_contrast,
                num_angles=int(dataset.metadata["num_angles"]),
                detector_bins=int(dataset.metadata["detector_bins"]),
                align_corners=bool(dataset.metadata.get("align_corners", False)),
            )
            residual_sum += float((reprojection - sinogram).abs().sum())
            residual_count += sinogram.numel()
    values = metrics.compute()
    values["observed_data_residual"] = residual_sum / residual_count
    values["inference_ms_per_sample"] = inference_seconds * 1000.0 / len(dataset)
    values["peak_gpu_memory_mb"] = (
        torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    )
    result = {
        "baseline": args.baseline,
        "split": args.split,
        "samples": len(dataset),
        **values,
    }
    output = Path(config.output.root)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{args.baseline}_{args.split}_metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"saved={result_path.resolve()}")
    return values


if __name__ == "__main__":
    main()
