"""Deterministic Phase 6 benchmark suite and artifact generation."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .baselines import fbp_normalized_speed, fbp_unet_features
from .config import ProjectConfig, project_config_from_dict
from .data.masks import load_validation_masks, sector_mask_to_angle_mask
from .metrics import ReconstructionMetricAccumulator
from .physics import parallel_beam_project
from .training import build_model

ModelKind = Literal["fbp", "unet", "heterowave"]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trained_model(path: str | Path, device: torch.device) -> tuple[nn.Module, ProjectConfig, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = project_config_from_dict(checkpoint["config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    provenance = {
        "path": str(Path(path)),
        "sha256": file_sha256(path),
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "best_nrmse": float(checkpoint["best_nrmse"]),
        "model": config.model.name,
        "aggregation": config.model.aggregation,
    }
    return model, config, provenance


def predict_with_mask(
    kind: ModelKind,
    model: nn.Module | None,
    sinogram: Tensor,
    sector_mask: Tensor,
    metadata: dict[str, Any],
) -> Tensor:
    angle_mask = sector_mask_to_angle_mask(sector_mask, sinogram.shape[1]).to(sinogram.device)
    if kind == "fbp":
        return fbp_normalized_speed(sinogram, metadata, angle_mask=angle_mask)
    if model is None:
        raise ValueError(f"A trained model is required for {kind}")
    if kind == "unet":
        return model(fbp_unet_features(sinogram, metadata, angle_mask=angle_mask))
    output = model(sinogram, sector_mask)
    return output["mean"] if isinstance(output, dict) else output


def _observed_residual(
    prediction: Tensor,
    sinogram: Tensor,
    sector_mask: Tensor,
    metadata: dict[str, Any],
) -> tuple[float, int]:
    speed = prediction * float(metadata["speed_std"]) + float(metadata["speed_mean"])
    slowness_contrast = speed.clamp_min(1.0).reciprocal() - (1.0 / float(metadata["water_speed"]))
    reprojection = parallel_beam_project(
        slowness_contrast,
        num_angles=int(metadata["num_angles"]),
        detector_bins=int(metadata["detector_bins"]),
        align_corners=bool(metadata.get("align_corners", False)),
    )
    angle_mask = sector_mask_to_angle_mask(sector_mask, sinogram.shape[1]).to(sinogram.device)
    weights = angle_mask.unsqueeze(-1).expand_as(sinogram)
    return float(((reprojection - sinogram).abs() * weights).sum()), int(weights.sum())


@torch.inference_mode()
def evaluate_scenario(
    *,
    kind: ModelKind,
    label: str,
    model: nn.Module | None,
    loader: DataLoader,
    metadata: dict[str, Any],
    scenario: str,
    mask: Tensor,
    device: torch.device,
    max_samples: int | None = None,
) -> dict[str, Any]:
    metrics = ReconstructionMetricAccumulator()
    residual_sum = 0.0
    residual_count = 0
    inference_seconds = 0.0
    processed = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        remaining = None if max_samples is None else max_samples - processed
        if remaining is not None and remaining <= 0:
            break
        target = batch["target"][:remaining].to(device, non_blocking=True)
        sinogram = batch["sinogram"][:remaining].to(device, non_blocking=True)
        batch_mask = mask.unsqueeze(0).expand(len(sinogram), -1).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = predict_with_mask(kind, model, sinogram, batch_mask, metadata)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - started
        metrics.update(prediction, target)
        batch_residual, batch_count = _observed_residual(prediction, sinogram, batch_mask, metadata)
        residual_sum += batch_residual
        residual_count += batch_count
        processed += len(target)
    values = metrics.compute()
    return {
        "model": label,
        "scenario": scenario,
        "observed_sectors": int(mask.sum()),
        "observed_fraction": float(mask.float().mean()),
        "samples": processed,
        **values,
        "observed_data_residual": residual_sum / residual_count,
        "inference_ms_per_sample": inference_seconds * 1000.0 / processed,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
    }


def write_metrics_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_robustness(rows: list[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    paths = []
    families = {
        "random": {"all_16", "observed_12", "random_8", "random_4", "random_2"},
        "wedge": {"all_16", "contiguous_8", "contiguous_4", "contiguous_2"},
    }
    for family, scenarios in families.items():
        figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        labels = sorted({row["model"] for row in rows})
        for label in labels:
            selected = sorted(
                (row for row in rows if row["model"] == label and row["scenario"] in scenarios),
                key=lambda row: row["observed_fraction"],
            )
            axis.plot(
                [100 * row["observed_fraction"] for row in selected],
                [row["nrmse"] for row in selected],
                marker="o",
                label=label,
            )
        axis.set_xlabel("Observed sectors (%)")
        axis.set_ylabel("Normalized RMSE")
        axis.set_title(f"Robustness: {family} missingness")
        axis.grid(alpha=0.25)
        axis.legend()
        path = output / f"robustness_{family}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths


@torch.inference_mode()
def plot_qualitative_grid(
    *,
    dataset,
    masks: dict[str, Tensor],
    unet: nn.Module,
    heterowave: nn.Module,
    device: torch.device,
    path: str | Path,
    unet_label: str = "FBP + U-Net",
    index: int = 0,
) -> Path:
    scenarios = [name for name in ("all_16", "random_8", "contiguous_4", "random_2") if name in masks]
    sample = dataset[index]
    target = sample["target"].unsqueeze(0).to(device)
    sinogram = sample["sinogram"].unsqueeze(0).to(device)
    figure, axes = plt.subplots(len(scenarios), 6, figsize=(15, 3.2 * len(scenarios)), constrained_layout=True)
    axes = axes.reshape(len(scenarios), 6)
    vmin, vmax = float(target.amin()), float(target.amax())
    for row_index, scenario in enumerate(scenarios):
        sector_mask = masks[scenario].unsqueeze(0).to(device)
        fbp = predict_with_mask("fbp", None, sinogram, sector_mask, dataset.metadata)
        unet_prediction = predict_with_mask("unet", unet, sinogram, sector_mask, dataset.metadata)
        hetero_prediction = predict_with_mask("heterowave", heterowave, sinogram, sector_mask, dataset.metadata)
        images = [target, sector_mask.float().view(1, 1, 1, -1), fbp, unet_prediction, hetero_prediction, (hetero_prediction - target).abs()]
        titles = ["Target", "Available sectors", "FBP", unet_label, "HeteroWave", "HeteroWave |error|"]
        for column, (image, title) in enumerate(zip(images, titles)):
            axis = axes[row_index, column]
            array = image[0, 0].float().cpu().numpy()
            if column == 1:
                axis.imshow(array, cmap="binary", vmin=0, vmax=1, aspect="auto")
            elif column == 5:
                axis.imshow(array, cmap="magma", vmin=0)
            else:
                axis.imshow(array, cmap="gray", vmin=vmin, vmax=vmax)
            axis.set_axis_off()
            if row_index == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(scenario)
    path = Path(path)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_provenance(
    *,
    output_dir: str | Path,
    evaluation_config: ProjectConfig,
    split: str,
    mask_path: str | Path,
    checkpoint_provenance: list[dict[str, Any]],
    heterowave_checkpoint: str | Path,
) -> None:
    output = Path(output_dir)
    (output / "evaluation_config.json").write_text(
        json.dumps({"config": asdict(evaluation_config), "split": split, "validation_masks": str(mask_path)}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "config.yaml").write_text(
        yaml.safe_dump({"config": asdict(evaluation_config), "split": split, "validation_masks": str(mask_path)}, sort_keys=False),
        encoding="utf-8",
    )
    (output / "checkpoint_provenance.json").write_text(
        json.dumps(checkpoint_provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(heterowave_checkpoint, output / "model.pt")


def plot_architecture(config: ProjectConfig, path: str | Path) -> Path:
    """Save a compact architecture flow diagram for the evaluated HeteroWave model."""
    figure, axis = plt.subplots(figsize=(12, 2.8), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    labels = [
        "Masked\nsinogram",
        f"{config.physics.num_sectors} sector\nbackprojections",
        "Shared encoder\n" + " → ".join(map(str, config.model.channels)),
        f"Masked {config.model.aggregation}\nat every scale",
        "U-Net decoder",
        "Normalized\nspeed map",
    ]
    positions = torch.linspace(0.08, 0.92, len(labels)).tolist()
    for index, (x, label) in enumerate(zip(positions, labels)):
        axis.text(x, 0.5, label, ha="center", va="center", bbox={"boxstyle": "round,pad=0.5", "facecolor": "#e8f0fe", "edgecolor": "#496a9b"})
        if index + 1 < len(labels):
            axis.annotate("", xy=(positions[index + 1] - 0.06, 0.5), xytext=(x + 0.06, 0.5), arrowprops={"arrowstyle": "->", "color": "#444"})
    path = Path(path)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path
