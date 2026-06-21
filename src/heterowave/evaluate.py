"""Evaluate individual baselines or the deterministic Phase 6 suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import CachedHeteroWaveDataset, SyntheticReconstructionDataset, load_validation_masks
from .evaluation import (
    evaluate_scenario,
    load_trained_model,
    plot_architecture,
    plot_qualitative_grid,
    plot_robustness,
    save_provenance,
    write_metrics_csv,
)
from .training import resolve_device


def _dataset(config, split):
    if config.data.dataset == "cached":
        return CachedHeteroWaveDataset(config.data.root, split)
    count = config.data.train_samples if split == "train" else config.data.val_samples
    return SyntheticReconstructionDataset(count, config.data.image_size, config.physics.num_angles, seed=config.seed + 1)


def _loader(config, dataset):
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
    )


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_smoke.yaml")
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--baseline", choices=("fbp", "unet"), default="fbp")
    parser.add_argument("--checkpoint", help="Checkpoint for single-model U-Net evaluation")
    parser.add_argument("--unet-checkpoint")
    parser.add_argument("--masked-unet-checkpoint")
    parser.add_argument("--heterowave-checkpoint")
    parser.add_argument("--masks")
    parser.add_argument("--output-dir")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--qualitative-index", type=int, default=0)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    device = resolve_device(config.device)
    dataset = _dataset(config, args.split)
    loader = _loader(config, dataset)
    mask_path = args.masks or config.masking.validation_masks
    if mask_path:
        masks = load_validation_masks(mask_path)
    else:
        masks = {"all_16": torch.ones(config.physics.num_sectors, dtype=torch.bool)}

    if not args.suite:
        model = None
        label = "fbp"
        if args.baseline == "unet":
            if not args.checkpoint:
                parser.error("--checkpoint is required for the unet baseline")
            model, model_config, _ = load_trained_model(args.checkpoint, device)
            label = model_config.model.name
        scenario = "all_16"
        row = evaluate_scenario(
            kind=args.baseline,
            label=label,
            model=model,
            loader=loader,
            metadata=dataset.metadata,
            scenario=scenario,
            mask=masks[scenario],
            device=device,
            max_samples=args.max_samples,
        )
        output = Path(args.output_dir or config.output.root)
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{label}_{args.split}_metrics.json"
        path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(row, indent=2, sort_keys=True))
        print(f"saved={path.resolve()}")
        return row

    if not (args.unet_checkpoint or args.masked_unet_checkpoint) or not args.heterowave_checkpoint:
        parser.error(
            "--suite requires --heterowave-checkpoint and at least one of "
            "--unet-checkpoint or --masked-unet-checkpoint"
        )
    if not mask_path:
        parser.error("--suite requires deterministic masks through --masks or config.masking.validation_masks")
    output = Path(args.output_dir or config.output.root)
    output.mkdir(parents=True, exist_ok=True)
    heterowave, heterowave_config, heterowave_provenance = load_trained_model(args.heterowave_checkpoint, device)
    heterowave_kind = (
        heterowave_config.model.name
        if heterowave_config.model.name in {"heterowave_v2", "heterowave_v3"}
        else "heterowave"
    )
    heterowave_label = f"{heterowave_config.model.name}_{heterowave_config.model.aggregation}"
    models = [("fbp", "fbp", None)]
    checkpoint_provenance = []
    qualitative_unet = None
    qualitative_unet_label = "FBP + U-Net"
    if args.unet_checkpoint:
        unet, unet_config, unet_provenance = load_trained_model(args.unet_checkpoint, device)
        models.append(("unet", unet_config.model.name, unet))
        checkpoint_provenance.append(unet_provenance)
        qualitative_unet = unet
        qualitative_unet_label = unet_config.model.name
    if args.masked_unet_checkpoint:
        masked_unet, masked_config, masked_provenance = load_trained_model(args.masked_unet_checkpoint, device)
        if masked_config.model.name != "masked_fbp_unet":
            parser.error("--masked-unet-checkpoint must contain a masked_fbp_unet model")
        models.append(("unet", masked_config.model.name, masked_unet))
        checkpoint_provenance.append(masked_provenance)
        qualitative_unet = masked_unet
        qualitative_unet_label = "Masked FBP + U-Net"
    models.append((heterowave_kind, heterowave_label, heterowave))
    checkpoint_provenance.append(heterowave_provenance)
    rows = []
    for scenario, mask in masks.items():
        for kind, label, model in models:
            row = evaluate_scenario(
                kind=kind,
                label=label,
                model=model,
                loader=loader,
                metadata=dataset.metadata,
                scenario=scenario,
                mask=mask,
                device=device,
                max_samples=args.max_samples,
            )
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    write_metrics_csv(rows, output / "metrics_by_scenario.csv")
    plot_robustness(rows, output)
    plot_qualitative_grid(
        dataset=dataset,
        masks=masks,
        unet=qualitative_unet,
        unet_label=qualitative_unet_label,
        heterowave_kind=heterowave_kind,
        heterowave=heterowave,
        device=device,
        path=output / "qualitative_grid.png",
        index=args.qualitative_index,
    )
    plot_architecture(heterowave_config, output / "architecture.png")
    save_provenance(
        output_dir=output,
        evaluation_config=config,
        split=args.split,
        mask_path=mask_path,
        checkpoint_provenance=checkpoint_provenance,
        heterowave_checkpoint=args.heterowave_checkpoint,
    )
    summary = {
        "split": args.split,
        "samples": rows[0]["samples"],
        "scenarios": list(masks),
        "models": [label for _, label, _ in models],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"saved={output.resolve()}")
    return rows


if __name__ == "__main__":
    main()
