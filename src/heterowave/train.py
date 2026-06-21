"""Train configured Phase 4 or Phase 5 reconstruction models."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from .acquisition import augment_sinogram
from .config import load_config
from .losses import observed_data_consistency_loss, reconstruction_loss
from .training import build_model, create_loaders, resolve_device, training_prediction, validate


def _atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_checkpoint(path: str | Path) -> dict:
    """Load training checkpoints without moving CPU RNG states onto CUDA."""
    return torch.load(path, map_location="cpu", weights_only=False)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_smoke.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--initialize-from", help="Warm-start model weights without optimizer or epoch state")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    train_loader, validation_loader, metadata = create_loaders(config)
    model = build_model(config).to(device)
    if args.resume and args.initialize_from:
        parser.error("--resume and --initialize-from are mutually exclusive")
    if args.initialize_from:
        checkpoint = _load_checkpoint(args.initialize_from)
        source = checkpoint["model"]
        target = model.state_dict()
        loaded, adapted, skipped = [], [], []
        for name, value in source.items():
            if name not in target:
                skipped.append(name)
            elif target[name].shape == value.shape:
                target[name] = value
                loaded.append(name)
            elif name in {"output.weight", "output.bias"} and value.shape[0] == 1 and target[name].shape[0] == 2:
                target[name][0].copy_(value[0])
                adapted.append(name)
            else:
                skipped.append(name)
        model.load_state_dict(target)
        print(
            f"initialized={args.initialize_from} loaded={len(loaded)} "
            f"adapted={len(adapted)} skipped={len(skipped)}"
        )
    if config.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay
    )
    output = Path(config.output.root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")

    mask_generator = torch.Generator().manual_seed(config.seed)
    physics_generator = torch.Generator().manual_seed(config.seed + 2)
    acquisition_generator = torch.Generator().manual_seed(config.seed + 3)
    start_epoch, global_step, best_nrmse = 0, 0, float("inf")
    resume_path = args.resume or config.training.resume
    if resume_path:
        checkpoint = _load_checkpoint(resume_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_nrmse = float(checkpoint["best_nrmse"])
        if "mask_generator_state" in checkpoint:
            mask_generator.set_state(checkpoint["mask_generator_state"])
        if "physics_generator_state" in checkpoint:
            physics_generator.set_state(checkpoint["physics_generator_state"])
        if "acquisition_generator_state" in checkpoint:
            acquisition_generator.set_state(checkpoint["acquisition_generator_state"])
        print(f"resumed={resume_path} step={global_step}")

    use_fp16 = device.type == "cuda" and config.precision in {"fp16", "auto"} and not torch.cuda.is_bf16_supported()
    use_bf16 = device.type == "cuda" and config.precision in {"bf16", "auto"} and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    stop = False
    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        for batch in train_loader:
            target = batch["target"].to(device, non_blocking=True)
            sinogram = batch["sinogram"].to(device, non_blocking=True)
            input_sinogram = augment_sinogram(sinogram, config.acquisition, generator=acquisition_generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_fp16 or use_bf16):
                raw_output, sector_mask = training_prediction(
                    model, input_sinogram, metadata, config, mask_generator, return_aux=True
                )
                if isinstance(raw_output, dict):
                    prediction = raw_output["mean"]
                    log_variance = raw_output["log_variance"]
                else:
                    prediction, log_variance = raw_output, None
                loss, parts = reconstruction_loss(
                    prediction,
                    target,
                    image_weight=config.loss.image_weight,
                    gradient_weight=config.loss.gradient_weight,
                    log_variance=log_variance,
                    uncertainty_weight=config.loss.uncertainty_weight,
                )
                compute_data_loss = (
                    config.loss.data_weight > 0
                    and (global_step + 1) % config.loss.data_every_n_steps == 0
                )
                if compute_data_loss:
                    data_loss = observed_data_consistency_loss(
                        prediction,
                        input_sinogram,
                        sector_mask,
                        metadata,
                        angle_fraction=config.loss.data_angle_fraction,
                        generator=physics_generator,
                    )
                    loss = loss + config.loss.data_weight * data_loss
                    parts["data_loss"] = data_loss.detach()
                    parts["loss"] = loss.detach()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if global_step == 1 or global_step % 10 == 0:
                print(
                    f"epoch={epoch + 1} step={global_step} loss={float(parts['loss']):.6f} "
                    f"image={float(parts['image_loss']):.6f} gradient={float(parts['gradient_loss']):.6f}"
                    + (f" data={float(parts['data_loss']):.6f}" if "data_loss" in parts else "")
                    + (
                        f" uncertainty={float(parts['uncertainty_nll']):.6f}"
                        if "uncertainty_nll" in parts
                        else ""
                    )
                )
            if config.training.max_steps is not None and global_step >= config.training.max_steps:
                stop = True
                break

        if (epoch + 1) % config.training.validate_every_epochs == 0 or stop:
            values = validate(model, validation_loader, metadata, device, config)
            print("validation=" + json.dumps(values, sort_keys=True))
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_nrmse": min(best_nrmse, values["nrmse"]),
                "metadata": metadata,
                "config": asdict(config),
                "mask_generator_state": mask_generator.get_state(),
                "physics_generator_state": physics_generator.get_state(),
                "acquisition_generator_state": acquisition_generator.get_state(),
            }
            _atomic_checkpoint(output / "last.pt", state)
            if values["nrmse"] < best_nrmse:
                best_nrmse = values["nrmse"]
                _atomic_checkpoint(output / "best.pt", state)
        if stop:
            break
    print(f"device={device} steps={global_step} checkpoint={(output / 'last.pt').resolve()}")
    return output / "last.pt"


if __name__ == "__main__":
    main()
