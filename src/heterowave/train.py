"""Train the Phase 4 fixed-input FBP + U-Net baseline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from .baselines import fbp_unet_features
from .config import load_config
from .losses import reconstruction_loss
from .models import FBPUNet
from .training import create_loaders, resolve_device, validate


def _atomic_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_smoke.yaml")
    parser.add_argument("--resume")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    config = load_config(args.config, args.overrides)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    train_loader, validation_loader, metadata = create_loaders(config)
    model = FBPUNet(
        channels=config.model.channels, residual_output=config.model.residual_output
    ).to(device)
    if config.compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay
    )
    output = Path(config.output.root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")

    start_epoch, global_step, best_nrmse = 0, 0, float("inf")
    resume_path = args.resume or config.training.resume
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_nrmse = float(checkpoint["best_nrmse"])
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
            with torch.no_grad():
                features = fbp_unet_features(sinogram, metadata)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_fp16 or use_bf16):
                prediction = model(features)
                loss, parts = reconstruction_loss(
                    prediction,
                    target,
                    image_weight=config.loss.image_weight,
                    gradient_weight=config.loss.gradient_weight,
                )
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
                )
            if config.training.max_steps is not None and global_step >= config.training.max_steps:
                stop = True
                break

        if (epoch + 1) % config.training.validate_every_epochs == 0 or stop:
            values = validate(model, validation_loader, metadata, device)
            print("validation=" + json.dumps(values, sort_keys=True))
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_nrmse": min(best_nrmse, values["nrmse"]),
                "metadata": metadata,
                "config": asdict(config),
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
