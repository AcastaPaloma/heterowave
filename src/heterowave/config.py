"""Typed YAML configuration with strict keys and dotted CLI overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass
class DataConfig:
    dataset: str = "synthetic"
    root: str | None = None
    image_size: int = 32
    batch_size: int = 2
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    train_samples: int = 16
    val_samples: int = 8
    preprocess: bool = False


@dataclass
class PhysicsConfig:
    num_angles: int = 16
    detector_bins: int | None = None
    num_sectors: int = 16
    align_corners: bool = False


@dataclass
class VisualizationConfig:
    output: str = "outputs/physics_smoke.png"


@dataclass
class ModelConfig:
    name: str = "fbp_unet"
    channels: list[int] = field(default_factory=lambda: [16, 32, 64, 96])
    residual_output: bool = True
    aggregation: str = "mean_var_count"
    geometry_channels: bool = True
    uncertainty: bool = False


@dataclass
class MaskingConfig:
    minimum_sectors: int = 2
    random_probability: float = 0.50
    wedge_probability: float = 0.35
    periodic_probability: float = 0.15
    validation_masks: str | None = None


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4


@dataclass
class LossConfig:
    image_weight: float = 1.0
    gradient_weight: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 30
    max_steps: int | None = None
    gradient_clip: float = 1.0
    validate_every_epochs: int = 1
    checkpoint_metric: str = "nrmse"
    checkpoint_mode: str = "min"
    resume: str | None = None


@dataclass
class OutputConfig:
    root: str = "outputs/baseline"


@dataclass
class ProjectConfig:
    seed: int = 1337
    device: str = "cuda"
    precision: str = "fp32"
    compile: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if self.precision not in {"fp32", "fp16", "bf16", "auto"}:
            raise ValueError(f"Unsupported precision: {self.precision}")
        if self.data.dataset not in {"synthetic", "cached"}:
            raise ValueError("dataset must be 'synthetic' or 'cached'")
        if self.data.dataset == "cached" and not self.data.root:
            raise ValueError("data.root is required for dataset='cached'")
        if self.data.image_size < 8 or self.physics.num_angles < 1:
            raise ValueError("image_size must be >= 8 and num_angles must be positive")
        if self.physics.num_sectors < 1:
            raise ValueError("num_sectors must be positive")
        if self.model.name == "heterowave" and self.physics.num_angles % self.physics.num_sectors != 0:
            raise ValueError("HeteroWave requires num_sectors to divide num_angles")
        if self.data.batch_size < 1 or self.data.train_samples < 1 or self.data.val_samples < 1:
            raise ValueError("batch size and synthetic split sizes must be positive")
        if self.physics.detector_bins is not None and self.physics.detector_bins < 2:
            raise ValueError("detector_bins must be at least two when provided")
        if self.data.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if self.data.persistent_workers and self.data.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        if self.data.preprocess:
            raise ValueError("Preprocessing must be run explicitly before training")
        if self.model.name not in {"fbp_unet", "masked_fbp_unet", "heterowave"} or len(self.model.channels) < 2:
            raise ValueError(
                "model.name must be fbp_unet, masked_fbp_unet, or heterowave with at least two channel levels"
            )
        if any(channel < 1 for channel in self.model.channels):
            raise ValueError("model channels must be positive")
        if self.model.aggregation not in {"mean", "mean_var", "mean_var_count"}:
            raise ValueError("Unsupported set aggregation mode")
        if not 1 <= self.masking.minimum_sectors <= self.physics.num_sectors:
            raise ValueError("minimum_sectors must be within the configured sector count")
        probability_sum = (
            self.masking.random_probability
            + self.masking.wedge_probability
            + self.masking.periodic_probability
        )
        if min(
            self.masking.random_probability,
            self.masking.wedge_probability,
            self.masking.periodic_probability,
        ) < 0 or abs(probability_sum - 1.0) > 1e-6:
            raise ValueError("mask probabilities must be nonnegative and sum to one")
        if self.optimizer.name != "adamw" or self.optimizer.learning_rate <= 0 or self.optimizer.weight_decay < 0:
            raise ValueError("Phase 4 supports AdamW with a positive learning rate")
        if min(self.loss.image_weight, self.loss.gradient_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.training.epochs < 1 or self.training.validate_every_epochs < 1:
            raise ValueError("training epochs and validation interval must be positive")
        if self.training.max_steps is not None and self.training.max_steps < 1:
            raise ValueError("training.max_steps must be positive when provided")
        if self.training.checkpoint_metric != "nrmse" or self.training.checkpoint_mode != "min":
            raise ValueError("Phase 4 checkpoints use minimum validation NRMSE")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _dataclass_type(annotation: Any) -> type[Any] | None:
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return next((arg for arg in get_args(annotation) if isinstance(arg, type) and is_dataclass(arg)), None)
    return None


def _construct(cls: type[T], values: dict[str, Any], path: str = "config") -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise KeyError(f"Unknown {path} key(s): {names}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        nested = _dataclass_type(hints[key])
        if nested is not None:
            if not isinstance(value, dict):
                raise TypeError(f"{path}.{key} must be a mapping")
            value = _construct(nested, value, f"{path}.{key}")
        kwargs[key] = value
    return cls(**kwargs)


def _set_override(values: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be key=value: {expression}")
    dotted_key, raw_value = expression.split("=", 1)
    keys = dotted_key.split(".")
    target = values
    for key in keys[:-1]:
        child = target.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot apply nested override beneath {key}")
        target = child
    target[keys[-1]] = yaml.safe_load(raw_value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ProjectConfig:
    """Load a strict project configuration and apply dotted ``key=value`` overrides."""
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise TypeError("Top-level configuration must be a mapping")
    for override in overrides or []:
        _set_override(values, override)
    config = _construct(ProjectConfig, values)
    config.validate()
    return config


def project_config_from_dict(values: dict[str, Any]) -> ProjectConfig:
    """Reconstruct and validate a configuration embedded in a checkpoint."""
    config = _construct(ProjectConfig, values)
    config.validate()
    return config
