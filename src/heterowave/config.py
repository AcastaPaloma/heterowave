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
    image_size: int = 32
    batch_size: int = 2
    num_workers: int = 0
    train_samples: int = 16
    val_samples: int = 8
    preprocess: bool = False


@dataclass
class PhysicsConfig:
    num_angles: int = 16
    detector_bins: int | None = None
    align_corners: bool = False


@dataclass
class VisualizationConfig:
    output: str = "outputs/physics_smoke.png"


@dataclass
class ProjectConfig:
    seed: int = 1337
    device: str = "cuda"
    precision: str = "fp32"
    compile: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def validate(self) -> None:
        if self.precision not in {"fp32", "fp16", "bf16", "auto"}:
            raise ValueError(f"Unsupported precision: {self.precision}")
        if self.data.dataset != "synthetic":
            raise ValueError("Phase 1-2 supports only dataset='synthetic'")
        if self.data.image_size < 8 or self.physics.num_angles < 1:
            raise ValueError("image_size must be >= 8 and num_angles must be positive")
        if self.data.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if self.data.preprocess:
            raise ValueError("Dataset preprocessing is outside the Phase 1-2 scope")

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

