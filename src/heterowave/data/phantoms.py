"""Small in-memory sound-speed phantom generator for development and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class PhantomRegion:
    shape: Literal["circle", "ellipse", "rectangle"]
    center: tuple[float, float]
    size: tuple[float, float]
    speed: float
    angle_degrees: float = 0.0


def _coordinate_grid(size: int, *, device: torch.device | str | None, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    axis = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    return torch.meshgrid(axis, axis, indexing="ij")


def render_phantom(
    size: int,
    regions: Sequence[PhantomRegion],
    *,
    background_speed: float = 1500.0,
    noise_std: float = 0.0,
    seed: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Render one phantom as ``[1, H, W]`` using normalized coordinates."""
    if size < 2 or noise_std < 0:
        raise ValueError("size must be >= 2 and noise_std must be nonnegative")
    yy, xx = _coordinate_grid(size, device=device, dtype=dtype)
    image = torch.full((size, size), background_speed, device=device, dtype=dtype)
    for region in regions:
        cx, cy = region.center
        sx, sy = region.size
        if sx <= 0 or sy <= 0:
            raise ValueError("Region sizes must be positive")
        radians = torch.tensor(region.angle_degrees * torch.pi / 180.0, device=image.device, dtype=dtype)
        cosine, sine = torch.cos(radians), torch.sin(radians)
        dx, dy = xx - cx, yy - cy
        xr = cosine * dx + sine * dy
        yr = -sine * dx + cosine * dy
        if region.shape == "circle":
            mask = xr.square() + yr.square() <= sx * sx
        elif region.shape == "ellipse":
            mask = (xr / sx).square() + (yr / sy).square() <= 1.0
        elif region.shape == "rectangle":
            mask = (xr.abs() <= sx) & (yr.abs() <= sy)
        else:
            raise ValueError(f"Unsupported region shape: {region.shape}")
        image = torch.where(mask, torch.as_tensor(region.speed, device=image.device, dtype=dtype), image)
    if noise_std:
        generator = torch.Generator(device=image.device)
        if seed is not None:
            generator.manual_seed(seed)
        image = image + torch.randn(image.shape, generator=generator, device=image.device, dtype=dtype) * noise_std
    return image.unsqueeze(0)


def make_disk_phantom(
    size: int,
    *,
    radius: float = 0.45,
    background_speed: float = 1500.0,
    disk_speed: float = 1540.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    return render_phantom(
        size,
        [PhantomRegion("circle", (0.0, 0.0), (radius, radius), disk_speed)],
        background_speed=background_speed,
        device=device,
        dtype=dtype,
    )


def make_random_phantoms(
    batch_size: int,
    size: int,
    *,
    seed: int = 1337,
    background_speed: float = 1500.0,
    noise_std: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate reproducible mixed-shape phantoms as ``[B, 1, H, W]``."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    shape_names = ("circle", "ellipse", "rectangle")
    images = []
    for sample in range(batch_size):
        regions = []
        count = int(torch.randint(2, 5, (), generator=rng).item())
        for _ in range(count):
            shape = shape_names[int(torch.randint(0, len(shape_names), (), generator=rng).item())]
            center = tuple((torch.rand(2, generator=rng) * 1.0 - 0.5).tolist())
            extent = torch.rand(2, generator=rng) * 0.18 + 0.12
            if shape == "circle":
                extent[1] = extent[0]
            speed = float((torch.rand((), generator=rng) * 100.0 + 1450.0).item())
            angle = float((torch.rand((), generator=rng) * 180.0).item())
            regions.append(PhantomRegion(shape, center, tuple(extent.tolist()), speed, angle))
        images.append(render_phantom(size, regions, background_speed=background_speed, noise_std=noise_std, seed=seed + sample, device=device, dtype=dtype))
    return torch.stack(images)


def speed_to_slowness_contrast(speed: Tensor, water_speed: float = 1500.0) -> Tensor:
    if torch.any(speed <= 0):
        raise ValueError("Sound speed must be positive")
    return speed.reciprocal() - (1.0 / water_speed)

