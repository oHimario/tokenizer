"""Image conversion utilities shared by training and reconstruction."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import Tensor


def tensor_to_image(image: Tensor) -> Image.Image:
    pixels = (
        image.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(pixels, mode="RGB")


def image_to_tensor(path: str | Path, image_size: int) -> Tensor:
    with Image.open(path) as source:
        rgb = source.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BICUBIC
        )
        pixels = np.asarray(rgb, dtype=np.uint8).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)


def save_comparison(images: Tensor, reconstructions: Tensor, path: str | Path, count: int = 8) -> None:
    number = min(len(images), count)
    if number < 1:
        raise ValueError("No images to save")
    height, width = images.shape[-2:]
    canvas = Image.new("RGB", (number * width, 2 * height))
    for index in range(number):
        canvas.paste(tensor_to_image(images[index]), (index * width, 0))
        canvas.paste(tensor_to_image(reconstructions[index]), (index * width, height))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
