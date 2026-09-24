"""Encode a CIFAR-10 image into token IDs, then decode it back to an image."""

import argparse
from pathlib import Path

import numpy as np
import torch

from data import CIFAR10Images
from image_io import image_to_tensor, save_comparison, tensor_to_image
from model import ImageTokenizer, TokenizerConfig


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, help="Path to a custom image; resized to the training resolution")
    parser.add_argument("--cifar-index", type=int, default=0, help="Test split image index, used if --image is absent")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = TokenizerConfig(**checkpoint["model_config"])
    model = ImageTokenizer(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if args.image:
        original = image_to_tensor(args.image, config.image_size)
    else:
        dataset = CIFAR10Images(args.data_dir, train=False)
        if not 0 <= args.cifar_index < len(dataset):
            parser.error(f"--cifar-index must be between 0 and {len(dataset) - 1}")
        original = dataset[args.cifar_index]

    with torch.inference_mode():
        images = original.unsqueeze(0).to(device)
        ids = model.encode_to_indices(images)
        reconstructed = model.decode_from_indices(ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "tokens.npy", ids[0].cpu().numpy())
    tensor_to_image(images[0]).save(args.output_dir / "original.png")
    tensor_to_image(reconstructed[0]).save(args.output_dir / "reconstruction.png")
    save_comparison(images, reconstructed, args.output_dir / "comparison.png", count=1)
    mse = torch.mean((images - reconstructed.clamp(0, 1)).square()).item()
    print(
        f"token shape={tuple(ids[0].shape)} "
        f"token range=[{ids.min().item()}, {ids.max().item()}] "
        f"MSE={mse:.6f}; saved to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
