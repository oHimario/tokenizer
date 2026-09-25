"""Print discrete token IDs produced from an image by a trained tokenizer."""

import argparse
from pathlib import Path

import numpy as np
import torch

from data import CIFAR10Images
from image_io import image_to_tensor
from model import ImageTokenizer, TokenizerConfig


PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "runs" / "cifar10" / "best.pt",
        help="Trained tokenizer checkpoint (default: runs/cifar10/best.pt)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Image file to encode; resized to training resolution")
    source.add_argument(
        "--cifar-index", type=int, default=0, help="CIFAR-10 test image index (default: 0)"
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save-npy", type=Path, help="Optionally save the token grid as a .npy file")
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

    if args.image is not None:
        image = image_to_tensor(args.image, config.image_size)
        source_name = str(args.image)
    else:
        dataset = CIFAR10Images(args.data_dir, train=False)
        if not 0 <= args.cifar_index < len(dataset):
            parser.error(f"--cifar-index must be between 0 and {len(dataset) - 1}")
        image = dataset[args.cifar_index]
        source_name = f"CIFAR-10 test image {args.cifar_index}"

    with torch.inference_mode():
        tokens = model.encode_to_indices(image.unsqueeze(0).to(device))[0].cpu().numpy()

    print(f"source: {source_name}")
    print(f"token grid: {tokens.shape}, codebook size: {config.codebook_size}")
    print(np.array2string(tokens, threshold=tokens.size, max_line_width=120))

    if args.save_npy is not None:
        if args.save_npy.suffix.lower() != ".npy":
            parser.error("--save-npy path must end with .npy")
        args.save_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_npy, tokens)
        print(f"saved: {args.save_npy.resolve()}")


if __name__ == "__main__":
    main()
