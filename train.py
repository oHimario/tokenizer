"""Train the first-stage image tokenizer on CIFAR-10."""

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CIFAR10Images
from image_io import save_comparison
from model import ImageTokenizer, TokenizerConfig


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "runs" / "cifar10")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-batches", type=int, help="Limit batches for a quick smoke run")
    parser.add_argument("--max-eval-batches", type=int, help="Limit batches for a quick smoke run")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "num_workers", "learning_rate"):
        value = getattr(args, name)
        if value < 0 or (name != "num_workers" and value == 0):
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_train_batches", "max_eval_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def run_epoch(
    model: ImageTokenizer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor] | None]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "reconstruction_l1": 0.0, "quantizer": 0.0, "mse": 0.0}
    seen = 0
    example = None
    code_counts = (
        None
        if training
        else torch.zeros(model.config.codebook_size, dtype=torch.long, device=device)
    )
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_number, images in enumerate(loader):
            if max_batches is not None and batch_number >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            reconstructions, ids, quantizer_loss = model(images)
            reconstruction_loss = F.l1_loss(reconstructions, images)
            mse = F.mse_loss(reconstructions.clamp(0, 1), images)
            loss = reconstruction_loss + quantizer_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            count = len(images)
            seen += count
            for key, value in (
                ("loss", loss),
                ("reconstruction_l1", reconstruction_loss),
                ("quantizer", quantizer_loss),
                ("mse", mse),
            ):
                totals[key] += value.detach().item() * count
            if not training and example is None:
                example = (images[:8].cpu(), reconstructions[:8].cpu())
            if code_counts is not None:
                code_counts += torch.bincount(
                    ids.reshape(-1), minlength=model.config.codebook_size
                )
    if seen == 0:
        raise ValueError("The dataset loader yielded no batches")
    metrics = {key: value / seen for key, value in totals.items()}
    if code_counts is not None:
        frequencies = code_counts.float() / code_counts.sum().clamp_min(1)
        metrics["codebook_usage"] = (code_counts > 0).float().mean().item()
        metrics["codebook_perplexity"] = torch.exp(
            -(frequencies * frequencies.clamp_min(1e-12).log()).sum()
        ).item()
    return metrics, example


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    device = torch.device("cpu" if device_name == "auto" else device_name)

    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=True)
        config = TokenizerConfig(**resume_state["model_config"])
    else:
        config = TokenizerConfig(
            base_channels=args.base_channels,
            embedding_dim=args.embedding_dim,
            codebook_size=args.codebook_size,
        )

    print(f"Loading CIFAR-10 from {args.data_dir.resolve()} (downloads once if missing)", flush=True)
    train_data = CIFAR10Images(args.data_dir, train=True, download=True)
    test_data = CIFAR10Images(args.data_dir, train=False)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = ImageTokenizer(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.5, 0.9))
    first_epoch = 1
    best_val_loss = float("inf")
    if resume_state:
        model.load_state_dict(resume_state["model_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        first_epoch = resume_state["epoch"] + 1
        best_val_loss = resume_state["best_val_loss"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device} train={len(train_data)} test={len(test_data)} "
        f"token_grid={config.token_grid_size}x{config.token_grid_size} "
        f"codebook_size={config.codebook_size}",
        flush=True,
    )
    with tqdm(
        total=args.epochs,
        initial=min(first_epoch - 1, args.epochs),
        desc="Epochs",
        unit="epoch",
        ascii=True,
    ) as progress:
        for epoch in range(first_epoch, args.epochs + 1):
            train_metrics, _ = run_epoch(
                model, train_loader, device, optimizer, args.max_train_batches
            )
            val_metrics, example = run_epoch(
                model, test_loader, device, None, args.max_eval_batches
            )
            if example is not None:
                save_comparison(
                    example[0], example[1], args.output_dir / f"reconstruction_epoch_{epoch:03d}.png"
                )
            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
            checkpoint = {
                "epoch": epoch,
                "model_config": asdict(config),
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, args.output_dir / "last.pt")
            if improved:
                torch.save(checkpoint, args.output_dir / "best.pt")
            progress.set_postfix(
                train_loss=f"{train_metrics['loss']:.4f}",
                val_loss=f"{val_metrics['loss']:.4f}",
                refresh=False,
            )
            progress.update(1)
            tqdm.write(
                f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_l1={val_metrics['reconstruction_l1']:.4f} "
                f"val_mse={val_metrics['mse']:.5f} "
                f"val_psnr={-10 * np.log10(max(val_metrics['mse'], 1e-12)):.2f}dB "
                f"codebook_usage={val_metrics['codebook_usage']:.1%} "
                f"codebook_perplexity={val_metrics['codebook_perplexity']:.1f}"
            )


if __name__ == "__main__":
    main()
