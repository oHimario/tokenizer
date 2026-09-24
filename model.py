"""Convolutional VQ tokenizer adapted from MaskGIT's VQGAN tokenizer.

This module implements the encoder, nearest-neighbor codebook, and decoder.
The first training stage uses reconstruction and VQ losses only; it does not
claim to reproduce the adversarially trained VQGAN weights.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TokenizerConfig:
    image_size: int = 32
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    embedding_dim: int = 128
    codebook_size: int = 512
    commitment_cost: float = 0.25

    def __post_init__(self) -> None:
        if not self.channel_multipliers or any(m < 1 for m in self.channel_multipliers):
            raise ValueError("channel_multipliers must contain positive integers")
        if self.image_size % self.downsample_factor:
            raise ValueError("image_size must be divisible by the downsample factor")
        if min(self.base_channels, self.num_res_blocks, self.embedding_dim, self.codebook_size) < 1:
            raise ValueError("model dimensions and num_res_blocks must be positive")
        if self.commitment_cost < 0:
            raise ValueError("commitment_cost must be non-negative")

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.channel_multipliers) - 1)

    @property
    def token_grid_size(self) -> int:
        return self.image_size // self.downsample_factor


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual


class Encoder(nn.Module):
    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()
        channels = config.base_channels
        self.input_conv = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        stages: list[nn.Module] = []
        for index, multiplier in enumerate(config.channel_multipliers):
            next_channels = config.base_channels * multiplier
            for _ in range(config.num_res_blocks):
                stages.append(ResBlock(channels, next_channels))
                channels = next_channels
            if index < len(config.channel_multipliers) - 1:
                stages.append(nn.AvgPool2d(2, stride=2))
        stages.extend(ResBlock(channels, channels) for _ in range(config.num_res_blocks))
        self.stages = nn.Sequential(*stages)
        self.output_norm = _group_norm(channels)
        self.output_conv = nn.Conv2d(channels, config.embedding_dim, 1)

    def forward(self, image: Tensor) -> Tensor:
        x = self.stages(self.input_conv(image))
        return self.output_conv(F.silu(self.output_norm(x)))


class Decoder(nn.Module):
    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()
        channels = config.base_channels * config.channel_multipliers[-1]
        self.input_conv = nn.Conv2d(config.embedding_dim, channels, 3, padding=1)
        stages: list[nn.Module] = [
            ResBlock(channels, channels) for _ in range(config.num_res_blocks)
        ]
        for index in reversed(range(len(config.channel_multipliers))):
            next_channels = config.base_channels * config.channel_multipliers[index]
            for _ in range(config.num_res_blocks):
                stages.append(ResBlock(channels, next_channels))
                channels = next_channels
            if index > 0:
                stages.append(nn.Upsample(scale_factor=2, mode="nearest"))
                stages.append(nn.Conv2d(channels, channels, 3, padding=1))
        self.stages = nn.Sequential(*stages)
        self.output_norm = _group_norm(channels)
        self.output_conv = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, features: Tensor) -> Tensor:
        x = self.stages(self.input_conv(features))
        return self.output_conv(F.silu(self.output_norm(x)))


class VectorQuantizer(nn.Module):
    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()
        self.codebook = nn.Embedding(config.codebook_size, config.embedding_dim)
        nn.init.uniform_(
            self.codebook.weight,
            -config.embedding_dim ** -0.5,
            config.embedding_dim ** -0.5,
        )
        self.commitment_cost = config.commitment_cost

    def indices(self, features: Tensor) -> Tensor:
        batch, channels, height, width = features.shape
        flat = features.permute(0, 2, 3, 1).reshape(-1, channels).float()
        codebook = self.codebook.weight.float()
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            - 2 * flat @ codebook.t()
            + codebook.square().sum(dim=1).unsqueeze(0)
        )
        return distances.argmin(dim=1).reshape(batch, height, width)

    def decode_ids(self, ids: Tensor) -> Tensor:
        if ids.ndim != 3:
            raise ValueError("token IDs must have shape [batch, height, width]")
        vectors = self.codebook(ids.long())
        return vectors.permute(0, 3, 1, 2).contiguous()

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        ids = self.indices(features)
        quantized = self.decode_ids(ids)
        codebook_loss = F.mse_loss(quantized, features.detach())
        commitment_loss = self.commitment_cost * F.mse_loss(features, quantized.detach())
        straight_through = features + (quantized - features).detach()
        return straight_through, ids, codebook_loss + commitment_loss


class ImageTokenizer(nn.Module):
    def __init__(self, config: TokenizerConfig = TokenizerConfig()) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.quantizer = VectorQuantizer(config)
        self.decoder = Decoder(config)

    @torch.no_grad()
    def encode_to_indices(self, images: Tensor) -> Tensor:
        """Map [B, 3, H, W] images to integer [B, h, w] visual tokens."""
        if images.ndim != 4 or images.shape[1:] != (3, self.config.image_size, self.config.image_size):
            raise ValueError(
                f"images must have shape [B, 3, {self.config.image_size}, {self.config.image_size}]"
            )
        return self.quantizer.indices(self.encoder(images))

    @torch.no_grad()
    def decode_from_indices(self, ids: Tensor) -> Tensor:
        """Decode [B, h, w] token IDs to [B, 3, H, W] images."""
        expected = self.config.token_grid_size
        if ids.ndim != 3 or ids.shape[1:] != (expected, expected):
            raise ValueError(f"token IDs must have shape [B, {expected}, {expected}]")
        return self.decoder(self.quantizer.decode_ids(ids))

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if images.ndim != 4 or images.shape[1:] != (3, self.config.image_size, self.config.image_size):
            raise ValueError(
                f"images must have shape [B, 3, {self.config.image_size}, {self.config.image_size}]"
            )
        features = self.encoder(images)
        quantized, ids, quantizer_loss = self.quantizer(features)
        reconstruction = self.decoder(quantized)
        return reconstruction, ids, quantizer_loss
