from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MomentLikeConfig:
    input_length: int = 4096
    patch_size: int = 4
    patch_stride: int = 4
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.10
    activation: str = "gelu"
    max_tokens: int = 1024


def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class MomentLikeReconstructor(nn.Module):
    """MOMENT-like transformer encoder for masked 1D signal reconstruction."""

    def __init__(self, config: MomentLikeConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Linear(config.patch_size, config.d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.patch_size),
        )
        self.register_buffer(
            "positional_embedding",
            sinusoidal_positions(config.max_tokens, config.d_model).unsqueeze(0),
            persistent=False,
        )
        nn.init.normal_(self.mask_token, std=0.02)

    @property
    def n_tokens(self) -> int:
        return 1 + (self.config.input_length - self.config.patch_size) // self.config.patch_stride

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected input shape (B, L) or (B, 1, L), got {tuple(x.shape)}")
        patches = x.unfold(dimension=-1, size=self.config.patch_size, step=self.config.patch_stride)
        return patches.squeeze(1)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, patch_size = patches.shape
        if patch_size != self.config.patch_size:
            raise ValueError("Unexpected patch size")
        fold_in = patches.transpose(1, 2)
        out = F.fold(
            fold_in,
            output_size=(1, self.config.input_length),
            kernel_size=(1, self.config.patch_size),
            stride=(1, self.config.patch_stride),
        )
        denom = F.fold(
            torch.ones_like(fold_in),
            output_size=(1, self.config.input_length),
            kernel_size=(1, self.config.patch_size),
            stride=(1, self.config.patch_stride),
        ).clamp_min(1.0)
        return (out / denom).view(bsz, 1, self.config.input_length)

    def encode(self, x: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        patches = self.patchify(x)
        tokens = self.patch_embed(patches)
        if token_mask is not None:
            if token_mask.shape != tokens.shape[:2]:
                raise ValueError(f"token_mask shape {tuple(token_mask.shape)} does not match tokens {tuple(tokens.shape[:2])}")
            mask = token_mask.to(device=tokens.device, dtype=torch.bool).unsqueeze(-1)
            tokens = torch.where(mask, self.mask_token.expand(tokens.shape[0], tokens.shape[1], -1), tokens)
        if tokens.shape[1] > self.positional_embedding.shape[1]:
            raise ValueError("Increase max_tokens in model config")
        tokens = tokens + self.positional_embedding[:, : tokens.shape[1]].to(tokens.device)
        return self.encoder(tokens)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encode(x, token_mask=token_mask)
        patches = self.reconstruction_head(encoded)
        return self.unpatchify(patches)

    def global_embedding(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        pool: str = "mean",
    ) -> torch.Tensor:
        """Return one fixed-width encoder embedding per signal."""
        encoded = self.encode(x, token_mask=token_mask)
        if pool == "mean":
            return encoded.mean(dim=1)
        if pool == "max":
            return encoded.max(dim=1).values
        raise ValueError(f"Unsupported embedding pool: {pool}")

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        keys = ("patch_embed.", "encoder.", "mask_token")
        return {k: v for k, v in self.state_dict().items() if k.startswith(keys) or k == "mask_token"}


class TinyTCNAutoencoder(nn.Module):
    """Small non-transformer baseline for reconstruction smoke tests."""

    def __init__(self, channels: int = 64, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding * 2, dilation=2),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding * 4, dilation=4),
            nn.GELU(),
            nn.Conv1d(channels, 1, kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.net(x)
