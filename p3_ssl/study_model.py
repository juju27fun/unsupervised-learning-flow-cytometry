from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .models import MomentLikeConfig, MomentLikeReconstructor
from .study_data import CONTINUOUS_FACTORS


@dataclass(frozen=True)
class YeastStudyModelConfig:
    input_length: int = 4096
    patch_size: int = 16
    patch_stride: int = 16
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    dim_feedforward: int = 384
    dropout: float = 0.10
    activation: str = "gelu"
    max_tokens: int = 256


class YeastStudyModel(nn.Module):
    def __init__(self, config: YeastStudyModelConfig) -> None:
        super().__init__()
        self.config = config
        self.reconstructor = MomentLikeReconstructor(
            MomentLikeConfig(
                input_length=config.input_length,
                patch_size=config.patch_size,
                patch_stride=config.patch_stride,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                max_tokens=config.max_tokens,
            )
        )
        self.continuous_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, len(CONTINUOUS_FACTORS)),
        )
        self.component_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, 2),
        )

    def forward(self, signals: torch.Tensor, token_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        encoded = self.reconstructor.encode(signals, token_mask=token_mask)
        embedding = encoded.mean(dim=1)
        patches = self.reconstructor.reconstruction_head(encoded)
        reconstruction = self.reconstructor.unpatchify(patches)
        return {
            "reconstruction": reconstruction,
            "embedding": embedding,
            "continuous": self.continuous_head(embedding),
            "component_logits": self.component_head(embedding),
        }


def physics_supervision_loss(
    output: dict[str, torch.Tensor],
    continuous_targets: torch.Tensor,
    continuous_valid: torch.Tensor,
    component_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    squared = torch.square(output["continuous"] - continuous_targets)
    valid = continuous_valid.to(dtype=squared.dtype)
    continuous = (squared * valid).sum() / valid.sum().clamp_min(1.0)
    component = F.cross_entropy(output["component_logits"], component_target)
    return continuous, component


def paired_nuisance_consistency(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.ndim != 3 or embeddings.shape[1] < 2:
        raise ValueError("embeddings must have shape (batch, views>=2, features)")
    normalized = F.normalize(embeddings, dim=-1)
    anchor = normalized[:, :1]
    return (1.0 - torch.sum(anchor * normalized[:, 1:], dim=-1)).mean()
