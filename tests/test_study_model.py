from __future__ import annotations

import torch

from p3_ssl.study_model import (
    YeastStudyModel,
    YeastStudyModelConfig,
    paired_nuisance_consistency,
    physics_supervision_loss,
)


def test_study_model_shape_and_losses() -> None:
    model = YeastStudyModel(
        YeastStudyModelConfig(d_model=32, n_heads=4, n_layers=1, dim_feedforward=64)
    )
    signals = torch.randn(2, 1, 4096)
    token_mask = torch.zeros(2, 256, dtype=torch.bool)
    token_mask[:, 10:20] = True
    output = model(signals, token_mask)
    assert output["reconstruction"].shape == signals.shape
    assert output["embedding"].shape == (2, 32)
    continuous, component = physics_supervision_loss(
        output,
        torch.zeros(2, 5),
        torch.ones(2, 5, dtype=torch.bool),
        torch.tensor([0, 1]),
    )
    assert torch.isfinite(continuous)
    assert torch.isfinite(component)


def test_consistency_is_zero_for_equal_views() -> None:
    embeddings = torch.randn(3, 8)
    paired = torch.stack([embeddings, embeddings], dim=1)
    assert float(paired_nuisance_consistency(paired)) < 1.0e-6
