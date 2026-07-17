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


def test_local_spectral_head_preserves_encoder_initialization() -> None:
    common = {
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 1,
        "dim_feedforward": 64,
        "dropout": 0.0,
    }
    torch.manual_seed(42)
    control = YeastStudyModel(YeastStudyModelConfig(**common))
    torch.manual_seed(42)
    treatment = YeastStudyModel(
        YeastStudyModelConfig(**common, local_spectral_features=24)
    )
    control_state = control.reconstructor.encoder_state_dict()
    treatment_state = treatment.reconstructor.encoder_state_dict()
    assert control_state.keys() == treatment_state.keys()
    assert all(torch.equal(control_state[key], treatment_state[key]) for key in control_state)
    output = treatment.forward_local_spectral(
        torch.randn(2, 1, 4096), torch.zeros(2, 256, dtype=torch.bool)
    )
    assert output["embedding"].shape == (2, 32)
    assert output["local_spectral_prediction"].shape == (2, 256, 24)
