from __future__ import annotations

import torch

from p3_ssl.config import load_config, validate_study_config
from p3_ssl.study_training import (
    build_mask_batch,
    evaluate_embedding_health,
    evaluate_physics_predictions,
    interpolation_baseline,
    nearest_baseline,
)


def test_rebuild_config_is_coherent() -> None:
    config = load_config("configs/yeast_ssl_rebuild_v1.yaml")
    validate_study_config(config)
    assert config["model"]["patch_size"] == 16
    assert "test" in config["data"]["forbidden_training_splits"]
    assert config["training"]["adaptation_replay"] == {
        "synthetic_weight_start": 0.30,
        "synthetic_weight_end": 0.10,
    }


def test_mask_batch_has_256_tokens_and_trivial_controls() -> None:
    config = load_config("configs/yeast_ssl_rebuild_v1.yaml")
    signal = torch.sin(torch.arange(4096, dtype=torch.float32) / 20.0).reshape(1, 1, -1)
    event = torch.zeros(1, 4096, dtype=torch.bool)
    event[:, 1500:2500] = True
    target, tokens, hidden = build_mask_batch(signal, event, config, seed=3)
    assert target.shape == (1, 4096)
    assert tokens.shape == (1, 256)
    assert hidden.shape == (1, 4096)
    interpolation = interpolation_baseline(signal, hidden)
    nearest = nearest_baseline(signal, hidden)
    assert interpolation.shape == signal.shape
    assert nearest.shape == signal.shape
    assert torch.isfinite(interpolation).all()


def test_physics_evaluation_reports_constant_and_majority_controls() -> None:
    class ConstantModel(torch.nn.Module):
        def forward(self, signals: torch.Tensor, token_mask=None):
            count = signals.shape[0]
            return {
                "continuous": torch.zeros(count, 5),
                "component_logits": torch.tensor([[1.0, 0.0]]).repeat(count, 1),
                "embedding": torch.ones(count, 3),
            }

    batch = {
        "signals": torch.zeros(2, 2, 1, 4),
        "continuous_targets": torch.tensor([[0.25] * 5, [0.75] * 5]),
        "continuous_valid": torch.ones(2, 5, dtype=torch.bool),
        "component_target": torch.tensor([0, 1]),
    }
    metrics = evaluate_physics_predictions(ConstantModel(), [batch], torch.device("cpu"))
    assert metrics["constant_prior_mean_normalized_mse"] == 0.0625
    assert metrics["mean_normalized_mse"] == 0.3125
    assert metrics["relative_mse_reduction_vs_constant_by_factor"]["duration_ms"] == -4.0
    assert metrics["component_count_accuracy"] == 0.5
    assert metrics["majority_component_count_accuracy"] == 0.5
    assert metrics["component_count_accuracy_gain"] == 0.0


def test_embedding_health_detects_constant_representation() -> None:
    class ConstantModel(torch.nn.Module):
        def forward(self, signals: torch.Tensor, token_mask=None):
            return {"embedding": torch.ones(signals.shape[0], 4)}

    batch = {"signal": torch.randn(3, 1, 16)}
    metrics = evaluate_embedding_health(
        ConstantModel(), [batch], simulation=False, device=torch.device("cpu")
    )
    assert metrics["active_dimensions_std_gt_1e_3"] == 0
    assert metrics["effective_rank"] == 1.0
    assert metrics["mean_off_diagonal_cosine_similarity"] == 1.0
