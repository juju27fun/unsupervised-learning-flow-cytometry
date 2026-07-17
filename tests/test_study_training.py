from __future__ import annotations

from pathlib import Path

import torch

from p3_ssl.config import load_config, validate_mask_ablation_config, validate_study_config
from p3_ssl.reconstruction_diagnostics import run_fixed_mask_overfit
from p3_ssl.study_model import YeastStudyModel, YeastStudyModelConfig
from p3_ssl.study_training import (
    build_mask_batch,
    evaluate_embedding_health,
    evaluate_physics_predictions,
    interpolation_baseline,
    nearest_baseline,
    reconstruction_error_components,
    visible_mean_baseline,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs/yeast_ssl_rebuild_v1.yaml"
MASK_ABLATION_CONFIG_PATH = (
    Path(__file__).parents[1] / "configs/yeast_ssl_mask_ablation_v1.yaml"
)


def test_rebuild_config_is_coherent() -> None:
    config = load_config(CONFIG_PATH)
    validate_study_config(config)
    assert config["model"]["patch_size"] == 16
    assert "test" in config["data"]["forbidden_training_splits"]
    assert config["training"]["adaptation_replay"] == {
        "synthetic_weight_start": 0.30,
        "synthetic_weight_end": 0.10,
    }
    assert config["study"]["real_dataset"] == "yeast-events-representation@v3"
    assert config["training"]["representation_seeds"] == [42, 43, 44]
    assert config["study"]["scope_decision"]["acquisition_ood_claim_allowed"] is False


def test_mask_ablation_is_controlled_and_predeclared() -> None:
    config = load_config(MASK_ABLATION_CONFIG_PATH)
    validate_mask_ablation_config(config)
    assert config["policies"]["L0"]["min_block_ms"] == 0.128
    assert config["policies"]["S25"]["mask_ratio"] == 0.25
    assert config["policies"]["S10"]["mask_ratio"] == 0.10
    assert config["policies"]["SE10"]["event_biased_probability"] == 0.75
    assert config["policies"]["P25"]["strategy"] == "patch_aligned_isolated"
    assert config["policies"]["PE10"]["minimum_visible_tokens_between_masks"] == 1
    assert (
        config["conditional_next_stage"]["branch_pretext_pass_geometry_fail"][
            "vicreg_global_weight"
        ]
        == 1.0
    )
    assert config["conditional_next_stage"]["branch_pretext_fail"]["requirements"] == [
        "target_must_be_phase_invariant",
        "target_must_not_be_computed_from_zero_padded_mask_edges",
        "zero_and_constant_feature_baselines_required",
        "one_example_and_one_batch_overfit_required",
    ]


def test_mask_batch_has_256_tokens_and_trivial_controls() -> None:
    config = load_config(CONFIG_PATH)
    signal = torch.sin(torch.arange(4096, dtype=torch.float32) / 20.0).reshape(1, 1, -1)
    event = torch.zeros(1, 4096, dtype=torch.bool)
    event[:, 1500:2500] = True
    target, tokens, hidden = build_mask_batch(signal, event, config, seed=3)
    assert target.shape == (1, 4096)
    assert tokens.shape == (1, 256)
    assert hidden.shape == (1, 4096)
    interpolation = interpolation_baseline(signal, hidden)
    nearest = nearest_baseline(signal, hidden)
    visible_mean = visible_mean_baseline(signal, hidden)
    assert interpolation.shape == signal.shape
    assert nearest.shape == signal.shape
    assert visible_mean.shape == signal.shape
    assert torch.isfinite(interpolation).all()


def test_reconstruction_components_separate_event_and_background() -> None:
    target = torch.tensor([[[1.0, -1.0, 2.0, -2.0]]])
    prediction = torch.zeros_like(target)
    target_mask = torch.tensor([[True, True, True, False]])
    event_mask = torch.tensor([[True, True, False, False]])

    components = reconstruction_error_components(
        prediction, target, target_mask, event_mask
    )

    assert components == {
        "squared_error_sum": 6.0,
        "event_squared_error_sum": 2.0,
        "background_squared_error_sum": 4.0,
        "prediction_squared_sum": 0.0,
        "target_squared_sum": 6.0,
        "target_count": 3,
        "event_target_count": 2,
        "background_target_count": 1,
    }


def test_visible_mean_baseline_excludes_hidden_samples() -> None:
    signals = torch.tensor([[[1.0, 100.0, 3.0, 5.0]]])
    hidden = torch.tensor([[False, True, False, False]])
    prediction = visible_mean_baseline(signals, hidden)
    assert torch.allclose(prediction, torch.full_like(signals, 3.0))


def test_fixed_mask_diagnostic_can_overfit_tiny_known_signal() -> None:
    config = load_config(CONFIG_PATH)
    config["data"]["sampling_frequency_hz"] = 1_000
    config["masking"].update(
        {"mask_ratio": 0.25, "min_block_ms": 4, "max_block_ms": 8, "guard_ms": 0}
    )
    config["model"].update(
        {"patch_size": 4, "patch_stride": 4, "max_tokens": 16}
    )
    signal = torch.sin(torch.arange(64, dtype=torch.float32) * 0.2).reshape(1, 1, -1)
    event = torch.zeros(1, 64, dtype=torch.bool)
    event[:, 8:56] = True
    model = YeastStudyModel(
        YeastStudyModelConfig(
            input_length=64,
            patch_size=4,
            patch_stride=4,
            d_model=16,
            n_heads=4,
            n_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            max_tokens=16,
        )
    )

    result, prediction, target_mask = run_fixed_mask_overfit(
        model,
        signal,
        event,
        config,
        seed=7,
        steps=120,
        learning_rate=0.01,
        device=torch.device("cpu"),
        log_every=40,
    )

    assert result["final_masked_mse"] < result["zero_masked_mse"] * 0.2
    assert result["gates"]["reduces_zero_error_by_0p80"] is True
    assert prediction.shape == signal.shape
    assert target_mask.shape == event.shape


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
    assert metrics["mean_absolute_off_diagonal_covariance"] == 0.0
