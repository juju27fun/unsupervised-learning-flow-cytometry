from __future__ import annotations

from p3_ssl.mask_collapse_reporting import (
    compare_mask_collapse_runs,
    summarize_mask_collapse_run,
)


def _metrics(cell: str, *, rank: float, cosine: float, model_mse: float) -> dict:
    history = [
        {"loss": 1.0, "time_reconstruction": 1.0, "vicreg": 0.4 if cell == "C1" else 0.0},
        {"loss": 0.4, "time_reconstruction": 0.2, "vicreg": 0.2 if cell == "C1" else 0.0},
    ]
    return {
        "cell": cell,
        "seed": 42,
        "profile": "full",
        "history": history,
        "validation_reconstruction_controls": {
            "real": {
                "model_masked_mse": model_mse,
                "zero_masked_mse": 1.0,
                "interpolation_masked_mse": 0.3,
                "model_output_rms_on_mask": 0.8,
                "target_rms_on_mask": 1.0,
                "target_event_fraction": 0.5,
            }
        },
        "validation_embedding_health": {
            "real": {
                "effective_rank": rank,
                "mean_off_diagonal_cosine_similarity": cosine,
                "mean_dimension_std": 0.5,
                "active_dimensions_std_gt_1e_3": 96,
            }
        },
    }


def _config() -> dict:
    return {
        "promotion_gates": {
            "pretext": {"output_rms_fraction_of_target_min": 0.1},
            "geometry": {"effective_rank_min": 8.0, "mean_pairwise_cosine_max": 0.95},
            "comparison": {"require_geometry_improvement_over_c0": True},
        },
        "decision": {
            "success_action": "run_development_utility_evaluation",
            "failure_action": "run_preregistered_phase_invariant_target_contrast",
        },
    }


def test_anti_collapse_promotes_only_c1_that_beats_strong_baseline_and_geometry() -> None:
    c0 = summarize_mask_collapse_run(
        "C0",
        _metrics("C0", rank=3.0, cosine=0.99, model_mse=0.2),
        _config()["promotion_gates"],
    )
    c1 = summarize_mask_collapse_run(
        "C1",
        _metrics("C1", rank=12.0, cosine=0.80, model_mse=0.2),
        _config()["promotion_gates"],
    )
    comparison = compare_mask_collapse_runs(c0, c1, _config())
    assert c1["gates"]["beats_interpolation"] is True
    assert comparison["eligible_for_utility_evaluation"] is True
    assert comparison["decision"] == "run_development_utility_evaluation"


def test_anti_collapse_rejects_geometry_gain_that_still_fails_absolute_gate() -> None:
    c0 = summarize_mask_collapse_run(
        "C0",
        _metrics("C0", rank=2.0, cosine=0.999, model_mse=0.2),
        _config()["promotion_gates"],
    )
    c1 = summarize_mask_collapse_run(
        "C1",
        _metrics("C1", rank=6.0, cosine=0.97, model_mse=0.2),
        _config()["promotion_gates"],
    )
    comparison = compare_mask_collapse_runs(c0, c1, _config())
    assert comparison["geometry_change_c1_minus_c0"]["improves_both_metrics"] is True
    assert comparison["eligible_for_utility_evaluation"] is False
    assert comparison["decision"] == "run_preregistered_phase_invariant_target_contrast"


def test_anti_collapse_requires_beating_interpolation() -> None:
    c1 = summarize_mask_collapse_run(
        "C1",
        _metrics("C1", rank=12.0, cosine=0.8, model_mse=0.4),
        _config()["promotion_gates"],
    )
    assert c1["gates"]["beats_zero"] is True
    assert c1["gates"]["beats_interpolation"] is False
    assert c1["passes_absolute_pretext_and_geometry_gates"] is False
