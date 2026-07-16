from __future__ import annotations

from p3_ssl.mask_ablation_reporting import summarize_mask_ablation_run


def test_mask_ablation_summary_requires_pretext_and_geometry() -> None:
    metrics = {
        "seed": 42,
        "profile": "full",
        "history": [{"loss": 1.0}, {"loss": 0.4}],
        "validation_reconstruction_controls": {
            "real": {
                "model_masked_mse": 0.4,
                "zero_masked_mse": 1.0,
                "interpolation_masked_mse": 0.2,
                "model_output_rms_on_mask": 0.8,
                "target_rms_on_mask": 1.0,
                "target_event_fraction": 0.5,
            }
        },
        "validation_embedding_health": {
            "real": {
                "effective_rank": 12.0,
                "mean_off_diagonal_cosine_similarity": 0.99,
            }
        },
    }
    gates = {
        "pretext": {"output_rms_fraction_of_target_min": 0.10},
        "geometry": {"effective_rank_min": 8.0, "mean_pairwise_cosine_max": 0.95},
    }
    summary = summarize_mask_ablation_run("P10", metrics, gates)
    assert summary["relative_improvement_vs_zero"] == 0.6
    assert summary["gates"]["beats_zero"] is True
    assert summary["gates"]["mean_pairwise_cosine"] is False
    assert summary["eligible_for_utility_evaluation"] is False
