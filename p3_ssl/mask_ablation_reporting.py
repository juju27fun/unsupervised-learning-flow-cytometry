from __future__ import annotations

from typing import Any


def summarize_mask_ablation_run(
    policy: str,
    metrics: dict[str, Any],
    promotion_gates: dict[str, Any],
) -> dict[str, Any]:
    reconstruction = metrics["validation_reconstruction_controls"]["real"]
    health = metrics["validation_embedding_health"]["real"]
    history = metrics["history"]
    zero = float(reconstruction["zero_masked_mse"])
    model = float(reconstruction["model_masked_mse"])
    rms_fraction = float(reconstruction["model_output_rms_on_mask"]) / float(
        reconstruction["target_rms_on_mask"]
    )
    rank = float(health["effective_rank"])
    cosine = float(health["mean_off_diagonal_cosine_similarity"])
    pretext = promotion_gates["pretext"]
    geometry = promotion_gates["geometry"]
    gates = {
        "beats_zero": model < zero,
        "nontrivial_amplitude": rms_fraction
        >= float(pretext["output_rms_fraction_of_target_min"]),
        "effective_rank": rank >= float(geometry["effective_rank_min"]),
        "mean_pairwise_cosine": cosine <= float(geometry["mean_pairwise_cosine_max"]),
    }
    return {
        "policy": policy,
        "seed": int(metrics["seed"]),
        "profile": metrics["profile"],
        "first_train_loss": float(history[0]["loss"]),
        "final_train_loss": float(history[-1]["loss"]),
        "minimum_train_loss": min(float(row["loss"]) for row in history),
        "model_masked_mse": model,
        "zero_masked_mse": zero,
        "interpolation_masked_mse": float(reconstruction["interpolation_masked_mse"]),
        "relative_improvement_vs_zero": (zero - model) / zero,
        "output_rms_fraction_of_target": rms_fraction,
        "target_event_fraction": float(reconstruction["target_event_fraction"]),
        "effective_rank": rank,
        "mean_pairwise_cosine": cosine,
        "gates": gates,
        "eligible_for_utility_evaluation": all(gates.values()),
    }
