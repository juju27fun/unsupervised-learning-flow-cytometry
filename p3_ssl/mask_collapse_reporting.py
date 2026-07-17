from __future__ import annotations

from typing import Any


def summarize_mask_collapse_run(
    cell: str,
    metrics: dict[str, Any],
    promotion_gates: dict[str, Any],
) -> dict[str, Any]:
    reconstruction = metrics["validation_reconstruction_controls"]["real"]
    health = metrics["validation_embedding_health"]["real"]
    history = metrics["history"]
    model_mse = float(reconstruction["model_masked_mse"])
    zero_mse = float(reconstruction["zero_masked_mse"])
    interpolation_mse = float(reconstruction["interpolation_masked_mse"])
    target_rms = float(reconstruction["target_rms_on_mask"])
    rms_fraction = float(reconstruction["model_output_rms_on_mask"]) / target_rms
    effective_rank = float(health["effective_rank"])
    cosine = float(health["mean_off_diagonal_cosine_similarity"])
    pretext = promotion_gates["pretext"]
    geometry = promotion_gates["geometry"]
    gates = {
        "beats_zero": model_mse < zero_mse,
        "beats_interpolation": model_mse < interpolation_mse,
        "nontrivial_amplitude": rms_fraction
        >= float(pretext["output_rms_fraction_of_target_min"]),
        "effective_rank": effective_rank >= float(geometry["effective_rank_min"]),
        "mean_pairwise_cosine": cosine <= float(geometry["mean_pairwise_cosine_max"]),
    }
    first = history[0]
    final = history[-1]
    return {
        "cell": cell,
        "seed": int(metrics["seed"]),
        "profile": str(metrics["profile"]),
        "first_total_loss": float(first["loss"]),
        "final_total_loss": float(final["loss"]),
        "final_time_reconstruction_loss": float(final["time_reconstruction"]),
        "final_vicreg_loss": float(final.get("vicreg", 0.0)),
        "loss_reduction_fraction": 1.0 - float(final["loss"]) / float(first["loss"]),
        "model_masked_mse": model_mse,
        "zero_masked_mse": zero_mse,
        "interpolation_masked_mse": interpolation_mse,
        "relative_improvement_vs_zero": (zero_mse - model_mse) / zero_mse,
        "relative_improvement_vs_interpolation": (
            interpolation_mse - model_mse
        ) / interpolation_mse,
        "output_rms_fraction_of_target": rms_fraction,
        "target_event_fraction": float(reconstruction["target_event_fraction"]),
        "effective_rank": effective_rank,
        "mean_pairwise_cosine": cosine,
        "mean_dimension_std": float(health["mean_dimension_std"]),
        "active_dimensions": int(health["active_dimensions_std_gt_1e_3"]),
        "gates": gates,
        "passes_absolute_pretext_and_geometry_gates": all(gates.values()),
    }


def compare_mask_collapse_runs(
    c0: dict[str, Any],
    c1: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if c0["cell"] != "C0" or c1["cell"] != "C1":
        raise ValueError("Mask-collapse comparison requires C0 and C1 in that order")
    for field in ("seed", "profile"):
        if c0[field] != c1[field]:
            raise ValueError(f"C0/C1 {field} mismatch")

    rank_delta = float(c1["effective_rank"]) - float(c0["effective_rank"])
    cosine_delta = float(c1["mean_pairwise_cosine"]) - float(
        c0["mean_pairwise_cosine"]
    )
    geometry_improves = rank_delta > 0.0 and cosine_delta < 0.0
    require_relative = bool(
        config["promotion_gates"]["comparison"][
            "require_geometry_improvement_over_c0"
        ]
    )
    c1_passes = bool(c1["passes_absolute_pretext_and_geometry_gates"])
    promoted = c1_passes and (geometry_improves or not require_relative)
    return {
        "rows": [c0, c1],
        "geometry_change_c1_minus_c0": {
            "effective_rank": rank_delta,
            "mean_pairwise_cosine": cosine_delta,
            "improves_both_metrics": geometry_improves,
        },
        "selected_cell": "C1" if promoted else None,
        "eligible_for_utility_evaluation": promoted,
        "decision": (
            config["decision"]["success_action"]
            if promoted
            else config["decision"]["failure_action"]
        ),
    }
