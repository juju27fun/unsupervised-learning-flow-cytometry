from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score


def fit_train_only_pca(
    train_signals: np.ndarray,
    all_signals: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    model = PCA(
        n_components=int(config["components"]),
        svd_solver=str(config["svd_solver"]),
        random_state=int(config["random_state"]),
        whiten=bool(config["whiten"]),
    )
    model.fit(np.asarray(train_signals, dtype=np.float32))
    transformed = model.transform(np.asarray(all_signals, dtype=np.float32)).astype(np.float32)
    return transformed, {
        "components": int(config["components"]),
        "fit_split": str(config["fit_split"]),
        "svd_solver": str(config["svd_solver"]),
        "random_state": int(config["random_state"]),
        "whiten": bool(config["whiten"]),
        "explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum()),
        "fit_mean_rms": float(np.sqrt(np.mean(np.square(model.mean_)))),
    }


def paired_group_macro_f1_difference(
    labels: np.ndarray,
    groups: np.ndarray,
    candidate_predictions: list[np.ndarray],
    baseline_predictions: list[np.ndarray],
    *,
    class_count: int,
    repeats: int,
    seed: int,
    interval_level: float,
) -> dict[str, Any]:
    if len(candidate_predictions) != len(baseline_predictions) or not candidate_predictions:
        raise ValueError("Candidate and baseline predictions must have matched non-empty seeds")
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=str)
    unique_groups = np.unique(groups)
    if not unique_groups.size:
        raise ValueError("At least one bootstrap group is required")
    class_ids = np.arange(class_count)

    def score(predictions: np.ndarray, indices: np.ndarray) -> float:
        return float(
            f1_score(
                labels[indices],
                np.asarray(predictions)[indices],
                labels=class_ids,
                average="macro",
                zero_division=0,
            )
        )

    full = np.arange(len(labels))
    candidate_score = float(np.mean([score(values, full) for values in candidate_predictions]))
    baseline_score = float(np.mean([score(values, full) for values in baseline_predictions]))
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    for repeat in range(repeats):
        sampled = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([group_indices[group] for group in sampled])
        candidate = np.mean([score(values, indices) for values in candidate_predictions])
        baseline = np.mean([score(values, indices) for values in baseline_predictions])
        differences[repeat] = candidate - baseline
    tail = (1.0 - interval_level) / 2.0
    return {
        "candidate_mean_macro_f1": candidate_score,
        "baseline_mean_macro_f1": baseline_score,
        "gain": candidate_score - baseline_score,
        "paired_interval_level": interval_level,
        "paired_interval": [
            float(np.quantile(differences, tail)),
            float(np.quantile(differences, 1.0 - tail)),
        ],
        "bootstrap_repeats": repeats,
        "n_groups": int(unique_groups.size),
    }


def apply_utility_gate(
    comparisons: dict[str, dict[str, Any]],
    config: dict[str, Any],
    *,
    scientific_decision_allowed: bool,
) -> dict[str, Any]:
    strongest = max(
        comparisons,
        key=lambda name: float(comparisons[name]["baseline_mean_macro_f1"]),
    )
    primary = comparisons[strongest]
    gate = config["primary_gate"]
    checks = {
        "minimum_gain_vs_strongest": float(primary["gain"]) >= float(gate["minimum_gain"]),
        "paired_interval_lower_above_zero": float(primary["paired_interval"][0]) > 0.0,
        "positive_gain_vs_each_baseline": all(
            float(comparison["gain"]) > 0.0 for comparison in comparisons.values()
        ),
        "full_profile": scientific_decision_allowed,
    }
    passed = all(checks.values())
    return {
        "strongest_baseline": strongest,
        "comparisons": comparisons,
        "checks": checks,
        "passed": passed,
        "decision": (
            gate["success_action"]
            if passed
            else gate["failure_action"]
            if scientific_decision_allowed
            else "smoke_only_no_scientific_decision"
        ),
    }
