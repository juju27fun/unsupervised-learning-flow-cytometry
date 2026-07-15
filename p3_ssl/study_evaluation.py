from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_squared_error, r2_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .study_baselines import (
    LOGISTIC_MAX_ITER,
    BaselineData,
    fit_linear_probe,
    fit_logistic_with_diagnostics,
    prediction_metrics,
)
from .study_data import CONTINUOUS_FACTORS, FACTOR_RANGES


NUISANCE_FACTORS = (
    "phase_rad",
    "event_position_fraction",
    "snr_db",
    "target_rms",
    "baseline_drift",
    "sensor_response",
)


def real_variability_summary(data: BaselineData) -> dict[str, Any]:
    signals = np.asarray(data.signals[data.train_indices], dtype=np.float32)
    rows = [data.rows[int(index)] for index in data.train_indices]
    means = signals.mean(axis=1)
    rms = np.sqrt(np.square(signals).mean(axis=1))
    background_ratios = []
    for signal, row in zip(signals, rows):
        start = max(0, min(signal.size, int(float(row.get("event_start_input_index", 0)))))
        end = max(0, min(signal.size, int(float(row.get("event_end_input_index", signal.size)))))
        mask = np.ones(signal.size, dtype=bool)
        mask[start:end] = False
        if mask.any():
            background_ratios.append(float(signal[mask].std() / max(signal.std(), 1.0e-8)))

    def quantiles(values: np.ndarray | list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        if not array.size:
            return {}
        result = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
        return dict(zip(("p05", "p25", "p50", "p75", "p95"), result.astype(float).tolist()))

    return {
        "split": "development_train",
        "n_events": int(len(signals)),
        "window_mean_quantiles": quantiles(means),
        "window_rms_quantiles": quantiles(rms),
        "background_to_total_std_ratio_quantiles": quantiles(background_ratios),
        "interpretation": (
            "offsets are bounded by the observed window-mean IQR; injected noise is well below "
            "the observed background-to-total variability and remains a sensitivity diagnostic"
        ),
    }


def calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.size:
        raise ValueError("Probabilities must have shape (n_samples, n_classes)")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6):
        raise ValueError("Class probabilities must sum to one")
    selected = np.clip(probabilities[np.arange(labels.size), labels], 1.0e-12, 1.0)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if mask.any():
            ece += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {
        "negative_log_likelihood": float(-np.log(selected).mean()),
        "multiclass_brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "expected_calibration_error": float(ece),
        "mean_confidence": float(confidence.mean()),
    }


def _interval(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "percentile_95": np.quantile(array, [0.025, 0.975]).astype(float).tolist(),
        "replicates": array.astype(float).tolist(),
    }


def grouped_bootstrap_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    class_count: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    groups = np.asarray(groups, dtype=str)
    unique_groups = np.unique(groups)
    if repeats <= 0 or unique_groups.size < 2:
        return {
            "status": "not_run",
            "reason": "at least two groups and one repeat are required",
            "n_groups": int(unique_groups.size),
        }
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    class_ids = np.arange(class_count)
    for _ in range(repeats):
        sampled = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([indices_by_group[group] for group in sampled])
        sample_labels = labels[indices]
        sample_predictions = predictions[indices]
        sample_probabilities = probabilities[indices]
        values["macro_f1"].append(
            float(
                f1_score(
                    sample_labels,
                    sample_predictions,
                    labels=class_ids,
                    average="macro",
                    zero_division=0,
                )
            )
        )
        recalls = recall_score(
            sample_labels,
            sample_predictions,
            labels=class_ids,
            average=None,
            zero_division=0,
        )
        present = np.unique(sample_labels)
        values["balanced_accuracy"].append(float(recalls[present].mean()))
        calibration = calibration_metrics(sample_labels, sample_probabilities)
        values["expected_calibration_error"].append(calibration["expected_calibration_error"])
        values["multiclass_brier"].append(calibration["multiclass_brier"])
    return {
        "status": "ok",
        "group_unit": "capture_block_id with record_id fallback",
        "n_groups": int(unique_groups.size),
        "n_repeats": repeats,
        "metrics": {name: _interval(metric_values) for name, metric_values in values.items()},
    }


def _subgroup_metric(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    mask: np.ndarray,
) -> dict[str, Any]:
    present = sorted(set(labels[mask].tolist()))
    return {
        **prediction_metrics(labels[mask], predictions[mask], class_names),
        "present_classes": [class_names[index] for index in present],
        "missing_classes": [name for index, name in enumerate(class_names) if index not in present],
    }


def subgroup_metrics(
    data: BaselineData,
    predictions: np.ndarray,
) -> dict[str, Any]:
    train_rows = [data.rows[int(index)] for index in data.train_indices]
    validation_rows = [data.rows[int(index)] for index in data.validation_indices]
    labels = data.labels[data.validation_indices]
    width_train = np.asarray([float(row.get("width_ms", "nan")) for row in train_rows])
    snr_train = np.asarray([float(row.get("snr_proxy", "nan")) for row in train_rows])
    width_edges = np.nanquantile(width_train, [1.0 / 3.0, 2.0 / 3.0])
    snr_edges = np.nanquantile(snr_train, [1.0 / 3.0, 2.0 / 3.0])

    strata: dict[str, list[str]] = {
        "quality": [row.get("quality", "unknown") for row in validation_rows],
        "duration_tertile": [],
        "snr_tertile": [],
        "crop_edge_status": [],
    }
    names = ("low", "middle", "high")
    for row in validation_rows:
        width = float(row.get("width_ms", "nan"))
        snr = float(row.get("snr_proxy", "nan"))
        strata["duration_tertile"].append(names[int(np.searchsorted(width_edges, width, side="right"))])
        strata["snr_tertile"].append(names[int(np.searchsorted(snr_edges, snr, side="right"))])
        padded = int(row.get("crop_8192_pad_left", 0)) + int(row.get("crop_8192_pad_right", 0))
        strata["crop_edge_status"].append("clamped_edge" if padded else "centered_unclamped")

    result: dict[str, Any] = {
        "threshold_source": "development_train only",
        "duration_tertile_edges_ms": width_edges.astype(float).tolist(),
        "snr_tertile_edges": snr_edges.astype(float).tolist(),
        "strata": {},
    }
    for axis, values in strata.items():
        value_array = np.asarray(values)
        result["strata"][axis] = {
            value: _subgroup_metric(labels, predictions, data.class_names, value_array == value)
            for value in sorted(set(values))
        }
    return result


def evaluate_linear_probe(
    features: np.ndarray,
    data: BaselineData,
    *,
    fraction: float,
    seed: int,
    bootstrap_repeats: int,
    calibration_bins: int = 10,
) -> tuple[dict[str, Any], Any]:
    model, train = fit_linear_probe(features, data, fraction=fraction, seed=seed)
    validation = data.validation_indices
    probabilities = model.predict_proba(features[validation])
    predictions = probabilities.argmax(axis=1)
    rows = [data.rows[int(index)] for index in validation]
    groups = np.asarray(
        [row.get("capture_block_id") or row["record_id"] for row in rows], dtype=str
    )
    metrics = {
        **prediction_metrics(data.labels[validation], predictions, data.class_names),
        "n_probe_events": int(train.size),
        "n_probe_records": len({data.rows[int(index)]["record_id"] for index in train}),
        "probe_optimization": model.probe_optimization_,
        "calibration": calibration_metrics(
            data.labels[validation], probabilities, n_bins=calibration_bins
        ),
        "grouped_bootstrap": grouped_bootstrap_metrics(
            data.labels[validation],
            predictions,
            probabilities,
            groups,
            class_count=len(data.class_names),
            repeats=bootstrap_repeats,
            seed=seed,
        ),
        "subgroups": subgroup_metrics(data, predictions),
    }
    return metrics, model


def cross_recording_retrieval(
    embeddings: np.ndarray,
    rows: list[dict[str, str]],
    labels: np.ndarray,
    *,
    neighbors: int = 5,
) -> dict[str, Any]:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1.0e-12)
    similarities = normalized @ normalized.T
    records = np.asarray([row["record_id"] for row in rows])
    acquisitions = np.asarray([row.get("acquisition_id", "unknown") for row in rows])
    qualities = np.asarray([row.get("quality", "unknown") for row in rows])
    metric_rows = []
    for query in range(len(rows)):
        eligible = np.flatnonzero(records != records[query])
        if not eligible.size:
            continue
        order = eligible[np.argsort(-similarities[query, eligible], kind="stable")]
        selected = order[: min(neighbors, order.size)]
        metric_rows.append(
            {
                "label": labels[selected] == labels[query],
                "acquisition": acquisitions[selected] == acquisitions[query],
                "quality": qualities[selected] == qualities[query],
            }
        )
    if not metric_rows:
        return {"status": "not_run", "reason": "no cross-recording neighbors"}
    return {
        "status": "ok",
        "n_queries": len(metric_rows),
        "neighbors": neighbors,
        "same_record_neighbors": 0,
        "top1_label_purity": float(np.mean([row["label"][0] for row in metric_rows])),
        "topk_label_purity": float(np.mean([row["label"].mean() for row in metric_rows])),
        "top1_acquisition_purity": float(
            np.mean([row["acquisition"][0] for row in metric_rows])
        ),
        "topk_acquisition_purity": float(
            np.mean([row["acquisition"].mean() for row in metric_rows])
        ),
        "top1_quality_purity": float(np.mean([row["quality"][0] for row in metric_rows])),
        "topk_quality_purity": float(np.mean([row["quality"].mean() for row in metric_rows])),
        "interpretation_warning": (
            "source-group labels are acquisition-condition proxies; label purity is not morphology purity"
        ),
    }


def label_efficiency_auc(results: list[dict[str, Any]], method_key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[(str(row[method_key]), int(row["seed"]))].append(row)
    output = []
    for (method, seed), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: float(row["label_fraction"]))
        fractions = np.asarray([float(row["label_fraction"]) for row in ordered])
        scores = np.asarray([float(row["macro_f1"]) for row in ordered])
        if np.unique(fractions).size < 2:
            continue
        area = float(np.trapezoid(scores, fractions))
        output.append(
            {
                method_key: method,
                "seed": seed,
                "fractions": fractions.tolist(),
                "macro_f1": scores.tolist(),
                "area": area,
                "normalized_area": area / float(fractions[-1] - fractions[0]),
            }
        )
    return output


def perturb_signals(
    signals: np.ndarray,
    perturbation: dict[str, Any],
    *,
    seed: int,
) -> np.ndarray:
    result = np.asarray(signals, dtype=np.float32).copy()
    kind = str(perturbation["kind"])
    value = float(perturbation["value"])
    if kind == "gain":
        result *= value
    elif kind == "offset":
        result += value
    elif kind == "shift_samples":
        result = np.roll(result, int(value), axis=1)
    elif kind == "noise_fraction_signal_std":
        scale = result.std(axis=1, keepdims=True) * value
        result += np.random.default_rng(seed).normal(size=result.shape).astype(np.float32) * scale
    elif kind == "center_mask_samples":
        width = min(int(value), result.shape[1])
        start = result.shape[1] // 2 - width // 2
        result[:, start : start + width] = 0.0
    else:
        raise ValueError(f"Unsupported perturbation kind: {kind}")
    return result


def robustness_metrics(
    base_embeddings: np.ndarray,
    perturbed_embeddings: np.ndarray,
    base_probabilities: np.ndarray,
    perturbed_probabilities: np.ndarray,
) -> dict[str, float]:
    base = np.asarray(base_embeddings, dtype=np.float64)
    perturbed = np.asarray(perturbed_embeddings, dtype=np.float64)
    base /= np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1.0e-12)
    perturbed /= np.maximum(np.linalg.norm(perturbed, axis=1, keepdims=True), 1.0e-12)
    cosine_distance = 1.0 - np.sum(base * perturbed, axis=1)
    base_predictions = np.argmax(base_probabilities, axis=1)
    perturbed_predictions = np.argmax(perturbed_probabilities, axis=1)
    return {
        "embedding_cosine_distance_mean": float(cosine_distance.mean()),
        "embedding_cosine_distance_std": float(cosine_distance.std()),
        "prediction_agreement": float(np.mean(base_predictions == perturbed_predictions)),
        "probability_l1_mean": float(
            np.abs(base_probabilities - perturbed_probabilities).sum(axis=1).mean()
        ),
    }


def physical_embedding_diagnostics(
    train_embeddings: np.ndarray,
    validation_embeddings: np.ndarray,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    *,
    neighbors: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    if len(train_embeddings) != len(train_rows) or len(validation_embeddings) != len(validation_rows):
        raise ValueError("Simulation embeddings and metadata rows must align")

    retained: dict[str, Any] = {}
    for factor in CONTINUOUS_FACTORS:
        conditional = factor in {
            "component_separation_ms",
            "relative_component_amplitude",
            "frequency_separation_khz",
        }
        train_mask = np.asarray(
            [not conditional or int(row["component_count"]) == 2 for row in train_rows]
        )
        validation_mask = np.asarray(
            [not conditional or int(row["component_count"]) == 2 for row in validation_rows]
        )
        y_train = np.asarray([float(row[factor]) for row in train_rows])[train_mask]
        y_validation = np.asarray([float(row[factor]) for row in validation_rows])[validation_mask]
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train_embeddings[train_mask], y_train)
        prediction = model.predict(validation_embeddings[validation_mask])
        low, high = FACTOR_RANGES[factor]
        mse = float(mean_squared_error(y_validation, prediction) / ((high - low) ** 2))
        constant = np.full_like(y_validation, y_train.mean())
        prior_mse = float(mean_squared_error(y_validation, constant) / ((high - low) ** 2))
        retained[factor] = {
            "n_train": int(train_mask.sum()),
            "n_validation": int(validation_mask.sum()),
            "normalized_mse": mse,
            "constant_prior_normalized_mse": prior_mse,
            "relative_mse_reduction_vs_constant": 1.0 - mse / max(prior_mse, 1.0e-12),
            "r2": float(r2_score(y_validation, prediction)),
            "spearman_rho": float(spearmanr(y_validation, prediction).statistic),
        }

    count_train = np.asarray([int(row["component_count"]) for row in train_rows])
    count_validation = np.asarray([int(row["component_count"]) for row in validation_rows])
    count_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=LOGISTIC_MAX_ITER,
            random_state=seed,
        ),
    )
    count_optimization = fit_logistic_with_diagnostics(
        count_model, train_embeddings, count_train
    )
    count_prediction = count_model.predict(validation_embeddings)
    majority = np.full_like(count_validation, np.bincount(count_train).argmax())

    nuisance: dict[str, Any] = {}
    for factor in NUISANCE_FACTORS:
        y_train = np.asarray([float(row[factor]) for row in train_rows])
        y_validation = np.asarray([float(row[factor]) for row in validation_rows])
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train_embeddings, y_train)
        prediction = model.predict(validation_embeddings)
        nuisance[factor] = {
            "r2": float(r2_score(y_validation, prediction)),
            "spearman_rho": float(spearmanr(y_validation, prediction).statistic),
            "interpretation": "lower recoverability is preferred only when retained-factor information is preserved",
        }

    normalized = np.asarray(validation_embeddings, dtype=np.float64)
    normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1.0e-12)
    similarities = normalized @ normalized.T
    latent_ids = np.asarray([row["latent_id"] for row in validation_rows])
    rng = np.random.default_rng(seed)
    neighborhood: dict[str, list[float]] = {factor: [] for factor in CONTINUOUS_FACTORS}
    random_distances: dict[str, list[float]] = {factor: [] for factor in CONTINUOUS_FACTORS}
    count_agreement = []
    random_count_agreement = []
    for query, row in enumerate(validation_rows):
        eligible = np.flatnonzero(latent_ids != latent_ids[query])
        if not eligible.size:
            continue
        nearest = eligible[np.argsort(-similarities[query, eligible], kind="stable")[:neighbors]]
        random_neighbors = rng.choice(eligible, size=min(neighbors, eligible.size), replace=False)
        query_count = int(row["component_count"])
        count_agreement.extend(int(validation_rows[index]["component_count"]) == query_count for index in nearest)
        random_count_agreement.extend(
            int(validation_rows[index]["component_count"]) == query_count for index in random_neighbors
        )
        for factor in CONTINUOUS_FACTORS:
            conditional = factor in {
                "component_separation_ms",
                "relative_component_amplitude",
                "frequency_separation_khz",
            }
            if conditional and query_count != 2:
                continue
            low, high = FACTOR_RANGES[factor]
            query_value = float(row[factor])
            for target, destination in ((nearest, neighborhood), (random_neighbors, random_distances)):
                for index in target:
                    if conditional and int(validation_rows[index]["component_count"]) != 2:
                        continue
                    destination[factor].append(
                        abs(query_value - float(validation_rows[index][factor])) / (high - low)
                    )
    continuity = {}
    for factor in CONTINUOUS_FACTORS:
        observed = float(np.mean(neighborhood[factor]))
        random_value = float(np.mean(random_distances[factor]))
        continuity[factor] = {
            "cross_latent_neighbor_normalized_absolute_difference": observed,
            "random_normalized_absolute_difference": random_value,
            "relative_reduction_vs_random": 1.0 - observed / max(random_value, 1.0e-12),
        }
    return {
        "retained_factor_linear_probes": retained,
        "component_count_probe": {
            "balanced_accuracy": float(balanced_accuracy_score(count_validation, count_prediction)),
            "majority_balanced_accuracy": float(
                balanced_accuracy_score(count_validation, majority)
            ),
            "optimization": count_optimization,
        },
        "nuisance_leakage_linear_probes": nuisance,
        "cross_latent_neighborhood_continuity": {
            "neighbors": neighbors,
            "continuous_factors": continuity,
            "component_count_agreement": float(np.mean(count_agreement)),
            "random_component_count_agreement": float(np.mean(random_count_agreement)),
        },
        "scope": "development simulation validation only; heldout generator test remains sealed",
    }
