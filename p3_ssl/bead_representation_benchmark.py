from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from particles2snr.particle_class_coverage import (
    FEATURE_NAMES as CLASSICAL_FEATURE_NAMES,
    relative_descriptors,
)

from .bead_ssl import make_model, seed_everything
from .decimation import normalize_signal


SIMULATION_TARGETS = ("duration_ms", "doppler_khz")
SIMULATION_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 1.0)
REAL_FRACTIONS = (0.25, 0.50, 0.75, 1.0)
RIDGE_ALPHAS = tuple(float(value) for value in np.logspace(-4, 4, 9))
LOGISTIC_CS = tuple(float(value) for value in np.logspace(-4, 4, 9))
REAL_CLASS_NAMES = ("2um", "4um", "10um")


@dataclass(frozen=True)
class BenchmarkPopulation:
    signals: np.ndarray
    ids: np.ndarray
    groups: np.ndarray
    labels: np.ndarray
    metadata: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class FeatureSet:
    method: str
    representation_seed: int | None
    values: np.ndarray
    kind: str = "direct"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalized(signals: np.ndarray, mode: str) -> np.ndarray:
    return np.stack(
        [normalize_signal(np.asarray(row, dtype=np.float32), mode=mode) for row in signals]
    ).astype(np.float32)


def load_simulation_population(
    root: Path,
    *,
    split: str,
    normalization: str = "window_zscore",
    allow_test: bool = False,
) -> BenchmarkPopulation:
    if split == "test" and not allow_test:
        raise PermissionError("Simulation test is sealed")
    rows = [
        row
        for row in _read_csv(root / "simulation_metadata.csv")
        if row["split"] == split and int(row["component_count"]) == 1
    ]
    if not rows:
        raise ValueError(f"No single-component simulation rows for split={split}")
    source = np.load(root / "signals.npy", mmap_mode="r", allow_pickle=False)
    signals = _normalized(
        np.stack([np.asarray(source[int(row["signal_row"])]) for row in rows]),
        normalization,
    )
    labels = np.asarray(
        [[float(row[name]) for name in SIMULATION_TARGETS] for row in rows],
        dtype=np.float64,
    )
    return BenchmarkPopulation(
        signals=signals,
        ids=np.asarray(
            [f"{row['latent_id']}:view-{row['view_index']}" for row in rows],
            dtype=str,
        ),
        groups=np.asarray([row["latent_id"] for row in rows], dtype=str),
        labels=labels,
        metadata=tuple(rows),
    )


def load_real_population(
    root: Path,
    *,
    split: str,
    normalization: str = "window_zscore",
    allow_test: bool = False,
) -> BenchmarkPopulation:
    if split == "test" and not allow_test:
        raise PermissionError("Real test is sealed")
    rows = [row for row in _read_csv(root / "events.csv") if row["split"] == split]
    if not rows:
        raise ValueError(f"No real events for split={split}")
    source = np.load(root / "signals.npy", mmap_mode="r", allow_pickle=False)
    signals = _normalized(
        np.stack([np.asarray(source[int(row["signal_row"])]) for row in rows]),
        normalization,
    )
    return BenchmarkPopulation(
        signals=signals,
        ids=np.asarray([row["event_id"] for row in rows], dtype=str),
        groups=np.asarray([row["source_group"] for row in rows], dtype=str),
        labels=np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64),
        metadata=tuple(rows),
    )


def average_simulation_views(
    population: BenchmarkPopulation,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values)
    output_values: list[np.ndarray] = []
    output_targets: list[np.ndarray] = []
    latent_ids: list[str] = []
    for latent_id in sorted(set(population.groups.tolist())):
        indices = np.flatnonzero(population.groups == latent_id)
        if indices.size != 2:
            raise ValueError(f"Expected two views for latent {latent_id}, got {indices.size}")
        targets = population.labels[indices]
        if not np.allclose(targets, targets[0]):
            raise ValueError(f"Retained targets differ across views for {latent_id}")
        output_values.append(values[indices].mean(axis=0))
        output_targets.append(targets[0])
        latent_ids.append(latent_id)
    return (
        np.asarray(output_values),
        np.asarray(output_targets, dtype=np.float64),
        np.asarray(latent_ids, dtype=str),
    )


def load_encoder(
    *,
    config: dict[str, Any],
    seed: int,
    checkpoint: Path | None,
    device: torch.device,
    expected_epoch: int = 20,
) -> torch.nn.Module:
    seed_everything(seed)
    model = make_model(config)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("epoch", -1)) != expected_epoch:
            raise ValueError(
                f"Expected epoch-{expected_epoch} checkpoint: {checkpoint}"
            )
        model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    signals: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.asarray(signals, dtype=np.float32)).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=False,
    )
    rows = []
    for (batch,) in loader:
        rows.append(
            model.global_embedding(batch.to(device), pool="mean").cpu().numpy()
        )
    values = np.concatenate(rows)
    if not np.all(np.isfinite(values)):
        raise ValueError("Embedding extraction produced non-finite values")
    return values


def classical_descriptor_matrix(signals: np.ndarray) -> np.ndarray:
    rows = []
    for signal in signals:
        descriptors = relative_descriptors(signal)
        rows.append([float(descriptors[name]) for name in CLASSICAL_FEATURE_NAMES])
    return np.asarray(rows, dtype=np.float64)


def embedding_health(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    weights = np.square(singular)
    weights /= max(float(weights.sum()), 1.0e-12)
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1.0e-12))))
    normalized = centered / np.maximum(
        np.linalg.norm(centered, axis=1, keepdims=True), 1.0e-12
    )
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~np.eye(cosine.shape[0], dtype=bool)]
    return {
        "effective_rank": float(np.exp(entropy)),
        "between_sample_variance": float(np.mean(np.var(values, axis=0))),
        "mean_off_diagonal_cosine": float(np.mean(off_diagonal)),
    }


def quantile_stratified_order(targets: np.ndarray, *, seed: int) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("Simulation targets must have shape (n, 2)")
    bins = []
    for column in range(2):
        edges = np.unique(np.quantile(targets[:, column], [0.25, 0.5, 0.75]))
        bins.append(np.digitize(targets[:, column], edges, right=True))
    strata = bins[0] * 4 + bins[1]
    rng = np.random.default_rng(seed)
    queues = {
        int(value): rng.permutation(np.flatnonzero(strata == value)).tolist()
        for value in np.unique(strata)
    }
    order: list[int] = []
    active = sorted(queues)
    while active:
        next_active = []
        for value in active:
            if queues[value]:
                order.append(int(queues[value].pop()))
            if queues[value]:
                next_active.append(value)
        active = next_active
    return np.asarray(order, dtype=np.int64)


def nested_simulation_subsets(
    targets: np.ndarray,
    *,
    fractions: Iterable[float] = SIMULATION_FRACTIONS,
    seed: int,
) -> dict[float, np.ndarray]:
    order = quantile_stratified_order(targets, seed=seed)
    output = {}
    for fraction in fractions:
        n_rows = len(order) if fraction >= 1.0 else max(1, int(round(len(order) * fraction)))
        output[float(fraction)] = np.sort(order[:n_rows])
    return output


def nested_real_subsets(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> dict[float, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=str)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    folds = [
        np.sort(test)
        for _, test in splitter.split(np.zeros(labels.size), labels, groups)
    ]
    for fold in folds:
        if set(labels[fold].tolist()) != {0, 1, 2}:
            raise ValueError("Every real subset fold must contain all three classes")
    return {
        fraction: np.sort(np.concatenate(folds[:count]))
        for fraction, count in zip(REAL_FRACTIONS, (1, 2, 3, 4), strict=True)
    }


def _simulation_estimator(kind: str, *, seed: int, n_rows: int) -> GridSearchCV:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if kind == "raw_pca":
        smallest_inner_train = n_rows - math.ceil(n_rows / 5)
        steps.append(
            (
                "pca",
                PCA(
                    n_components=min(64, max(1, smallest_inner_train)),
                    svd_solver="randomized",
                    random_state=seed,
                ),
            )
        )
    steps.append(("probe", Ridge()))
    return GridSearchCV(
        Pipeline(steps),
        {"probe__alpha": RIDGE_ALPHAS},
        scoring="r2",
        cv=KFold(n_splits=5, shuffle=True, random_state=seed),
        n_jobs=1,
        refit=True,
    )


def _validate_real_cv(labels: np.ndarray, groups: np.ndarray, *, seed: int) -> None:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, validation in splitter.split(np.zeros(labels.size), labels, groups):
        if set(labels[train].tolist()) != {0, 1, 2}:
            raise ValueError("A real inner-CV training fold misses a class")
        if set(labels[validation].tolist()) != {0, 1, 2}:
            raise ValueError("A real inner-CV validation fold misses a class")
        if set(groups[train]) & set(groups[validation]):
            raise ValueError("Source group leakage in real inner CV")


def _real_estimator(
    kind: str,
    *,
    seed: int,
    n_rows: int,
    labels: np.ndarray,
    groups: np.ndarray,
) -> GridSearchCV:
    _validate_real_cv(labels, groups, seed=seed)
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if kind == "raw_pca":
        steps.append(
            (
                "pca",
                PCA(
                    n_components=min(64, max(1, n_rows - 1)),
                    svd_solver="randomized",
                    random_state=seed,
                ),
            )
        )
    steps.append(
        (
            "probe",
            LogisticRegression(
                class_weight="balanced",
                max_iter=4000,
                solver="lbfgs",
                random_state=seed,
            ),
        )
    )
    return GridSearchCV(
        Pipeline(steps),
        {"probe__C": LOGISTIC_CS},
        scoring="f1_macro",
        cv=list(
            StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=seed
            ).split(np.zeros(labels.size), labels, groups)
        ),
        n_jobs=1,
        refit=True,
    )


def evaluate_simulation_features(
    train_features: FeatureSet,
    validation_features: np.ndarray,
    train_targets: np.ndarray,
    validation_targets: np.ndarray,
    subsets: dict[float, np.ndarray],
    *,
    subset_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows = []
    prediction_rows = []
    target_iqr = np.maximum(
        np.quantile(validation_targets, 0.75, axis=0)
        - np.quantile(validation_targets, 0.25, axis=0),
        1.0e-12,
    )
    for fraction, indices in subsets.items():
        estimator = _simulation_estimator(
            train_features.kind, seed=subset_seed, n_rows=indices.size
        )
        estimator.fit(train_features.values[indices], train_targets[indices])
        predictions = estimator.predict(validation_features)
        target_r2 = r2_score(
            validation_targets, predictions, multioutput="raw_values"
        )
        target_mae = mean_absolute_error(
            validation_targets, predictions, multioutput="raw_values"
        )
        metric_rows.append(
            {
                "domain": "simulation",
                "method": train_features.method,
                "representation_seed": train_features.representation_seed,
                "subset_seed": subset_seed,
                "fraction": fraction,
                "n_train": int(indices.size),
                "duration_r2": float(target_r2[0]),
                "doppler_r2": float(target_r2[1]),
                "mean_r2": float(np.mean(target_r2)),
                "duration_nmae_iqr": float(target_mae[0] / target_iqr[0]),
                "doppler_nmae_iqr": float(target_mae[1] / target_iqr[1]),
                "best_parameter": float(estimator.best_params_["probe__alpha"]),
            }
        )
        prediction_rows.extend(
            {
                "domain": "simulation",
                "method": train_features.method,
                "representation_seed": train_features.representation_seed,
                "subset_seed": subset_seed,
                "fraction": fraction,
                "row": int(row),
                "target_duration_ms": float(validation_targets[row, 0]),
                "predicted_duration_ms": float(predictions[row, 0]),
                "target_doppler_khz": float(validation_targets[row, 1]),
                "predicted_doppler_khz": float(predictions[row, 1]),
            }
            for row in range(validation_targets.shape[0])
        )
    return metric_rows, prediction_rows


def evaluate_real_features(
    train_features: FeatureSet,
    validation_features: np.ndarray,
    train: BenchmarkPopulation,
    validation: BenchmarkPopulation,
    subsets: dict[float, np.ndarray],
    *,
    subset_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows = []
    prediction_rows = []
    for fraction, indices in subsets.items():
        estimator = _real_estimator(
            train_features.kind,
            seed=subset_seed,
            n_rows=indices.size,
            labels=train.labels[indices],
            groups=train.groups[indices],
        )
        estimator.fit(
            train_features.values[indices],
            train.labels[indices],
        )
        predictions = estimator.predict(validation_features)
        recalls = recall_score(
            validation.labels,
            predictions,
            labels=np.arange(3),
            average=None,
            zero_division=0,
        )
        matrix = confusion_matrix(validation.labels, predictions, labels=np.arange(3))
        metric_rows.append(
            {
                "domain": "real",
                "method": train_features.method,
                "representation_seed": train_features.representation_seed,
                "subset_seed": subset_seed,
                "fraction": fraction,
                "n_train": int(indices.size),
                "n_train_groups": int(np.unique(train.groups[indices]).size),
                "macro_f1": float(
                    f1_score(
                        validation.labels,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(validation.labels, predictions)
                ),
                "recall_2um": float(recalls[0]),
                "recall_4um": float(recalls[1]),
                "recall_10um": float(recalls[2]),
                "confusion_matrix": json.dumps(matrix.tolist()),
                "best_parameter": float(estimator.best_params_["probe__C"]),
            }
        )
        prediction_rows.extend(
            {
                "domain": "real",
                "method": train_features.method,
                "representation_seed": train_features.representation_seed,
                "subset_seed": subset_seed,
                "fraction": fraction,
                "event_id": str(validation.ids[row]),
                "source_group": str(validation.groups[row]),
                "target": int(validation.labels[row]),
                "prediction": int(predictions[row]),
                "is_10um": bool(validation.labels[row] == 2),
            }
            for row in range(validation.labels.size)
        )
    return metric_rows, prediction_rows


def label_efficiency_auc(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | None, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["method"]),
            row.get("representation_seed"),
            int(row["subset_seed"]),
        )
        grouped.setdefault(key, []).append(row)
    shared_full = {
        (str(row["method"]), row.get("representation_seed")): row
        for row in rows
        if np.isclose(float(row["fraction"]), 1.0)
    }
    output = []
    for (method, representation_seed, subset_seed), values in grouped.items():
        if not any(np.isclose(float(item["fraction"]), 1.0) for item in values):
            full = shared_full.get((method, representation_seed))
            if full is None:
                raise ValueError(f"Missing shared full-label endpoint for {method}")
            values = [*values, full]
        ordered = sorted(values, key=lambda item: float(item["fraction"]))
        x = np.log10(np.asarray([float(item["fraction"]) for item in ordered]))
        y = np.asarray([float(item[score_key]) for item in ordered])
        if np.unique(x).size < 2:
            continue
        output.append(
            {
                "method": method,
                "representation_seed": representation_seed,
                "subset_seed": subset_seed,
                "score": score_key,
                "normalized_log_fraction_auc": float(
                    np.trapezoid(y, x) / (x[-1] - x[0])
                ),
            }
        )
    return output


def paired_hierarchical_interval(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    left: str = "P25",
    right: str = "CYCLIC25",
    repeats: int = 10_000,
    seed: int = 20260720,
) -> dict[str, Any]:
    lookup = {
        (
            row["method"],
            int(row["representation_seed"]),
            int(row["subset_seed"]),
            float(row.get("fraction", -1.0)),
        ): float(row[metric])
        for row in rows
        if row.get("representation_seed") is not None
    }
    keys = sorted(
        {
            (seed_value, subset_seed, fraction)
            for method, seed_value, subset_seed, fraction in lookup
            if method == left
            and (right, seed_value, subset_seed, fraction) in lookup
        }
    )
    if not keys:
        raise ValueError(f"No paired {left}/{right} rows for {metric}")
    differences: dict[int, list[float]] = {}
    for seed_value, subset_seed, fraction in keys:
        differences.setdefault(seed_value, []).append(
            lookup[(right, seed_value, subset_seed, fraction)]
            - lookup[(left, seed_value, subset_seed, fraction)]
        )
    rng = np.random.default_rng(seed)
    representation_seeds = np.asarray(sorted(differences))
    bootstrap = []
    for _ in range(repeats):
        sampled_seeds = rng.choice(
            representation_seeds, size=representation_seeds.size, replace=True
        )
        sampled = []
        for representation_seed in sampled_seeds:
            values = np.asarray(differences[int(representation_seed)])
            sampled.extend(rng.choice(values, size=values.size, replace=True))
        bootstrap.append(float(np.mean(sampled)))
    observed_by_seed = {
        str(seed_value): float(np.mean(values))
        for seed_value, values in sorted(differences.items())
    }
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "contrast": f"{right}-{left}",
        "metric": metric,
        "mean_difference": float(np.mean(list(observed_by_seed.values()))),
        "ci95": [float(low), float(high)],
        "differences_by_representation_seed": observed_by_seed,
        "n_independent_training_pairs": len(observed_by_seed),
        "bootstrap_repeats": repeats,
        "warning": "Bootstrap precision does not increase the number of independent training pairs.",
    }


def paired_grouped_classification_interval(
    rows: list[dict[str, Any]],
    *,
    left: str = "P25",
    right: str = "CYCLIC25",
    repeats: int = 10_000,
    seed: int = 20260720,
) -> dict[str, Any]:
    def macro_f1(targets: np.ndarray, predictions: np.ndarray) -> float:
        matrix = np.bincount(
            3 * targets.astype(np.int64) + predictions.astype(np.int64),
            minlength=9,
        ).reshape(3, 3)
        true_positive = np.diag(matrix).astype(float)
        denominator = matrix.sum(axis=1) + matrix.sum(axis=0)
        values = np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros(3, dtype=float),
            where=denominator > 0,
        )
        return float(values.mean())

    selected = [
        row
        for row in rows
        if row["method"] in {left, right}
        and row.get("representation_seed") is not None
        and int(row["subset_seed"]) == 0
        and np.isclose(float(row["fraction"]), 1.0)
    ]
    lookup = {
        (
            str(row["method"]),
            int(row["representation_seed"]),
            str(row["event_id"]),
        ): row
        for row in selected
    }
    paired_seeds = sorted(
        {
            int(row["representation_seed"])
            for row in selected
            if row["method"] == left
        }
        & {
            int(row["representation_seed"])
            for row in selected
            if row["method"] == right
        }
    )
    if not paired_seeds:
        raise ValueError("No paired real-classification prediction rows")

    aligned: dict[
        int,
        tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]],
    ] = {}
    observed_by_seed: dict[str, float] = {}
    for representation_seed in paired_seeds:
        event_ids = sorted(
            {
                str(row["event_id"])
                for row in selected
                if row["method"] == left
                and int(row["representation_seed"]) == representation_seed
            }
        )
        values = []
        for event_id in event_ids:
            left_row = lookup.get((left, representation_seed, event_id))
            right_row = lookup.get((right, representation_seed, event_id))
            if left_row is None or right_row is None:
                raise ValueError(
                    f"Unpaired prediction for seed={representation_seed}, "
                    f"event={event_id}"
                )
            if (
                int(left_row["target"]) != int(right_row["target"])
                or str(left_row["source_group"]) != str(right_row["source_group"])
            ):
                raise ValueError("Paired prediction metadata differ")
            values.append(
                (
                    str(left_row["source_group"]),
                    int(left_row["target"]),
                    int(left_row["prediction"]),
                    int(right_row["prediction"]),
                )
            )
        if not values:
            raise ValueError(f"No aligned predictions for seed={representation_seed}")
        targets = np.asarray([row[1] for row in values])
        left_predictions = np.asarray([row[2] for row in values])
        right_predictions = np.asarray([row[3] for row in values])
        groups = np.asarray([row[0] for row in values])
        group_indices = [
            np.flatnonzero(groups == group) for group in np.unique(groups)
        ]
        aligned[representation_seed] = (
            targets,
            left_predictions,
            right_predictions,
            group_indices,
        )
        observed_by_seed[str(representation_seed)] = float(
            macro_f1(targets, right_predictions)
            - macro_f1(targets, left_predictions)
        )

    rng = np.random.default_rng(seed)
    bootstrap = []
    seed_array = np.asarray(paired_seeds)
    for _ in range(repeats):
        seed_differences = []
        for sampled_seed in rng.choice(
            seed_array, size=seed_array.size, replace=True
        ):
            targets, left_predictions, right_predictions, group_indices = (
                aligned[int(sampled_seed)]
            )
            sampled_groups = rng.integers(
                0, len(group_indices), size=len(group_indices)
            )
            sampled_rows = np.concatenate(
                [group_indices[index] for index in sampled_groups]
            )
            seed_differences.append(
                macro_f1(
                    targets[sampled_rows],
                    right_predictions[sampled_rows],
                )
                - macro_f1(
                    targets[sampled_rows],
                    left_predictions[sampled_rows],
                )
            )
        bootstrap.append(float(np.mean(seed_differences)))
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "contrast": f"{right}-{left}",
        "metric": "macro_f1",
        "mean_difference": float(np.mean(list(observed_by_seed.values()))),
        "ci95": [float(low), float(high)],
        "differences_by_representation_seed": observed_by_seed,
        "n_independent_training_pairs": len(observed_by_seed),
        "cluster_unit": "source_group",
        "bootstrap_repeats": repeats,
        "warning": (
            "The 10um class remains exploratory because validation contains "
            "only 13 events; bootstrap repetitions do not add observations."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_nested(subsets: dict[float, np.ndarray]) -> None:
    previous: set[int] = set()
    for fraction in sorted(subsets):
        current = set(np.asarray(subsets[fraction], dtype=int).tolist())
        if not previous <= current:
            raise ValueError("Label-efficiency subsets are not nested")
        previous = current


def nominal_ssl_budget(*, signals: int = 6982, epochs: int = 20) -> dict[str, int]:
    return {
        "epochs": epochs,
        "batch_size": 32,
        "optimizer_updates": epochs * math.ceil(signals / 32),
        "signals_seen": signals * epochs,
        "masked_values_contributing_to_loss": signals * epochs * 1024,
    }
