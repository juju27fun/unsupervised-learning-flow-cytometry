from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_FAMILY_ORDER = (
    "time_morphology",
    "frequency",
    "envelope",
    "energy_amplitude",
    "quality",
)
FINAL_SPLITS = frozenset({"in_session_test", "sealed_acquisition_test", "followup_test", "test"})


@dataclass(frozen=True)
class FollowupData:
    signals: np.ndarray
    rows: list[dict[str, str]]
    labels: np.ndarray
    class_names: list[str]
    train_indices: np.ndarray
    validation_indices: np.ndarray


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_followup_development(root: Path) -> FollowupData:
    development_index = root / "development_events.csv"
    if not development_index.is_file():
        raise FileNotFoundError(
            "The prospective loader requires a physically separated development_events.csv"
        )
    rows = _read_rows(development_index)
    forbidden = sorted({row["development_split"] for row in rows} & FINAL_SPLITS)
    if forbidden:
        raise PermissionError(f"Development index contains final splits: {forbidden}")
    allowed_rows = list(rows)
    if not allowed_rows:
        raise ValueError("No follow-up development rows")
    signals = np.load(root / "signals.npy", mmap_mode="r")
    row_indices = np.asarray([int(row["signal_row"]) for row in allowed_rows], dtype=np.int64)
    if np.any(row_indices < 0) or np.any(row_indices >= len(signals)):
        raise ValueError("Event signal_row is outside signals.npy")
    class_names = sorted({row["source_group"] for row in allowed_rows})
    class_to_id = {name: index for index, name in enumerate(class_names)}
    labels = np.asarray([class_to_id[row["source_group"]] for row in allowed_rows], dtype=np.int64)
    split = np.asarray([row["development_split"] for row in allowed_rows])
    train = np.flatnonzero(split == "followup_train")
    validation = np.flatnonzero(split == "followup_validation")
    if not train.size or not validation.size:
        raise ValueError("followup_train and followup_validation must be non-empty")
    return FollowupData(
        np.asarray(signals[row_indices], dtype=np.float32),
        allowed_rows,
        labels,
        class_names,
        train,
        validation,
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, 1.0e-12)


def extract_feature_families(
    signals: np.ndarray,
    rows: list[dict[str, str]],
    *,
    sampling_frequency_hz: float = 1_000_000.0,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2 or len(rows) != len(x):
        raise ValueError("Expected one metadata row per two-dimensional signal")
    abs_x = np.abs(x)
    rms = np.sqrt(np.mean(np.square(x), axis=1))
    q05, q25, q50, q75, q95 = np.quantile(x, (0.05, 0.25, 0.50, 0.75, 0.95), axis=1)
    iqr = np.maximum(q75 - q25, 1.0e-12)
    centered = x - np.mean(x, axis=1, keepdims=True)
    lag_16 = _safe_ratio(
        np.mean(centered[:, :-16] * centered[:, 16:], axis=1),
        np.mean(np.square(centered), axis=1),
    )
    time_morphology = np.column_stack(
        [
            np.mean(np.diff(np.signbit(x), axis=1), axis=1),
            stats.skew(x, axis=1, bias=False),
            stats.kurtosis(x, axis=1, fisher=True, bias=False),
            (q95 + q05 - 2.0 * q50) / iqr,
            (q75 + q25 - 2.0 * q50) / iqr,
            lag_16,
            np.asarray([float(row["width_ms"]) for row in rows]),
        ]
    )

    spectrum = np.abs(np.fft.rfft(x, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(x.shape[1], d=1.0 / sampling_frequency_hz)
    total = np.maximum(np.sum(spectrum, axis=1), 1.0e-12)
    centroid = np.sum(spectrum * frequencies, axis=1) / total
    bandwidth = np.sqrt(
        np.sum(spectrum * np.square(frequencies[None, :] - centroid[:, None]), axis=1) / total
    )
    peak_frequency = frequencies[np.argmax(spectrum, axis=1)]
    band_edges = np.asarray((5, 10, 20, 40, 60, 80, 100), dtype=float) * 1000.0
    bands = []
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(np.sum(spectrum[:, mask], axis=1) / total)
    frequency = np.column_stack([centroid, bandwidth, peak_frequency, *bands])

    analytic_envelope = np.abs(scipy_signal.hilbert(x, axis=1))
    envelope_peak = np.max(analytic_envelope, axis=1)
    envelope_mean = np.mean(analytic_envelope, axis=1)
    envelope = np.column_stack(
        [
            _safe_ratio(envelope_peak, envelope_mean),
            np.mean(analytic_envelope >= 0.25 * envelope_peak[:, None], axis=1),
            np.mean(analytic_envelope >= 0.50 * envelope_peak[:, None], axis=1),
            _safe_ratio(
                np.sum(np.square(analytic_envelope), axis=1),
                np.square(np.sum(analytic_envelope, axis=1)),
            ),
        ]
    )
    energy_amplitude = np.column_stack(
        [rms, np.std(x, axis=1), np.mean(abs_x, axis=1), np.max(abs_x, axis=1), np.ptp(x, axis=1), q95 - q05]
    )
    quality = np.column_stack(
        [
            np.log1p(np.asarray([float(row["snr_proxy"]) for row in rows])),
            np.asarray([float(row["energy_concentration"]) for row in rows]),
            np.asarray([float(row["phase_coherence"]) for row in rows]),
            np.asarray([float(row["n_doppler_peaks"]) for row in rows]),
        ]
    )
    families = {
        "time_morphology": time_morphology,
        "frequency": frequency,
        "envelope": envelope,
        "energy_amplitude": energy_amplitude,
        "quality": quality,
    }
    names = {
        "time_morphology": [
            "zero_crossing_rate", "skewness", "excess_kurtosis", "tail_asymmetry",
            "central_asymmetry", "autocorrelation_lag_16", "detected_width_ms",
        ],
        "frequency": [
            "spectral_centroid_hz", "spectral_bandwidth_hz", "dominant_frequency_hz",
            "band_5_10khz", "band_10_20khz", "band_20_40khz", "band_40_60khz",
            "band_60_80khz", "band_80_100khz",
        ],
        "envelope": [
            "envelope_peak_to_mean", "envelope_support_25pct", "envelope_support_50pct",
            "envelope_energy_concentration",
        ],
        "energy_amplitude": [
            "rms", "standard_deviation", "mean_absolute_amplitude",
            "maximum_absolute_amplitude", "peak_to_peak", "q95_q05_range",
        ],
        "quality": [
            "log1p_detector_snr_proxy", "detector_energy_concentration",
            "detector_phase_coherence", "detector_doppler_peak_count",
        ],
    }
    for family, values in families.items():
        if values.shape[1] != len(names[family]) or not np.isfinite(values).all():
            raise ValueError(f"Invalid feature family: {family}")
    return {key: value.astype(np.float32) for key, value in families.items()}, names


def feature_matrix(
    families: dict[str, np.ndarray], *, include: tuple[str, ...] = FEATURE_FAMILY_ORDER
) -> np.ndarray:
    missing = set(include) - set(families)
    if missing:
        raise ValueError(f"Unknown feature families: {sorted(missing)}")
    lengths = {len(families[name]) for name in include}
    if len(lengths) != 1:
        raise ValueError("Feature families have inconsistent row counts")
    return np.concatenate([families[name] for name in include], axis=1)


def sample_record_groups(
    rows: list[dict[str, str]], labels: np.ndarray, indices: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    records: set[str] = set()
    for class_id in sorted(set(labels[indices].tolist())):
        candidates = indices[labels[indices] == class_id]
        unique = np.asarray(sorted({rows[int(index)]["record_id"] for index in candidates}))
        count = max(1, int(np.ceil(fraction * len(unique))))
        selected = unique if count >= len(unique) else rng.choice(unique, count, replace=False)
        records.update(str(value) for value in selected)
    return np.asarray(
        [index for index in indices if rows[int(index)]["record_id"] in records], dtype=np.int64
    )


def fit_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    probe: str,
    seed: int,
    class_names: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    if probe == "linear":
        classifier: Any = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
    elif probe == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(32,), alpha=1.0e-3, batch_size=128,
            learning_rate_init=1.0e-3, max_iter=300, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown probe: {probe}")
    model = make_pipeline(StandardScaler(), classifier)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_x, train_y)
    predictions = model.predict(validation_x)
    probabilities = model.predict_proba(validation_x)
    estimator = model.steps[-1][1]
    observed = set(validation_y.tolist())
    recalls = recall_score(
        validation_y, predictions, labels=np.arange(len(class_names)), average=None, zero_division=0
    )
    metrics = {
        "macro_f1": float(
            f1_score(
                validation_y, predictions, labels=np.arange(len(class_names)), average="macro", zero_division=0
            )
        ),
        "balanced_accuracy_observed_classes": float(balanced_accuracy_score(validation_y, predictions)),
        "multiclass_log_loss": float(log_loss(validation_y, probabilities, labels=np.arange(len(class_names)))),
        "per_class_recall": {
            name: (None if index not in observed else float(recalls[index]))
            for index, name in enumerate(class_names)
        },
        "converged": not any(issubclass(item.category, ConvergenceWarning) for item in caught),
        "convergence_warning_count": sum(issubclass(item.category, ConvergenceWarning) for item in caught),
        "n_iter": np.asarray(getattr(estimator, "n_iter_", []), dtype=int).tolist(),
    }
    return metrics, predictions


def load_historical_embeddings(
    *,
    artifact_root: Path,
    followup_rows: list[dict[str, str]],
    cells: tuple[str, ...] = ("a3", "a4"),
    seeds: tuple[int, ...] = (42, 43, 44),
) -> dict[str, np.ndarray]:
    parent_rows = np.asarray([int(row["parent_signal_row"]) for row in followup_rows])
    available_rows = np.load(artifact_root / "real_embedding_row_indices.npy")
    lookup = {int(value): index for index, value in enumerate(available_rows)}
    missing = sorted(set(parent_rows.tolist()) - set(lookup))
    if missing:
        raise ValueError(f"Historical embeddings miss {len(missing)} follow-up development rows")
    positions = np.asarray([lookup[int(value)] for value in parent_rows], dtype=np.int64)
    output = {}
    for cell in cells:
        for seed in seeds:
            values = np.load(artifact_root / f"real_embeddings_{cell}_s{seed}.npy", mmap_mode="r")
            output[f"{cell.upper()}_s{seed}"] = np.asarray(values[positions], dtype=np.float32)
    return output


def write_feature_manifest(path: Path, names: dict[str, list[str]]) -> None:
    payload = {
        "schema_version": 1,
        "families": names,
        "family_order": list(FEATURE_FAMILY_ORDER),
        "quality_shortcut_warning": "Quality features are detector-derived controls, not biological measurements.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
