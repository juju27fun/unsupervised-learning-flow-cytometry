from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal as scipy_signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OBSERVABLE_NAMES = (
    "duration_ms",
    "dominant_frequency_hz",
    "rms",
    "snr_estimate",
    "spectral_peak_count",
    "event_offset_fraction",
)


@dataclass(frozen=True)
class DomainProbeResult:
    roc_auc: float
    converged: bool
    importance: dict[str, float]
    probabilities: np.ndarray


def signal_observables(
    signals: np.ndarray, *, sampling_frequency_hz: float = 1_000_000.0
) -> np.ndarray:
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("signals must be two-dimensional")
    envelope = np.abs(scipy_signal.hilbert(x, axis=1))
    peak = np.max(envelope, axis=1)
    threshold = 0.25 * peak[:, None]
    support = envelope >= threshold
    duration = np.sum(support, axis=1) / sampling_frequency_hz * 1000.0
    weights = envelope / np.maximum(np.sum(envelope, axis=1, keepdims=True), 1.0e-12)
    positions = np.arange(x.shape[1], dtype=np.float64)
    offset = np.sum(weights * positions, axis=1) / max(x.shape[1] - 1, 1) - 0.5
    rms = np.sqrt(np.mean(np.square(x), axis=1))
    edge = np.concatenate([x[:, :512], x[:, -512:]], axis=1)
    median = np.median(edge, axis=1, keepdims=True)
    noise = 1.4826 * np.median(np.abs(edge - median), axis=1)
    snr = rms / np.maximum(noise, 1.0e-8)
    spectrum = np.abs(np.fft.rfft(x, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(x.shape[1], d=1.0 / sampling_frequency_hz)
    band = (frequencies >= 5_000.0) & (frequencies <= 100_000.0)
    band_spectrum = spectrum[:, band]
    band_frequencies = frequencies[band]
    dominant = band_frequencies[np.argmax(band_spectrum, axis=1)]
    peak_count = np.asarray(
        [
            len(scipy_signal.find_peaks(row, prominence=max(float(np.max(row)) * 0.10, 1.0e-12))[0])
            for row in band_spectrum
        ],
        dtype=np.float64,
    )
    result = np.column_stack([duration, dominant, rms, snr, peak_count, offset])
    if not np.isfinite(result).all():
        raise ValueError("Signal observables contain non-finite values")
    return result.astype(np.float32)


def signal_summary_features(
    signals: np.ndarray, *, sampling_frequency_hz: float = 1_000_000.0
) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(signals, dtype=np.float64)
    observables = signal_observables(x, sampling_frequency_hz=sampling_frequency_hz)
    spectrum = np.abs(np.fft.rfft(x, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(x.shape[1], d=1.0 / sampling_frequency_hz)
    total = np.maximum(np.sum(spectrum, axis=1), 1.0e-12)
    edges = np.asarray((5, 10, 20, 40, 60, 80, 100), dtype=float) * 1000.0
    bands = []
    band_names = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(np.sum(spectrum[:, mask], axis=1) / total)
        band_names.append(f"spectral_fraction_{int(low)}_{int(high)}hz")
    quantiles = np.quantile(x, (0.05, 0.25, 0.50, 0.75, 0.95), axis=1).T
    values = np.column_stack([observables, *bands, quantiles])
    names = [*OBSERVABLE_NAMES, *band_names, "q05", "q25", "q50", "q75", "q95"]
    return values.astype(np.float32), names


def matched_pairs(
    real: np.ndarray,
    synthetic: np.ndarray,
    *,
    scaler: StandardScaler,
    caliper: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if caliper <= 0.0:
        raise ValueError("caliper must be positive")
    real_z = scaler.transform(real)
    synthetic_z = scaler.transform(synthetic)
    neighbors = NearestNeighbors(n_neighbors=min(10, len(synthetic_z))).fit(synthetic_z)
    distances, candidates = neighbors.kneighbors(real_z)
    proposals = []
    for real_index in range(len(real_z)):
        for rank, synthetic_index in enumerate(candidates[real_index]):
            distance = float(distances[real_index, rank])
            if distance <= caliper:
                proposals.append((distance, real_index, int(synthetic_index)))
    used_real: set[int] = set()
    used_synthetic: set[int] = set()
    pairs = []
    for distance, real_index, synthetic_index in sorted(proposals):
        if real_index in used_real or synthetic_index in used_synthetic:
            continue
        used_real.add(real_index)
        used_synthetic.add(synthetic_index)
        pairs.append((real_index, synthetic_index, distance))
    if not pairs:
        raise ValueError("No common support within the frozen matching caliper")
    real_indices = np.asarray([item[0] for item in pairs], dtype=np.int64)
    synthetic_indices = np.asarray([item[1] for item in pairs], dtype=np.int64)
    matched_real = real_z[real_indices]
    matched_synthetic = synthetic_z[synthetic_indices]
    pooled = np.sqrt(
        np.maximum((np.var(matched_real, axis=0) + np.var(matched_synthetic, axis=0)) / 2.0, 1.0e-12)
    )
    smd = np.abs(np.mean(matched_real, axis=0) - np.mean(matched_synthetic, axis=0)) / pooled
    report = {
        "n_real_input": len(real),
        "n_synthetic_input": len(synthetic),
        "n_pairs": len(pairs),
        "real_retained_fraction": len(pairs) / len(real),
        "synthetic_retained_fraction": len(pairs) / len(synthetic),
        "distance_mean": float(np.mean([item[2] for item in pairs])),
        "distance_max": float(np.max([item[2] for item in pairs])),
        "post_match_smd_max": float(np.max(smd)),
        "post_match_smd": smd.tolist(),
        "caliper": caliper,
    }
    return real_indices, synthetic_indices, report


def fit_domain_probe(
    train_real: np.ndarray,
    train_synthetic: np.ndarray,
    validation_real: np.ndarray,
    validation_synthetic: np.ndarray,
    *,
    feature_names: list[str],
    model: str,
    seed: int = 42,
    compute_importance: bool = True,
) -> DomainProbeResult:
    train_x = np.concatenate([train_real, train_synthetic])
    train_y = np.concatenate([np.zeros(len(train_real)), np.ones(len(train_synthetic))])
    validation_x = np.concatenate([validation_real, validation_synthetic])
    validation_y = np.concatenate(
        [np.zeros(len(validation_real)), np.ones(len(validation_synthetic))]
    )
    if model == "linear":
        estimator: Any = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
        )
    elif model == "forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown domain model: {model}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(train_x, train_y)
    probabilities = estimator.predict_proba(validation_x)[:, 1]
    importance = {}
    if compute_importance:
        importance_result = permutation_importance(
            estimator,
            validation_x,
            validation_y,
            scoring="roc_auc",
            n_repeats=5,
            random_state=seed,
            n_jobs=-1,
        )
        importance = {
            name: float(value)
            for name, value in sorted(
                zip(feature_names, importance_result.importances_mean),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
        }
    return DomainProbeResult(
        roc_auc=float(roc_auc_score(validation_y, probabilities)),
        converged=not any(issubclass(item.category, ConvergenceWarning) for item in caught),
        importance=importance,
        probabilities=probabilities,
    )
