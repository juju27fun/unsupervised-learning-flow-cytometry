from __future__ import annotations

import numpy as np


def sample_region_block_mask(
    event_mask: np.ndarray,
    block_length: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one contiguous block, preferably fully inside the requested region."""
    event = np.asarray(event_mask, dtype=bool)
    if event.ndim != 1:
        raise ValueError("event_mask must be one-dimensional")
    if not 0 < block_length <= event.size:
        raise ValueError("block_length must be in [1, signal length]")
    if mode not in {"random", "event", "background"}:
        raise ValueError("mode must be random, event, or background")

    if mode == "random":
        valid_starts = np.arange(event.size - block_length + 1)
        anchor_region = np.ones_like(event)
    else:
        anchor_region = event if mode == "event" else ~event
        coverage = np.convolve(
            anchor_region.astype(np.int32), np.ones(block_length, dtype=np.int32), mode="valid"
        )
        valid_starts = np.flatnonzero(coverage == block_length)

    if valid_starts.size:
        start = int(rng.choice(valid_starts))
    else:
        anchors = np.flatnonzero(anchor_region)
        center = int(rng.choice(anchors)) if anchors.size else event.size // 2
        start = max(0, min(center - block_length // 2, event.size - block_length))
    mask = np.zeros(event.size, dtype=bool)
    mask[start : start + block_length] = True
    return mask


def visible_mean_prediction(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    visible = ~mask
    value = float(signal[visible].mean()) if np.any(visible) else 0.0
    return np.full_like(signal, value)


def interpolation_prediction(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    indices = np.arange(signal.size)
    visible = ~mask
    prediction = signal.copy()
    if np.count_nonzero(visible) < 2:
        prediction[mask] = 0.0
    else:
        prediction[mask] = np.interp(indices[mask], indices[visible], signal[visible])
    return prediction


def nearest_prediction(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    indices = np.arange(signal.size)
    visible_indices = indices[~mask]
    prediction = signal.copy()
    if visible_indices.size == 0:
        prediction[mask] = 0.0
        return prediction
    insertion = np.searchsorted(visible_indices, indices)
    left = visible_indices[np.clip(insertion - 1, 0, visible_indices.size - 1)]
    right = visible_indices[np.clip(insertion, 0, visible_indices.size - 1)]
    nearest = np.where(indices - left <= right - indices, left, right)
    prediction[mask] = signal[nearest[mask]]
    return prediction


def _fit_autoregressive_coefficients(
    signal: np.ndarray,
    mask: np.ndarray,
    order: int,
    ridge: float,
) -> np.ndarray | None:
    rows = []
    targets = []
    for index in range(order, signal.size):
        if mask[index] or np.any(mask[index - order : index]):
            continue
        rows.append(signal[index - order : index][::-1])
        targets.append(signal[index])
    if len(rows) < max(order + 1, 32):
        return None
    design = np.asarray(rows, dtype=np.float64)
    response = np.asarray(targets, dtype=np.float64)
    design = np.column_stack([design, np.ones(len(design))])
    gram = design.T @ design
    scale = float(np.trace(gram[:-1, :-1])) / max(order, 1)
    regularizer = np.eye(order + 1, dtype=np.float64) * ridge * max(scale, 1.0e-12)
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(gram + regularizer, design.T @ response)


def _autoregressive_forward(
    signal: np.ndarray,
    mask: np.ndarray,
    order: int,
    ridge: float,
) -> np.ndarray:
    coefficients = _fit_autoregressive_coefficients(signal, mask, order, ridge)
    prediction = signal.copy()
    if coefficients is None:
        prediction[mask] = 0.0
        return prediction
    visible_std = float(np.std(signal[~mask])) if np.any(~mask) else 0.0
    limit = max(5.0 * visible_std, 1.0e-6)
    for index in np.flatnonzero(mask):
        if index < order:
            prediction[index] = 0.0
            continue
        history = prediction[index - order : index][::-1]
        value = float(history @ coefficients[:-1] + coefficients[-1])
        prediction[index] = float(np.clip(value, -limit, limit))
    return prediction


def autoregressive_prediction(
    signal: np.ndarray,
    mask: np.ndarray,
    *,
    order: int = 64,
    ridge: float = 1.0e-3,
) -> np.ndarray:
    """Fill gaps with blended forward and backward ridge autoregression."""
    signal = np.asarray(signal, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if signal.ndim != 1 or mask.shape != signal.shape:
        raise ValueError("signal and mask must be matching one-dimensional arrays")
    if order <= 0 or order >= signal.size or ridge < 0.0:
        raise ValueError("invalid autoregressive configuration")
    forward = _autoregressive_forward(signal, mask, order, ridge)
    backward = _autoregressive_forward(signal[::-1], mask[::-1], order, ridge)[::-1]
    prediction = signal.copy()
    masked_indices = np.flatnonzero(mask)
    split_points = np.flatnonzero(np.diff(masked_indices) > 1) + 1
    for gap in np.split(masked_indices, split_points):
        if gap.size == 0:
            continue
        backward_weight = (np.arange(gap.size, dtype=np.float64) + 1.0) / (gap.size + 1.0)
        prediction[gap] = (
            (1.0 - backward_weight) * forward[gap] + backward_weight * backward[gap]
        )
    return prediction


def harmonic_regression_prediction(
    signal: np.ndarray,
    mask: np.ndarray,
    *,
    sampling_frequency_hz: float,
    minimum_frequency_hz: float = 5_000.0,
    maximum_frequency_hz: float = 30_000.0,
    frequency_bins: int = 96,
    context_radius: int = 512,
) -> np.ndarray:
    """Fill contiguous gaps with local least-squares frequency-grid sinusoids."""
    signal = np.asarray(signal, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    masked_indices = np.flatnonzero(mask)
    if masked_indices.size == 0:
        return signal.copy()
    prediction = signal.copy()
    frequencies = np.linspace(minimum_frequency_hz, maximum_frequency_hz, frequency_bins)
    split_points = np.flatnonzero(np.diff(masked_indices) > 1) + 1
    for gap_indices in np.split(masked_indices, split_points):
        start = int(gap_indices[0])
        end = int(gap_indices[-1]) + 1
        context_start = max(0, start - context_radius)
        context_end = min(signal.size, end + context_radius)
        context_indices = np.arange(context_start, context_end)
        context_indices = context_indices[~mask[context_indices]]
        if context_indices.size < 8:
            prediction[gap_indices] = 0.0
            continue

        center = 0.5 * (start + end - 1)
        context_time = (context_indices - center) / sampling_frequency_hz
        context_values = signal[context_indices]
        centered_values = context_values - context_values.mean()
        phase = 2.0 * np.pi * np.outer(context_time, frequencies)
        cosine_score = centered_values @ np.cos(phase)
        sine_score = centered_values @ np.sin(phase)
        frequency = float(frequencies[np.argmax(cosine_score**2 + sine_score**2)])

        fit_phase = 2.0 * np.pi * frequency * context_time
        design = np.column_stack(
            [np.sin(fit_phase), np.cos(fit_phase), np.ones(context_indices.size)]
        )
        coefficients, *_ = np.linalg.lstsq(design, context_values, rcond=None)
        target_time = (gap_indices - center) / sampling_frequency_hz
        target_phase = 2.0 * np.pi * frequency * target_time
        target_design = np.column_stack(
            [np.sin(target_phase), np.cos(target_phase), np.ones(gap_indices.size)]
        )
        prediction[gap_indices] = target_design @ coefficients
    return prediction


def masked_mse_numpy(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    difference = np.asarray(prediction)[np.asarray(mask, dtype=bool)] - np.asarray(target)[
        np.asarray(mask, dtype=bool)
    ]
    return float(np.mean(np.square(difference)))
