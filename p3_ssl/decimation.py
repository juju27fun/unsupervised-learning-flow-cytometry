from __future__ import annotations

import numpy as np


def ensure_1d_signal(array: np.ndarray, channel: int = 0) -> np.ndarray:
    """Return a single 1D float32 signal from common repository `.npy` shapes."""
    arr = np.asarray(array)
    if arr.ndim == 1:
        signal = arr
    elif arr.ndim == 2:
        if arr.shape[0] <= arr.shape[1]:
            signal = arr[channel]
        else:
            signal = arr[:, channel]
    else:
        signal = np.squeeze(arr)
        if signal.ndim != 1:
            raise ValueError(f"Expected 1D or 2D signal array, got shape {arr.shape}")
    return np.asarray(signal, dtype=np.float32)


def crop_or_pad(signal: np.ndarray, length: int, mode: str = "center") -> np.ndarray:
    """Crop or zero-pad a 1D signal to a fixed length."""
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {x.shape}")
    if x.size == length:
        return x
    if x.size > length:
        if mode == "left":
            start = 0
        elif mode == "right":
            start = x.size - length
        else:
            start = (x.size - length) // 2
        return x[start : start + length]
    out = np.zeros(length, dtype=np.float32)
    if mode == "left":
        out[: x.size] = x
    elif mode == "right":
        out[-x.size :] = x
    else:
        start = (length - x.size) // 2
        out[start : start + x.size] = x
    return out


def decimate_signal(signal: np.ndarray, factor: int = 8, method: str = "mean") -> np.ndarray:
    """Decimate a 1D signal by an integer factor.

    `mean` uses block averaging as a cheap anti-aliasing default. `stride` keeps
    every `factor`-th sample for exact indexing experiments.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {x.shape}")
    usable = (x.size // factor) * factor
    if usable == 0:
        raise ValueError("signal shorter than decimation factor")
    x = x[:usable]
    if method == "stride":
        return x[::factor].astype(np.float32, copy=False)
    if method == "mean":
        return x.reshape(-1, factor).mean(axis=1).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported decimation method: {method}")


def normalize_signal(signal: np.ndarray, mode: str = "window_zscore", eps: float = 1.0e-6) -> np.ndarray:
    """Normalize a signal with deterministic local statistics."""
    x = np.asarray(signal, dtype=np.float32)
    if mode in ("none", None):
        return x
    if mode == "window_zscore":
        mean = float(np.mean(x))
        std = float(np.std(x))
        return ((x - mean) / max(std, eps)).astype(np.float32, copy=False)
    if mode == "robust_zscore":
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = max(1.4826 * mad, eps)
        return ((x - med) / scale).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported normalization mode: {mode}")

