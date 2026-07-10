from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from p3_ssl.decimation import crop_or_pad, normalize_signal


PREPROCESS_NONE = "none"
PREPROCESS_P1 = "p1_bandpass_saturation"
PREPROCESS_MODES = (PREPROCESS_NONE, PREPROCESS_P1)


@dataclass(frozen=True)
class P1PreprocessConfig:
    mode: str = PREPROCESS_NONE
    sampling_frequency_hz: float = 2_000_000.0
    low_khz: float = 5.0
    high_khz_max: float = 100.0
    saturation_fmin_hz: float = 7_000.0
    saturation_fmax_hz: float = 80_000.0
    saturation_min_flat: int = 500
    saturation_zero_threshold: float = 1.0e-4
    saturation_guard_before: int = 0
    saturation_guard_after: int = 0
    min_std: float = 1.0e-7
    normalization: str = "window_zscore"

    @property
    def preprocessing_id(self) -> str:
        if self.mode == PREPROCESS_NONE:
            return PREPROCESS_NONE
        return (
            f"{self.mode}:bp{self.low_khz:g}-{self.high_khz_max:g}khz:"
            f"sat{self.saturation_fmin_hz:g}-{self.saturation_fmax_hz:g}hz:"
            f"flat{self.saturation_min_flat}:thr{self.saturation_zero_threshold:g}:"
            f"norm{self.normalization}"
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["preprocessing_id"] = self.preprocessing_id
        return data


def adaptive_bandpass_decimate_np(
    signal: np.ndarray,
    *,
    target_length: int,
    native_fs_hz: float = 2_000_000.0,
    low_khz: float = 5.0,
    high_khz_max: float = 100.0,
    chunk_length: int | None = None,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {x.shape}")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if x.size % int(target_length) != 0:
        raise ValueError(f"signal length {x.size} must be divisible by target_length {target_length}")
    factor = x.size // int(target_length)
    new_fs = float(native_fs_hz) / max(factor, 1)
    high_hz = min(float(high_khz_max) * 1000.0, 0.9 * new_fs / 2.0)
    low_hz = float(low_khz) * 1000.0

    def _filter(row: np.ndarray) -> np.ndarray:
        spectrum = np.fft.fft(row.astype(np.float64))
        freqs = np.fft.fftfreq(row.size, d=1.0 / float(native_fs_hz))
        mask = (np.abs(freqs) >= low_hz) & (np.abs(freqs) <= high_hz)
        filtered = np.fft.ifft(spectrum * mask).real.astype(np.float32)
        return filtered[::factor].astype(np.float32, copy=False) if factor > 1 else filtered

    if chunk_length is not None and 0 < int(chunk_length) < x.size:
        chunk = int(chunk_length)
        if x.size % chunk != 0:
            raise ValueError(f"signal length {x.size} must be divisible by chunk_length {chunk}")
        if chunk % factor != 0:
            raise ValueError(f"chunk_length {chunk} must be divisible by decimation factor {factor}")
        pieces = [_filter(x[start : start + chunk]) for start in range(0, x.size, chunk)]
        return np.concatenate(pieces).astype(np.float32, copy=False)
    return _filter(x)


def _merge_intervals(intervals: list[tuple[int, int]], signal_len: int) -> list[tuple[int, int]]:
    clipped = []
    for start, end in intervals:
        s = max(0, min(int(signal_len), int(start)))
        e = max(0, min(int(signal_len), int(end)))
        if e > s:
            clipped.append((s, e))
    if not clipped:
        return []
    clipped.sort()
    merged = [clipped[0]]
    for start, end in clipped[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def detect_flat_saturation_intervals(
    signal: np.ndarray,
    *,
    min_flat: int = 500,
    zero_threshold: float = 1.0e-4,
    guard_before: int = 0,
    guard_after: int = 0,
) -> list[tuple[int, int]]:
    x = np.asarray(signal, dtype=np.float32)
    if x.size < 2:
        return [(0, int(x.size))] if x.size else []
    flat = np.abs(np.diff(x.astype(np.float64))) < float(zero_threshold)
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    count = 0
    for idx, is_flat in enumerate(flat.tolist()):
        if is_flat:
            if start is None:
                start = idx
            count += 1
        else:
            if start is not None and count >= int(min_flat):
                intervals.append((start - int(guard_before), start + count + 1 + int(guard_after)))
            start = None
            count = 0
    if start is not None and count >= int(min_flat):
        intervals.append((start - int(guard_before), start + count + 1 + int(guard_after)))
    return _merge_intervals(intervals, int(x.size))


def signal_quality_report(signal: np.ndarray, cfg: P1PreprocessConfig) -> dict[str, Any]:
    x = np.asarray(signal, dtype=np.float32)
    finite = bool(np.isfinite(x).all())
    std = float(np.std(x[np.isfinite(x)])) if np.any(np.isfinite(x)) else float("nan")
    raw_intervals: list[tuple[int, int]] = []
    filtered_intervals: list[tuple[int, int]] = []
    if finite and cfg.mode == PREPROCESS_P1:
        raw_intervals = detect_flat_saturation_intervals(
            x,
            min_flat=cfg.saturation_min_flat,
            zero_threshold=cfg.saturation_zero_threshold,
            guard_before=cfg.saturation_guard_before,
            guard_after=cfg.saturation_guard_after,
        )
        sat_probe = adaptive_bandpass_decimate_np(
            x,
            target_length=x.size,
            native_fs_hz=cfg.sampling_frequency_hz,
            low_khz=cfg.saturation_fmin_hz / 1000.0,
            high_khz_max=cfg.saturation_fmax_hz / 1000.0,
        )
        filtered_intervals = detect_flat_saturation_intervals(
            sat_probe,
            min_flat=cfg.saturation_min_flat,
            zero_threshold=cfg.saturation_zero_threshold,
            guard_before=cfg.saturation_guard_before,
            guard_after=cfg.saturation_guard_after,
        )
    intervals = _merge_intervals([*raw_intervals, *filtered_intervals], int(x.size))
    reason = ""
    if not finite:
        reason = "non_finite"
    elif not np.isfinite(std) or std < float(cfg.min_std):
        reason = "near_constant"
    elif intervals:
        reason = "flat_saturation_interval"
    return {
        "finite": finite,
        "std": std,
        "n_saturation_intervals": int(len(intervals)),
        "saturation_intervals": [{"start": int(s), "end": int(e)} for s, e in intervals],
        "n_raw_flat_intervals": int(len(raw_intervals)),
        "n_filtered_flat_intervals": int(len(filtered_intervals)),
        "reject_reason": reason,
        "ok": reason == "",
    }


def preprocess_signal(signal: np.ndarray, *, output_length: int, cfg: P1PreprocessConfig) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    if cfg.mode == PREPROCESS_NONE:
        prepared = crop_or_pad(x, int(output_length), mode="center")
    elif cfg.mode == PREPROCESS_P1:
        prepared = adaptive_bandpass_decimate_np(
            x,
            target_length=int(output_length),
            native_fs_hz=cfg.sampling_frequency_hz,
            low_khz=cfg.low_khz,
            high_khz_max=cfg.high_khz_max,
        )
    else:
        raise ValueError(f"Unsupported preprocess mode: {cfg.mode}")
    return normalize_signal(prepared, mode=cfg.normalization).astype(np.float32, copy=False)


def preprocess_batch(signals: np.ndarray, *, cfg: P1PreprocessConfig, reject_saturation: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    rows = np.asarray(signals, dtype=np.float32)
    if rows.ndim != 2:
        raise ValueError(f"Expected 2D signal batch, got shape {rows.shape}")
    processed: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    keep: list[bool] = []
    for row in rows:
        report = signal_quality_report(row, cfg)
        reports.append(report)
        ok = bool(report["ok"]) or (report["reject_reason"] == "flat_saturation_interval" and not reject_saturation)
        keep.append(ok)
        if ok:
            processed.append(preprocess_signal(row, output_length=rows.shape[1], cfg=cfg))
    summary = summarize_quality_reports(reports)
    summary["kept_rows"] = int(sum(keep))
    summary["rejected_rows"] = int(len(keep) - sum(keep))
    summary["preprocessing"] = cfg.to_dict()
    if not processed:
        raise ValueError(f"Preprocessing rejected every row: {summary}")
    return np.stack(processed).astype(np.float32), summary


def summarize_quality_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    n_intervals = 0
    for report in reports:
        reason = str(report.get("reject_reason") or "ok")
        reasons[reason] = reasons.get(reason, 0) + 1
        n_intervals += int(report.get("n_saturation_intervals", 0))
    return {
        "rows": int(len(reports)),
        "reason_counts": reasons,
        "saturation_intervals": int(n_intervals),
    }
