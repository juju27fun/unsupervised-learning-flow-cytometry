#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks, spectrogram

ROOT = Path(__file__).resolve().parents[1]

from p3_ssl.decimation import decimate_signal, ensure_1d_signal, normalize_signal
from p3_ssl.signal_preprocessing import PREPROCESS_MODES, PREPROCESS_NONE, PREPROCESS_P1, P1PreprocessConfig, preprocess_signal, signal_quality_report, summarize_quality_reports


@dataclass(frozen=True)
class YeastDetectionConfig:
    sampling_frequency_hz: float = 2_000_000.0
    low_freq_hz: float = 7_000.0
    high_freq_hz: float = 80_000.0
    filter_order: int = 4
    stft_nperseg: int = 512
    stft_noverlap: int = 384
    smooth_frames: int = 3
    active_snr_z: float = 3.5
    boundary_snr_z: float = 1.5
    medium_min_snr: float = 3.0
    strict_min_snr: float = 5.0
    medium_min_concentration: float = 0.08
    strict_min_concentration: float = 0.12
    strict_min_phase_coherence: float = 0.0
    frequency_peak_height_frac: float = 0.20
    frequency_peak_prominence_frac: float = 0.08
    cluster_gap_ms: float = 0.25
    boundary_pad_ms: float = 0.04
    min_width_ms: float = 0.06
    max_width_ms: float = 1.60
    raw_crop_length: int = 4096
    output_length: int = 4096
    max_events_per_signal: int = 3
    class_id: int = 3
    class_name: str = "yeast"


@dataclass(frozen=True)
class YeastEvent:
    event_id: str
    sample_id: str
    split: str
    signal_path: str
    label_path: str
    class_id: int
    class_name: str
    center_norm: float
    width_norm: float
    center_index: int
    crop_start: int
    crop_end: int
    event_start: int
    event_end: int
    width_samples: int
    width_ms: float
    snr_proxy: float
    energy_concentration: float
    phase_coherence: float
    n_doppler_peaks: int
    doppler_low_hz: float
    doppler_high_hz: float
    doppler_peak_hz: float
    quality: str
    source_group: str
    rejection_reason: str


@dataclass(frozen=True)
class FileDetectionReport:
    signal_path: str
    source_group: str
    n_candidates: int
    n_kept: int
    rejection_reason: str


def robust_scale(values: np.ndarray, eps: float = 1.0e-12) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, max(mad, eps)


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, order: int, fs: float) -> np.ndarray:
    nyquist = 0.5 * float(fs)
    low = float(lowcut) / nyquist
    high = float(highcut) / nyquist
    if not 0.0 < low < high < 1.0:
        raise ValueError(f"Invalid bandpass range: {lowcut}..{highcut} Hz at fs={fs}")
    b, a = butter(int(order), [low, high], btype="band", analog=False)
    return filtfilt(b, a, np.asarray(data, dtype=np.float64)).astype(np.float32)


def crop_around_index(raw: np.ndarray, crop_length: int, center_index: int) -> np.ndarray:
    x = ensure_1d_signal(raw)
    center = int(center_index)
    start = center - int(crop_length) // 2
    end = start + int(crop_length)
    crop = np.zeros(int(crop_length), dtype=np.float32)
    src_start = max(0, start)
    src_end = min(x.shape[0], end)
    if src_end > src_start:
        dst_start = src_start - start
        crop[dst_start : dst_start + (src_end - src_start)] = x[src_start:src_end]
    return crop


def build_aligned_signal_at_center(
    raw: np.ndarray,
    center_index: int,
    raw_crop_length: int = 4096,
    output_length: int = 4096,
    preprocess_config: P1PreprocessConfig | None = None,
) -> np.ndarray:
    if int(raw_crop_length) % int(output_length) != 0:
        raise ValueError("raw_crop_length must be divisible by output_length")
    crop = crop_around_index(raw, raw_crop_length, int(center_index))
    if preprocess_config is not None and preprocess_config.mode != PREPROCESS_NONE:
        return preprocess_signal(crop, output_length=int(output_length), cfg=preprocess_config)
    decimated = decimate_signal(crop, int(raw_crop_length) // int(output_length), method="mean")
    return normalize_signal(decimated, mode="window_zscore").astype(np.float32, copy=False)


def build_aligned_512_signal_at_center(
    raw: np.ndarray,
    center_index: int,
    raw_crop_length: int = 4096,
    output_length: int = 512,
) -> np.ndarray:
    return build_aligned_signal_at_center(
        raw,
        center_index=center_index,
        raw_crop_length=raw_crop_length,
        output_length=output_length,
        preprocess_config=None,
    )


def _group_active_frames(active: np.ndarray, max_gap_frames: int) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start: int | None = None
    last = -1
    gap = 0
    for idx, is_active in enumerate(active.astype(bool).tolist()):
        if is_active:
            if start is None:
                start = idx
            elif gap > max_gap_frames:
                groups.append((start, last))
                start = idx
            last = idx
            gap = 0
        elif start is not None:
            gap += 1
    if start is not None:
        groups.append((start, last))
    return [(left, right) for left, right in groups if right >= left]


def _expand_group_bounds(
    left: int,
    right: int,
    energy_z: np.ndarray,
    boundary_snr_z: float,
) -> tuple[int, int]:
    lo = int(left)
    hi = int(right)
    while lo > 0 and float(energy_z[lo - 1]) >= float(boundary_snr_z):
        lo -= 1
    while hi < energy_z.size - 1 and float(energy_z[hi + 1]) >= float(boundary_snr_z):
        hi += 1
    return lo, hi


def _frequency_peaks(
    event_power_by_freq: np.ndarray,
    freqs_hz: np.ndarray,
    config: YeastDetectionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    power = np.asarray(event_power_by_freq, dtype=np.float64)
    if power.size == 0 or float(np.max(power)) <= 0.0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    pmax = float(np.max(power))
    peaks, _ = find_peaks(
        power,
        height=float(config.frequency_peak_height_frac) * pmax,
        prominence=float(config.frequency_peak_prominence_frac) * pmax,
    )
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(power))], dtype=np.int64)
    order = np.argsort(power[peaks])[::-1]
    peaks = peaks[order].astype(np.int64, copy=False)
    return peaks, freqs_hz[peaks].astype(np.float32, copy=False)


def _phase_coherence(
    complex_stft: np.ndarray,
    frame_indices: np.ndarray,
    peak_indices: np.ndarray,
    max_peaks: int = 3,
) -> float:
    if frame_indices.size < 2 or peak_indices.size == 0:
        return float("nan")
    selected = peak_indices[:max_peaks]
    z = complex_stft[np.asarray(selected, dtype=np.int64)[:, None], frame_indices[None, :]]
    if z.size == 0:
        return float("nan")

    phases = np.unwrap(np.angle(z), axis=1)
    phase_step = np.diff(phases, axis=1)
    temporal = np.abs(np.mean(np.exp(1j * phase_step), axis=1))
    values = [float(np.mean(temporal))]

    if selected.size >= 2:
        pair_phase = np.angle(z[0] * np.conj(z[1]))
        values.append(float(np.abs(np.mean(np.exp(1j * pair_phase)))))
    return float(np.mean(values))


def _classify_quality(
    *,
    snr_proxy: float,
    concentration: float,
    phase_coherence: float,
    width_ms: float,
    center_index: int,
    signal_length: int,
    config: YeastDetectionConfig,
) -> tuple[str, str]:
    if width_ms < float(config.min_width_ms):
        return "reject", "width_below_min"
    if width_ms > float(config.max_width_ms):
        return "reject", "width_above_max"
    half_crop = int(config.raw_crop_length) // 2
    if center_index < half_crop or center_index > signal_length - half_crop:
        return "reject", "crop_would_hit_edge"
    phase_ok = (
        not math.isfinite(float(phase_coherence))
        or float(phase_coherence) >= float(config.strict_min_phase_coherence)
    )
    if (
        snr_proxy >= float(config.strict_min_snr)
        and concentration >= float(config.strict_min_concentration)
        and phase_ok
    ):
        return "strict", ""
    if snr_proxy >= float(config.medium_min_snr) and concentration >= float(config.medium_min_concentration):
        return "medium", ""
    return "reject", "quality_below_threshold"


def detect_yeast_passages(
    signal: np.ndarray,
    *,
    sample_id: str,
    signal_path: str,
    source_group: str,
    split: str,
    config: YeastDetectionConfig,
) -> tuple[list[YeastEvent], str]:
    raw = ensure_1d_signal(signal)
    n = int(raw.shape[0])
    if n < int(config.stft_nperseg):
        return [], "signal_too_short"

    try:
        filtered = butter_bandpass_filter(
            raw - float(np.mean(raw)),
            lowcut=float(config.low_freq_hz),
            highcut=float(config.high_freq_hz),
            order=int(config.filter_order),
            fs=float(config.sampling_frequency_hz),
        )
    except Exception:
        return [], "broadband_filter_failed"
    filtered = filtered - float(np.mean(filtered))

    try:
        freqs, _times, stft_complex = spectrogram(
            filtered,
            fs=float(config.sampling_frequency_hz),
            nperseg=int(config.stft_nperseg),
            noverlap=int(config.stft_noverlap),
            window="hann",
            mode="complex",
        )
    except Exception:
        return [], "spectrogram_failed"

    freq_mask = (freqs >= float(config.low_freq_hz)) & (freqs <= float(config.high_freq_hz))
    if not bool(np.any(freq_mask)):
        return [], "broadband_band_empty"
    freqs_band = freqs[freq_mask].astype(np.float32)
    complex_band = stft_complex[freq_mask, :]
    power = np.square(np.abs(complex_band)).astype(np.float64)

    freq_baseline = np.percentile(power, 25, axis=1, keepdims=True)
    excess = np.clip(power - freq_baseline, 0.0, None)
    frame_energy = excess.sum(axis=0)
    if int(config.smooth_frames) > 1 and frame_energy.size >= int(config.smooth_frames):
        frame_energy = uniform_filter1d(frame_energy, size=int(config.smooth_frames), mode="nearest")

    energy_med, energy_scale = robust_scale(frame_energy)
    energy_z = (frame_energy - energy_med) / energy_scale

    top_count = min(5, excess.shape[0])
    top_power = np.partition(excess, kth=excess.shape[0] - top_count, axis=0)[-top_count:, :].sum(axis=0)
    broadband_power = power.sum(axis=0) + 1.0e-12
    concentration_frame = top_power / broadband_power

    active = (energy_z >= float(config.active_snr_z)) & (
        concentration_frame >= float(config.medium_min_concentration)
    )
    hop = int(config.stft_nperseg) - int(config.stft_noverlap)
    max_gap_frames = max(0, int(round(float(config.cluster_gap_ms) / 1000.0 * float(config.sampling_frequency_hz) / hop)))
    groups = _group_active_frames(active, max_gap_frames=max_gap_frames)
    if not groups:
        return [], "no_active_time_frequency_group"

    pad_samples = int(round(float(config.boundary_pad_ms) / 1000.0 * float(config.sampling_frequency_hz)))
    half_win = int(config.stft_nperseg) // 2
    candidates: list[YeastEvent] = []

    for local_idx, (group_left, group_right) in enumerate(groups):
        left_frame, right_frame = _expand_group_bounds(group_left, group_right, energy_z, config.boundary_snr_z)
        frame_indices = np.arange(left_frame, right_frame + 1, dtype=np.int64)
        frame_centers = frame_indices.astype(np.float64) * float(hop) + float(half_win)
        weights = np.maximum(frame_energy[frame_indices] - energy_med, 0.0)
        if float(np.sum(weights)) <= 0.0:
            center_index = int(round(float(np.mean(frame_centers))))
        else:
            center_index = int(round(float(np.sum(frame_centers * weights) / np.sum(weights))))

        event_start = max(0, int(left_frame * hop) - pad_samples)
        event_end = min(n, int(right_frame * hop + int(config.stft_nperseg)) + pad_samples)
        if event_end <= event_start:
            continue
        width_samples = int(event_end - event_start)
        width_ms = float(width_samples) / float(config.sampling_frequency_hz) * 1000.0
        snr_proxy = float(np.max(energy_z[frame_indices]))

        event_power_by_freq = excess[:, frame_indices].sum(axis=1)
        total_event_power = float(np.sum(event_power_by_freq))
        if total_event_power <= 0.0:
            continue
        peak_indices, peak_freqs = _frequency_peaks(event_power_by_freq, freqs_band, config)
        sorted_freq_power = np.sort(event_power_by_freq)
        concentration_bins = min(5, event_power_by_freq.size)
        event_concentration = float(np.sum(sorted_freq_power[-concentration_bins:]) / total_event_power)
        phase_coherence = _phase_coherence(complex_band, frame_indices, peak_indices)

        if peak_freqs.size:
            peak_powers = event_power_by_freq[peak_indices]
            dominant_peak_idx = int(peak_indices[int(np.argmax(peak_powers))])
            doppler_peak_hz = float(freqs_band[dominant_peak_idx])
            doppler_low_hz = float(np.min(peak_freqs))
            doppler_high_hz = float(np.max(peak_freqs))
        else:
            doppler_peak_hz = float("nan")
            doppler_low_hz = float("nan")
            doppler_high_hz = float("nan")

        quality, rejection_reason = _classify_quality(
            snr_proxy=snr_proxy,
            concentration=event_concentration,
            phase_coherence=phase_coherence,
            width_ms=width_ms,
            center_index=center_index,
            signal_length=n,
            config=config,
        )
        crop_start = int(center_index) - int(config.raw_crop_length) // 2
        crop_end = crop_start + int(config.raw_crop_length)
        candidates.append(
            YeastEvent(
                event_id=f"{split}/{config.class_name}/{sample_id}__yeast{local_idx:03d}",
                sample_id=f"{sample_id}__yeast{local_idx:03d}",
                split=split,
                signal_path=str(signal_path),
                label_path="",
                class_id=int(config.class_id),
                class_name=str(config.class_name),
                center_norm=float(center_index) / float(n),
                width_norm=float(width_samples) / float(n),
                center_index=int(center_index),
                crop_start=int(crop_start),
                crop_end=int(crop_end),
                event_start=int(event_start),
                event_end=int(event_end),
                width_samples=int(width_samples),
                width_ms=float(width_ms),
                snr_proxy=float(snr_proxy),
                energy_concentration=float(event_concentration),
                phase_coherence=float(phase_coherence),
                n_doppler_peaks=int(peak_indices.size),
                doppler_low_hz=float(doppler_low_hz),
                doppler_high_hz=float(doppler_high_hz),
                doppler_peak_hz=float(doppler_peak_hz),
                quality=quality,
                source_group=source_group,
                rejection_reason=rejection_reason,
            )
        )

    candidates.sort(key=lambda e: e.snr_proxy, reverse=True)
    if int(config.max_events_per_signal) > 0:
        candidates = candidates[: int(config.max_events_per_signal)]
    candidates.sort(key=lambda e: (e.center_index, -e.snr_proxy))
    return candidates, ""


def iter_signal_paths(input_dir: Path, include_groups: set[str] | None = None) -> list[Path]:
    paths = sorted(p for p in input_dir.rglob("*.npy") if p.is_file())
    if include_groups:
        paths = [p for p in paths if p.parent.name in include_groups]
    return paths


def keep_event_for_quality(event: YeastEvent, quality: str) -> bool:
    if quality == "all":
        return event.quality in {"strict", "medium"}
    if quality == "medium":
        return event.quality in {"strict", "medium"}
    if quality == "strict":
        return event.quality == "strict"
    raise ValueError(f"Unsupported quality filter: {quality}")


def balanced_visual_indices(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(set(int(v) for v in labels.tolist())):
        idx = np.flatnonzero(labels == class_id)
        if max_per_class > 0 and idx.size > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.extend(int(v) for v in idx.tolist())
    arr = np.asarray(selected, dtype=np.int64)
    arr.sort()
    return arr


def write_event_rows(path: Path, events: list[YeastEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not events:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(events[0]).keys()))
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def write_detection_reports(path: Path, reports: list[FileDetectionReport]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not reports:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(reports[0]).keys()))
        writer.writeheader()
        for report in reports:
            writer.writerow(asdict(report))


def summarize_events(events: list[YeastEvent]) -> dict[str, Any]:
    quality_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for event in events:
        quality_counts[event.quality] = quality_counts.get(event.quality, 0) + 1
        group_counts[event.source_group] = group_counts.get(event.source_group, 0) + 1

    def quantiles(values: Iterable[float]) -> dict[str, float]:
        arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
        if arr.size == 0:
            return {}
        qs = np.quantile(arr, [0.05, 0.25, 0.5, 0.75, 0.95])
        return {name: float(value) for name, value in zip(("p05", "p25", "p50", "p75", "p95"), qs)}

    return {
        "n": int(len(events)),
        "quality_counts": quality_counts,
        "source_group_counts": group_counts,
        "snr_proxy": quantiles(event.snr_proxy for event in events),
        "width_ms": quantiles(event.width_ms for event in events),
        "energy_concentration": quantiles(event.energy_concentration for event in events),
        "phase_coherence": quantiles(event.phase_coherence for event in events),
        "n_doppler_peaks": quantiles(event.n_doppler_peaks for event in events),
    }


def write_audit_pdf(path: Path, events: list[YeastEvent], signals: np.ndarray, max_events: int) -> None:
    if max_events <= 0 or not events:
        return
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for idx, event in enumerate(events[:max_events]):
            raw = ensure_1d_signal(np.load(event.signal_path))
            fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.4), constrained_layout=True)
            t_ms = np.arange(raw.size, dtype=np.float32) / 2_000_000.0 * 1000.0
            axes[0].plot(t_ms, raw, color="black", linewidth=0.6)
            for sample, color, label in [
                (event.event_start, "#009E73", "event"),
                (event.event_end, "#009E73", "event"),
                (event.center_index, "#D55E00", "center"),
                (max(0, event.crop_start), "#0072B2", "crop"),
                (min(raw.size, event.crop_end), "#0072B2", "crop"),
            ]:
                axes[0].axvline(sample / 2_000_000.0 * 1000.0, color=color, linewidth=0.8, alpha=0.85)
            axes[0].set_title(
                f"{event.sample_id} | {event.quality} | snr={event.snr_proxy:.2f}, "
                f"conc={event.energy_concentration:.2f}, peaks={event.n_doppler_peaks}"
            )
            axes[0].set_xlabel("time (ms)")
            axes[0].set_ylabel("raw")

            crop = crop_around_index(raw, event.crop_end - event.crop_start, event.center_index)
            axes[1].plot(crop, color="black", linewidth=0.7)
            axes[1].axvline(crop.size // 2, color="#D55E00", linewidth=0.8)
            axes[1].set_title("Raw crop around selected passage")

            axes[2].plot(signals[idx], color="black", linewidth=0.8)
            axes[2].set_title(f"P3 input: raw {event.crop_end - event.crop_start} -> {signals.shape[1]} -> window z-score")
            axes[2].set_xlabel(f"{signals.shape[1]}-sample index")
            for ax in axes:
                ax.grid(False)
            pdf.savefig(fig)
            plt.close(fig)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    preprocess_mode = str(getattr(args, "preprocess_mode", PREPROCESS_NONE))
    preprocess_config = P1PreprocessConfig(
        mode=preprocess_mode,
        sampling_frequency_hz=float(args.sampling_frequency_hz),
        low_khz=float(getattr(args, "preprocess_low_khz", 5.0)),
        high_khz_max=float(getattr(args, "preprocess_high_khz_max", 100.0)),
        saturation_fmin_hz=float(getattr(args, "saturation_fmin_hz", 7_000.0)),
        saturation_fmax_hz=float(getattr(args, "saturation_fmax_hz", 80_000.0)),
        saturation_min_flat=int(getattr(args, "saturation_min_flat", 500)),
        saturation_zero_threshold=float(getattr(args, "saturation_zero_threshold", 1.0e-4)),
        saturation_guard_before=int(getattr(args, "saturation_guard_before", 0)),
        saturation_guard_after=int(getattr(args, "saturation_guard_after", 0)),
        normalization="window_zscore",
    )
    config = YeastDetectionConfig(
        sampling_frequency_hz=args.sampling_frequency_hz,
        low_freq_hz=args.low_freq_hz,
        high_freq_hz=args.high_freq_hz,
        filter_order=args.filter_order,
        stft_nperseg=args.stft_nperseg,
        stft_noverlap=args.stft_noverlap,
        smooth_frames=args.smooth_frames,
        active_snr_z=args.active_snr_z,
        boundary_snr_z=args.boundary_snr_z,
        medium_min_snr=args.medium_min_snr,
        strict_min_snr=args.strict_min_snr,
        medium_min_concentration=args.medium_min_concentration,
        strict_min_concentration=args.strict_min_concentration,
        strict_min_phase_coherence=args.strict_min_phase_coherence,
        frequency_peak_height_frac=args.frequency_peak_height_frac,
        frequency_peak_prominence_frac=args.frequency_peak_prominence_frac,
        cluster_gap_ms=args.cluster_gap_ms,
        boundary_pad_ms=args.boundary_pad_ms,
        min_width_ms=args.min_width_ms,
        max_width_ms=args.max_width_ms,
        raw_crop_length=args.raw_crop_length,
        output_length=args.output_length,
        max_events_per_signal=args.max_events_per_signal,
        class_id=args.class_id,
        class_name=args.class_name,
    )

    include_groups = set(args.include_groups.split(",")) if args.include_groups else None
    paths = iter_signal_paths(args.input_dir, include_groups=include_groups)
    if args.max_files > 0:
        paths = paths[: args.max_files]
    if not paths:
        raise ValueError(f"No .npy files found under {args.input_dir}")

    all_candidates: list[YeastEvent] = []
    kept_events: list[YeastEvent] = []
    reports: list[FileDetectionReport] = []
    aligned_signals: list[np.ndarray] = []
    preprocess_reports: list[dict[str, Any]] = []

    for path in paths:
        source_group = path.parent.name
        raw = ensure_1d_signal(np.load(path))
        candidates, reason = detect_yeast_passages(
            raw,
            sample_id=path.stem,
            signal_path=str(path),
            source_group=source_group,
            split=args.split,
            config=config,
        )
        all_candidates.extend(candidates)
        selected = [event for event in candidates if keep_event_for_quality(event, args.quality)]
        selected_kept = 0
        for event in selected:
            crop = crop_around_index(raw, args.raw_crop_length, event.center_index)
            report = signal_quality_report(crop, preprocess_config)
            report.update({"event_id": event.event_id, "signal_path": str(path)})
            preprocess_reports.append(report)
            if preprocess_config.mode == PREPROCESS_P1 and not report["ok"]:
                continue
            aligned_signals.append(
                build_aligned_signal_at_center(
                    raw,
                    center_index=event.center_index,
                    raw_crop_length=args.raw_crop_length,
                    output_length=args.output_length,
                    preprocess_config=preprocess_config,
                )
            )
            kept_events.append(event)
            selected_kept += 1
        reports.append(
            FileDetectionReport(
                signal_path=str(path),
                source_group=source_group,
                n_candidates=int(len(candidates)),
                n_kept=int(selected_kept),
                rejection_reason=reason if reason else "",
            )
        )

    if not kept_events:
        raise ValueError(
            f"No yeast events passed quality={args.quality}; inspect candidate_events_metadata.csv "
            "or relax thresholds."
        )

    signals = np.stack(aligned_signals).astype(np.float32)
    labels = np.asarray([event.class_id for event in kept_events], dtype=np.int64)
    split = np.asarray([event.split for event in kept_events])
    event_ids = np.asarray([event.event_id for event in kept_events])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_event_rows(args.output_dir / "candidate_events_metadata.csv", all_candidates)
    write_event_rows(args.output_dir / "events_metadata.csv", kept_events)
    visual_idx = balanced_visual_indices(labels, args.max_plot_per_class, args.seed)
    visual_events = [kept_events[int(i)] for i in visual_idx]
    write_event_rows(args.output_dir / "visual_events_metadata.csv", visual_events)
    write_detection_reports(args.output_dir / "file_detection_report.csv", reports)
    np.savez_compressed(
        args.output_dir / "aligned_inputs.npz",
        signals=signals,
        labels=labels,
        split=split,
        event_id=event_ids,
        center_index=np.asarray([event.center_index for event in kept_events], dtype=np.int64),
        source_path=np.asarray([event.signal_path for event in kept_events]),
        preprocessing_id=np.asarray(preprocess_config.preprocessing_id),
    )
    preprocessing_summary = summarize_quality_reports(preprocess_reports)
    preprocessing_summary["preprocessing"] = preprocess_config.to_dict()
    preprocessing_summary["kept_events_after_preprocessing"] = int(len(kept_events))
    preprocessing_summary["rejected_events_after_preprocessing"] = int(len(preprocess_reports) - len(kept_events))
    with (args.output_dir / "preprocessing_summary.json").open("w") as f:
        json.dump(preprocessing_summary, f, indent=2, sort_keys=True)

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "n_files_scanned": int(len(paths)),
        "quality_filter": args.quality,
        "config": asdict(config),
        "all_candidates": summarize_events(all_candidates),
        "kept_events": summarize_events(kept_events),
        "file_reports": {
            "files_with_candidates": int(sum(1 for report in reports if report.n_candidates > 0)),
            "files_with_kept_events": int(sum(1 for report in reports if report.n_kept > 0)),
            "no_candidate_reasons": {
                reason: int(sum(1 for report in reports if report.rejection_reason == reason))
                for reason in sorted(set(report.rejection_reason for report in reports if report.rejection_reason))
            },
        },
        "preprocessing": preprocessing_summary,
        "input_representation_all_models": (
            f"center on detected yeast passage -> crop raw {int(args.raw_crop_length)} -> "
            f"{int(args.output_length)} -> {preprocess_config.preprocessing_id}"
        ),
    }
    with (args.output_dir / "detection_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    if args.write_audit:
        write_audit_pdf(args.output_dir / "yeast_event_audit.pdf", visual_events, signals[visual_idx], args.audit_max_events)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build P3-compatible yeast event crops from multi-Doppler passage detections.")
    parser.add_argument("--input-dir", type=Path, default=Path("/home/intern/Downloads/Yeast_folder"))
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones" / "yeast_passage_events_p3_4096")
    parser.add_argument("--include-groups", default="", help="Optional comma-separated source folders, e.g. budding,shmoo,shmoo2,mix.")
    parser.add_argument("--quality", choices=("strict", "medium", "all"), default="strict")
    parser.add_argument("--split", default="test")
    parser.add_argument("--class-id", type=int, default=3)
    parser.add_argument("--class-name", default="yeast")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-events-per-signal", type=int, default=3)
    parser.add_argument("--max-plot-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sampling-frequency-hz", type=float, default=2_000_000.0)
    parser.add_argument("--low-freq-hz", type=float, default=7_000.0)
    parser.add_argument("--high-freq-hz", type=float, default=80_000.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--stft-nperseg", type=int, default=512)
    parser.add_argument("--stft-noverlap", type=int, default=384)
    parser.add_argument("--smooth-frames", type=int, default=3)
    parser.add_argument("--active-snr-z", type=float, default=3.5)
    parser.add_argument("--boundary-snr-z", type=float, default=1.5)
    parser.add_argument("--medium-min-snr", type=float, default=3.0)
    parser.add_argument("--strict-min-snr", type=float, default=5.0)
    parser.add_argument("--medium-min-concentration", type=float, default=0.08)
    parser.add_argument("--strict-min-concentration", type=float, default=0.12)
    parser.add_argument("--strict-min-phase-coherence", type=float, default=0.0)
    parser.add_argument("--frequency-peak-height-frac", type=float, default=0.20)
    parser.add_argument("--frequency-peak-prominence-frac", type=float, default=0.08)
    parser.add_argument("--cluster-gap-ms", type=float, default=0.25)
    parser.add_argument("--boundary-pad-ms", type=float, default=0.04)
    parser.add_argument("--min-width-ms", type=float, default=0.06)
    parser.add_argument("--max-width-ms", type=float, default=1.60)
    parser.add_argument("--raw-crop-length", type=int, default=4096)
    parser.add_argument("--output-length", type=int, default=4096)
    parser.add_argument("--preprocess-mode", choices=PREPROCESS_MODES, default=PREPROCESS_NONE)
    parser.add_argument("--preprocess-low-khz", type=float, default=5.0)
    parser.add_argument("--preprocess-high-khz-max", type=float, default=100.0)
    parser.add_argument("--saturation-fmin-hz", type=float, default=7_000.0)
    parser.add_argument("--saturation-fmax-hz", type=float, default=80_000.0)
    parser.add_argument("--saturation-min-flat", type=int, default=500)
    parser.add_argument("--saturation-zero-threshold", type=float, default=1.0e-4)
    parser.add_argument("--saturation-guard-before", type=int, default=0)
    parser.add_argument("--saturation-guard-after", type=int, default=0)
    parser.add_argument("--write-audit", action="store_true")
    parser.add_argument("--audit-max-events", type=int, default=48)
    return parser


def main() -> None:
    summary = build_dataset(build_parser().parse_args())
    print(json.dumps({"output_dir": summary["output_dir"], "kept_events": summary["kept_events"]["n"]}, sort_keys=True))


if __name__ == "__main__":
    main()
