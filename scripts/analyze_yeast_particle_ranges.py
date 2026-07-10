#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, hilbert, peak_widths


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YEAST_ROOT = ROOT / "outputs" / "pretrained_backbones-4096_20260701" / "yeast_passage_events_p3_4096"


def _finite_quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    qs = np.quantile(arr, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p01": float(qs[0]),
        "p05": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p95": float(qs[5]),
        "p99": float(qs[6]),
        "max": float(np.max(arr)),
    }


def _read_events(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _parse_groups(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _dominant_cycles_per_window(signal: np.ndarray) -> float:
    x = np.asarray(signal, dtype=np.float64)
    x = x - float(np.mean(x))
    if x.size < 4:
        return float("nan")
    spectrum = np.abs(np.fft.rfft(x))
    if spectrum.size <= 1:
        return float("nan")
    spectrum[0] = 0.0
    return float(np.argmax(spectrum))


def _local_width_norm(envelope: np.ndarray, peak_index: int) -> float:
    try:
        widths, *_ = peak_widths(envelope, np.asarray([peak_index], dtype=np.int64), rel_height=0.5)
    except Exception:
        return float("nan")
    if widths.size == 0:
        return float("nan")
    # FWHM ~= 2.355 sigma for a Gaussian envelope.
    return float(widths[0] / 2.355 / envelope.size)


def estimate_two_component_row(
    signal: np.ndarray,
    *,
    window_duration_ms: float,
    min_peak_distance_ms: float,
    smooth_ms: float,
    min_prominence_fraction: float,
) -> dict[str, float]:
    x = np.asarray(signal, dtype=np.float32)
    analytic = hilbert(x.astype(np.float64))
    envelope = np.abs(analytic)
    smooth_samples = max(1, int(round(float(smooth_ms) / float(window_duration_ms) * x.size)))
    if smooth_samples > 1:
        envelope_smooth = uniform_filter1d(envelope, size=smooth_samples, mode="nearest")
    else:
        envelope_smooth = envelope

    distance = max(1, int(round(float(min_peak_distance_ms) / float(window_duration_ms) * x.size)))
    prominence = max(1.0e-12, float(min_prominence_fraction) * float(np.max(envelope_smooth)))
    peaks, props = find_peaks(envelope_smooth, distance=distance, prominence=prominence)
    if peaks.size < 2:
        peaks, props = find_peaks(envelope_smooth, distance=distance)
    if peaks.size < 2:
        return {"two_peak_found": 0.0}

    order = np.argsort(envelope_smooth[peaks])[::-1][:2]
    selected = np.sort(peaks[order].astype(np.int64))
    left, right = int(selected[0]), int(selected[1])
    amp_left = float(envelope_smooth[left])
    amp_right = float(envelope_smooth[right])
    phase = np.unwrap(np.angle(analytic))
    delta_phi = float(np.angle(np.exp(1j * (phase[right] - phase[left]))))
    delta_t0 = float((right - left) / x.size)
    delta_t0_ms = float(delta_t0 * window_duration_ms)
    width_left = _local_width_norm(envelope_smooth, left)
    width_right = _local_width_norm(envelope_smooth, right)
    cycles = _dominant_cycles_per_window(x)

    return {
        "two_peak_found": 1.0,
        "component_a_index": float(left),
        "component_b_index": float(right),
        "component_a_t0": float(left / x.size),
        "component_b_t0": float(right / x.size),
        "delta_t0": delta_t0,
        "delta_t0_ms": delta_t0_ms,
        "component_a_amplitude": amp_left,
        "component_b_amplitude": amp_right,
        "amplitude_ratio_b_over_a": float(amp_right / amp_left) if amp_left > 0.0 else float("nan"),
        "delta_amplitude": float(amp_right - amp_left),
        "delta_phi_rad": delta_phi,
        "abs_delta_phi_rad": abs(delta_phi),
        "component_a_tau": width_left,
        "component_b_tau": width_right,
        "delta_tau": float(width_right - width_left) if math.isfinite(width_left) and math.isfinite(width_right) else float("nan"),
        "tau_ratio_b_over_a": float(width_right / width_left) if math.isfinite(width_left) and width_left > 0.0 else float("nan"),
        "dominant_cycles_per_window": cycles,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    events = _read_events(args.events_metadata)
    with np.load(args.aligned_inputs, allow_pickle=True) as data:
        signals = data["signals"].astype(np.float32, copy=False)

    if len(events) != int(signals.shape[0]):
        raise ValueError(f"events/signals length mismatch: {len(events)} vs {signals.shape[0]}")

    groups = _parse_groups(args.source_groups)
    indices = [i for i, row in enumerate(events) if not groups or row.get("source_group", "") in groups]
    if args.max_events > 0 and len(indices) > args.max_events:
        rng = np.random.default_rng(args.seed)
        indices = sorted(int(i) for i in rng.choice(np.asarray(indices), size=args.max_events, replace=False).tolist())
    if not indices:
        raise ValueError("No yeast events selected")

    rows: list[dict[str, Any]] = []
    edge_len = max(64, int(round(signals.shape[1] * float(args.edge_fraction))))
    for i in indices:
        event = events[i]
        edge = np.concatenate([signals[i, :edge_len], signals[i, -edge_len:]]).astype(np.float64, copy=False)
        edge_median = float(np.median(edge))
        edge_mad_std = float(np.median(np.abs(edge - edge_median)) * 1.4826)
        estimate = estimate_two_component_row(
            signals[i],
            window_duration_ms=args.window_duration_ms,
            min_peak_distance_ms=args.min_peak_distance_ms,
            smooth_ms=args.smooth_ms,
            min_prominence_fraction=args.min_prominence_fraction,
        )
        rows.append(
            {
                "row_index": i,
                "event_id": event.get("event_id", ""),
                "sample_id": event.get("sample_id", ""),
                "source_group": event.get("source_group", ""),
                "quality": event.get("quality", ""),
                "width_ms": float(event.get("width_ms", "nan")),
                "snr_proxy": float(event.get("snr_proxy", "nan")),
                "energy_concentration": float(event.get("energy_concentration", "nan")),
                "phase_coherence": float(event.get("phase_coherence", "nan")),
                "n_doppler_peaks": float(event.get("n_doppler_peaks", "nan")),
                "doppler_peak_hz": float(event.get("doppler_peak_hz", "nan")),
                "background_edge_std": float(np.std(edge)),
                "background_edge_mad_std": edge_mad_std,
                "background_edge_rms": float(np.sqrt(np.mean(np.square(edge)))),
                **estimate,
            }
        )

    keys = [
        "width_ms",
        "snr_proxy",
        "energy_concentration",
        "phase_coherence",
        "n_doppler_peaks",
        "doppler_peak_hz",
        "background_edge_std",
        "background_edge_mad_std",
        "background_edge_rms",
        "delta_t0",
        "delta_t0_ms",
        "component_a_amplitude",
        "component_b_amplitude",
        "amplitude_ratio_b_over_a",
        "delta_amplitude",
        "delta_phi_rad",
        "abs_delta_phi_rad",
        "component_a_tau",
        "component_b_tau",
        "delta_tau",
        "tau_ratio_b_over_a",
        "dominant_cycles_per_window",
    ]
    found_rows = [row for row in rows if float(row.get("two_peak_found", 0.0)) > 0.0]
    summary = {
        "source": {
            "events_metadata": str(args.events_metadata),
            "aligned_inputs": str(args.aligned_inputs),
            "source_groups": sorted(groups) if groups else "all",
            "window_duration_ms": float(args.window_duration_ms),
            "min_peak_distance_ms": float(args.min_peak_distance_ms),
            "smooth_ms": float(args.smooth_ms),
            "min_prominence_fraction": float(args.min_prominence_fraction),
            "edge_fraction": float(args.edge_fraction),
            "edge_len_samples": int(edge_len),
        },
        "n_selected": int(len(rows)),
        "n_two_peak_found": int(len(found_rows)),
        "two_peak_fraction": float(len(found_rows) / len(rows)),
        "all_selected": {key: _finite_quantiles([row.get(key, float("nan")) for row in rows]) for key in keys},
        "two_peak_only": {key: _finite_quantiles([row.get(key, float("nan")) for row in found_rows]) for key in keys},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "yeast_two_component_range_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (args.output_dir / "yeast_two_component_estimates.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate two-close-component parameter ranges from yeast event crops.")
    parser.add_argument("--events-metadata", type=Path, default=DEFAULT_YEAST_ROOT / "events_metadata.csv")
    parser.add_argument("--aligned-inputs", type=Path, default=DEFAULT_YEAST_ROOT / "aligned_inputs.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "yeast_particle_range_analysis")
    parser.add_argument("--source-groups", default="", help="Optional comma-separated groups, e.g. budding or budding,mix.")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--window-duration-ms", type=float, default=2.048)
    parser.add_argument("--min-peak-distance-ms", type=float, default=0.025)
    parser.add_argument("--smooth-ms", type=float, default=0.012)
    parser.add_argument("--min-prominence-fraction", type=float, default=0.08)
    parser.add_argument("--edge-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    summary = analyze(build_parser().parse_args())
    print(
        json.dumps(
            {
                "n_selected": summary["n_selected"],
                "n_two_peak_found": summary["n_two_peak_found"],
                "two_peak_fraction": summary["two_peak_fraction"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
