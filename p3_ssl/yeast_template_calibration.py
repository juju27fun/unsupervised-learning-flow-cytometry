#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, hilbert

ROOT = Path(__file__).resolve().parents[1]
from p3_ssl.decimation import normalize_signal
from p3_ssl.particle_equation_sweeps import (
    DEFAULT_YEAST_RANGE_SUMMARY,
    WINDOW_DURATION_MS,
    _load_yeast_range_summary,
    _summary_stat,
    particle_wave,
)

DEFAULT_YEAST_ROOT = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones-4096_20260701" / "yeast_passage_events_p3_4096"
DEFAULT_OUTPUT_DIR = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "yeast_template_generator_calibration"
METRIC_KEYS = (
    "edge_std",
    "support_frac",
    "envelope_peak_count",
    "zero_crossings",
    "spectral_centroid",
    "spectral_bandwidth",
    "envelope_asymmetry",
)
SCORE_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
FALLBACK_TEMPLATE_PARAMS = {
    "template_envelope_strength": 0.40,
    "texture_strength": 0.18,
    "low_frequency_template_strength": 0.35,
    "noise_scale": 0.75,
    "tau_width_divisor": 4.20,
    "doppler_scale": 0.78,
    "common_t0_jitter": 0.025,
    "asymmetry_strength": 0.12,
    "secondary_doppler_strength": 0.12,
}


def _read_events(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_yeast_templates(
    yeast_root: Path,
    *,
    source_group: str = "budding",
    max_events: int = 0,
    seed: int = 42,
) -> tuple[np.ndarray, list[dict[str, str]], np.ndarray]:
    events = _read_events(yeast_root / "events_metadata.csv")
    with np.load(yeast_root / "aligned_inputs.npz", allow_pickle=True) as data:
        signals = data["signals"].astype(np.float32, copy=False)
    if len(events) != int(signals.shape[0]):
        raise ValueError(f"events/signals length mismatch: {len(events)} vs {signals.shape[0]}")

    indices = [
        i
        for i, row in enumerate(events)
        if row.get("source_group", "") == source_group and row.get("quality", "strict") == "strict"
    ]
    if not indices:
        indices = [i for i, row in enumerate(events) if row.get("source_group", "") == source_group]
    if max_events > 0 and len(indices) > max_events:
        rng = np.random.default_rng(seed)
        indices = sorted(int(i) for i in rng.choice(np.asarray(indices), size=max_events, replace=False).tolist())
    if not indices:
        raise ValueError(f"No yeast events selected for source_group={source_group!r}")
    selected = np.asarray(indices, dtype=np.int64)
    return signals[selected].astype(np.float32, copy=False), [events[i] for i in selected.tolist()], selected


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    size = max(3, int(size) | 1)
    x = np.arange(size, dtype=np.float64) - size // 2
    kernel = np.exp(-0.5 * np.square(x / max(float(sigma), 1.0e-6)))
    kernel /= max(float(np.sum(kernel)), 1.0e-12)
    return kernel.astype(np.float32)


def smooth_envelope(signal: np.ndarray, sigma_fraction: float = 0.020) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    envelope = np.abs(hilbert(x.astype(np.float64))).astype(np.float32)
    sigma = max(2.0, float(x.size) * float(sigma_fraction))
    kernel = _gaussian_kernel(int(round(8.0 * sigma)), sigma)
    smoothed = np.convolve(envelope, kernel, mode="same").astype(np.float32)
    return smoothed / max(float(np.max(smoothed)), 1.0e-6)


def smooth_signal(signal: np.ndarray, sigma_fraction: float = 0.040) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    sigma = max(2.0, float(x.size) * float(sigma_fraction))
    kernel = _gaussian_kernel(int(round(8.0 * sigma)), sigma)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def compute_signal_metrics(
    signals: np.ndarray,
    *,
    edge_fraction: float = 0.15,
    support_threshold: float = 0.15,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    n_edge = max(8, int(round(signals.shape[1] * float(edge_fraction))))
    freqs = np.fft.rfftfreq(signals.shape[1], d=1.0 / signals.shape[1]).astype(np.float64)
    for row in np.asarray(signals, dtype=np.float32):
        x = np.asarray(row, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        edge = np.concatenate([x[:n_edge], x[-n_edge:]])
        env = smooth_envelope(x.astype(np.float32))
        peaks, _ = find_peaks(env, distance=max(4, int(round(0.05 * x.size))), prominence=0.06)
        centered = x - float(np.mean(x))
        power = np.square(np.abs(np.fft.rfft(centered)))
        power[0] = 0.0
        total_power = float(np.sum(power))
        if total_power > 1.0e-12:
            centroid = float(np.sum(freqs * power) / total_power)
            bandwidth = float(np.sqrt(np.sum(np.square(freqs - centroid) * power) / total_power))
        else:
            centroid = 0.0
            bandwidth = 0.0
        mid = x.size // 2
        left_mass = float(np.sum(env[:mid]))
        right_mass = float(np.sum(env[mid:]))
        rows.append(
            {
                "edge_std": float(np.std(edge)),
                "support_frac": float(np.mean(env > support_threshold)),
                "envelope_peak_count": float(peaks.size),
                "zero_crossings": float(np.count_nonzero(np.diff(np.signbit(centered)))),
                "spectral_centroid": centroid,
                "spectral_bandwidth": bandwidth,
                "envelope_asymmetry": float((right_mass - left_mass) / max(right_mass + left_mass, 1.0e-12)),
            }
        )
    return rows


def summarize_metrics(metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = np.asarray([row[key] for row in metrics], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary[key] = {}
            continue
        qs = np.quantile(values, SCORE_QUANTILES)
        summary[key] = {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p05": float(qs[0]),
            "p25": float(qs[1]),
            "p50": float(qs[2]),
            "p75": float(qs[3]),
            "p95": float(qs[4]),
        }
    return summary


def score_metric_summary(real_summary: dict[str, dict[str, float]], generated_summary: dict[str, dict[str, float]]) -> float:
    losses: list[float] = []
    weights = {
        "edge_std": 1.55,
        "support_frac": 1.65,
        "zero_crossings": 1.45,
        "spectral_centroid": 1.20,
        "spectral_bandwidth": 1.10,
        "envelope_peak_count": 1.10,
        "envelope_asymmetry": 0.65,
    }
    for key in METRIC_KEYS:
        real = real_summary.get(key, {})
        generated = generated_summary.get(key, {})
        if not real or not generated:
            continue
        real_vec = np.asarray([real[f"p{int(q * 100):02d}"] for q in SCORE_QUANTILES], dtype=np.float64)
        gen_vec = np.asarray([generated[f"p{int(q * 100):02d}"] for q in SCORE_QUANTILES], dtype=np.float64)
        scale = max(float(real.get("p75", 0.0) - real.get("p25", 0.0)), abs(float(real.get("p50", 0.0))) * 0.10, 1.0e-3)
        losses.append(float(weights.get(key, 1.0) * np.mean(np.abs(gen_vec - real_vec) / scale)))
    normalizer = float(sum(weights.get(key, 1.0) for key in METRIC_KEYS if real_summary.get(key) and generated_summary.get(key)))
    return float(np.sum(losses) / max(normalizer, 1.0e-6)) if losses else float("inf")


def _background_noise(rng: np.random.Generator, length: int, std: float) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=length).astype(np.float32)
    kernel = _gaussian_kernel(41, 5.5)
    colored = np.convolve(white, kernel, mode="same").astype(np.float32)
    mixed = 0.22 * white + 0.78 * colored
    mixed /= max(float(np.std(mixed)), 1.0e-6)
    return (mixed * float(std)).astype(np.float32)


def generate_template_candidate_signals(
    *,
    rng: np.random.Generator,
    templates: np.ndarray,
    range_summary: dict[str, Any],
    params: dict[str, float],
    n_samples: int,
    window_duration_ms: float,
    normalization: str = "window_zscore",
) -> np.ndarray:
    length = int(templates.shape[1])
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    edge_std = _summary_stat(range_summary, "background_edge_std", "p50", 0.51) * float(params["noise_scale"])
    width_ms = _summary_stat(range_summary, "width_ms", "p50", 1.0)
    tau = np.clip((width_ms / float(params["tau_width_divisor"])) / float(window_duration_ms), 0.035, 0.260)
    amp = _summary_stat(range_summary, "component_a_amplitude", "p50", 1.8)
    amp_low = _summary_stat(range_summary, "amplitude_ratio_b_over_a", "p05", 0.55)
    amp_high = _summary_stat(range_summary, "amplitude_ratio_b_over_a", "p95", 1.65)
    dt_low = _summary_stat(range_summary, "delta_t0", "p05", 0.012)
    dt_high = _summary_stat(range_summary, "delta_t0", "p75", 0.080)
    phase_high = min(np.pi, _summary_stat(range_summary, "abs_delta_phi_rad", "p95", 3.05))
    cycles = (
        _summary_stat(range_summary, "doppler_peak_hz", "p50", 17_500.0)
        / 1000.0
        * float(window_duration_ms)
        * float(params.get("doppler_scale", FALLBACK_TEMPLATE_PARAMS["doppler_scale"]))
    )
    cycles_spread = max(1.0, 0.18 * cycles)

    outputs = np.empty((n_samples, length), dtype=np.float32)
    template_indices = rng.integers(0, templates.shape[0], size=n_samples)
    for i, template_idx in enumerate(template_indices.tolist()):
        template = templates[template_idx].astype(np.float32, copy=False)
        template_norm = normalize_signal(template, mode="window_zscore").astype(np.float32)
        template_env = smooth_envelope(template_norm)
        template_low = smooth_signal(template_norm, sigma_fraction=0.045)
        template_low = normalize_signal(template_low, mode="window_zscore").astype(np.float32)
        delta_t = float(rng.uniform(dt_low, max(dt_low + 1.0e-4, dt_high)))
        ratio = float(rng.uniform(max(0.05, amp_low), max(amp_low + 1.0e-3, amp_high)))
        phase_delta = float(rng.uniform(-phase_high, phase_high))
        freq_delta = float(rng.normal(0.0, cycles_spread))
        tau_ratio = float(rng.uniform(0.75, 1.35))
        common_shift = float(
            np.clip(
                rng.normal(0.0, float(params.get("common_t0_jitter", FALLBACK_TEMPLATE_PARAMS["common_t0_jitter"]))),
                -0.085,
                0.085,
            )
        )
        t0_a = np.clip(0.5 + common_shift - 0.5 * delta_t, 0.12, 0.88)
        t0_b = np.clip(0.5 + common_shift + 0.5 * delta_t, 0.12, 0.88)
        signal_a = particle_wave(
            t,
            np.asarray([amp], dtype=np.float32),
            np.asarray([cycles - 0.5 * freq_delta], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([t0_a], dtype=np.float32),
            np.asarray([tau], dtype=np.float32),
        )[0]
        signal_b = particle_wave(
            t,
            np.asarray([amp * ratio], dtype=np.float32),
            np.asarray([cycles + 0.5 * freq_delta], dtype=np.float32),
            np.asarray([phase_delta], dtype=np.float32),
            np.asarray([t0_b], dtype=np.float32),
            np.asarray([tau * tau_ratio], dtype=np.float32),
        )[0]
        carrier = signal_a + signal_b
        secondary = float(params["secondary_doppler_strength"]) * particle_wave(
            t,
            np.asarray([amp], dtype=np.float32),
            np.asarray([1.35 * cycles], dtype=np.float32),
            np.asarray([0.8 + phase_delta], dtype=np.float32),
            np.asarray([0.5 + 0.05 * float(params["asymmetry_strength"])], dtype=np.float32),
            np.asarray([tau * 1.45], dtype=np.float32),
        )[0]
        real_enveloped = (carrier + secondary) * (
            (1.0 - float(params["template_envelope_strength"])) + float(params["template_envelope_strength"]) * template_env
        )
        template_mix = (
            float(params["texture_strength"]) * template_norm
            + float(params.get("low_frequency_template_strength", FALLBACK_TEMPLATE_PARAMS["low_frequency_template_strength"])) * template_low
        )
        textured = real_enveloped + float(np.std(real_enveloped)) * template_mix
        noisy = textured + _background_noise(rng, length, edge_std)
        outputs[i] = normalize_signal(noisy, mode=normalization).astype(np.float32)
    return outputs


def sample_candidate_params(rng: np.random.Generator) -> dict[str, float]:
    return {
        "template_envelope_strength": float(rng.uniform(0.15, 0.70)),
        "texture_strength": float(rng.uniform(0.08, 0.55)),
        "low_frequency_template_strength": float(rng.uniform(0.05, 0.90)),
        "noise_scale": float(rng.uniform(0.35, 1.75)),
        "tau_width_divisor": float(rng.uniform(3.2, 5.8)),
        "doppler_scale": float(rng.uniform(0.52, 1.02)),
        "common_t0_jitter": float(rng.uniform(0.00, 0.055)),
        "asymmetry_strength": float(rng.uniform(0.00, 0.30)),
        "secondary_doppler_strength": float(rng.uniform(0.00, 0.25)),
    }


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    templates, events, selected_indices = load_yeast_templates(
        args.yeast_root,
        source_group=args.source_group,
        max_events=args.max_real_events,
        seed=args.seed,
    )
    real_metrics = compute_signal_metrics(templates, edge_fraction=args.edge_fraction)
    real_summary = summarize_metrics(real_metrics)
    range_summary = _load_yeast_range_summary(args.range_summary)

    candidate_rows: list[dict[str, float]] = []
    best_score = float("inf")
    best_params = dict(FALLBACK_TEMPLATE_PARAMS)
    best_summary: dict[str, dict[str, float]] = {}
    best_examples: np.ndarray | None = None
    for candidate_idx in range(int(args.n_candidates)):
        params = sample_candidate_params(rng)
        generated = generate_template_candidate_signals(
            rng=rng,
            templates=templates,
            range_summary=range_summary,
            params=params,
            n_samples=int(args.samples_per_candidate),
            window_duration_ms=float(args.window_duration_ms),
            normalization=args.normalization,
        )
        generated_summary = summarize_metrics(compute_signal_metrics(generated, edge_fraction=args.edge_fraction))
        score = score_metric_summary(real_summary, generated_summary)
        candidate_rows.append({"candidate": float(candidate_idx), "score": score, **params})
        if score < best_score:
            best_score = score
            best_params = params
            best_summary = generated_summary
            best_examples = generated[: min(12, generated.shape[0])].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": {
            "yeast_root": str(args.yeast_root),
            "range_summary": str(args.range_summary),
            "source_group": args.source_group,
            "selected_indices_n": int(selected_indices.size),
            "selected_event_examples": [row.get("event_id", "") for row in events[:8]],
            "window_duration_ms": float(args.window_duration_ms),
            "edge_fraction": float(args.edge_fraction),
            "normalization": args.normalization,
        },
        "real_metric_summary": real_summary,
        "best_generated_metric_summary": best_summary,
        "best_params": best_params,
        "best_score": float(best_score),
        "n_candidates": int(args.n_candidates),
        "samples_per_candidate": int(args.samples_per_candidate),
        "score_note": "Mean normalized quantile distance between real budding crops and generated candidates.",
    }
    with (args.output_dir / "budding_template_calibration_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with (args.output_dir / "candidate_scores.csv").open("w", newline="") as f:
        fieldnames = ["candidate", "score", *FALLBACK_TEMPLATE_PARAMS.keys()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(candidate_rows, key=lambda item: item["score"]):
            writer.writerow(row)
    if best_examples is not None:
        plot_calibration_pdf(
            real_signals=templates,
            generated_signals=best_examples,
            real_summary=real_summary,
            generated_summary=best_summary,
            output_pdf=args.output_dir / "real_vs_generated_calibration.pdf",
            output_png=args.output_dir / "real_vs_generated_calibration.png",
            seed=args.seed,
        )
    return summary


def plot_calibration_pdf(
    *,
    real_signals: np.ndarray,
    generated_signals: np.ndarray,
    real_summary: dict[str, dict[str, float]],
    generated_summary: dict[str, dict[str, float]],
    output_pdf: Path,
    output_png: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    n_examples = min(6, real_signals.shape[0], generated_signals.shape[0])
    real_idx = rng.choice(np.arange(real_signals.shape[0]), size=n_examples, replace=False)
    x = np.linspace(0.0, WINDOW_DURATION_MS, real_signals.shape[1], dtype=np.float32)
    fig, axes = plt.subplots(3, n_examples, figsize=(15.4, 8.2), squeeze=False)
    for col in range(n_examples):
        axes[0, col].plot(x, real_signals[int(real_idx[col])], color="#00876c", linewidth=0.75)
        axes[0, col].set_title("real budding", fontsize=8)
        axes[1, col].plot(x, generated_signals[col], color="#1f77b4", linewidth=0.75)
        axes[1, col].set_title("generated template", fontsize=8)
        for row in (0, 1):
            axes[row, col].tick_params(labelsize=6, length=2)
            if col == 0:
                axes[row, col].set_ylabel("signal", fontsize=8)
            if row == 1:
                axes[row, col].set_xlabel("Time [ms]", fontsize=7)
    labels = list(METRIC_KEYS)
    real_medians = [real_summary[key].get("p50", np.nan) for key in labels]
    generated_medians = [generated_summary[key].get("p50", np.nan) for key in labels]
    for col in range(n_examples):
        axes[2, col].remove()
    ax = fig.add_subplot(3, 1, 3)
    pos = np.arange(len(labels))
    ax.bar(pos - 0.18, real_medians, width=0.36, label="real p50", color="#00876c", alpha=0.85)
    ax.bar(pos + 0.18, generated_medians, width=0.36, label="generated p50", color="#1f77b4", alpha=0.85)
    ax.set_xticks(pos, labels, rotation=20, ha="right", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Yeast budding template-generator calibration", fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.92, bottom=0.12, hspace=0.52, wspace=0.26)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=300)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate template-based yeast budding synthetic signal generator.")
    parser.add_argument("--yeast-root", type=Path, default=DEFAULT_YEAST_ROOT)
    parser.add_argument("--range-summary", type=Path, default=DEFAULT_YEAST_RANGE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-group", default="budding")
    parser.add_argument("--max-real-events", type=int, default=0)
    parser.add_argument("--n-candidates", type=int, default=300)
    parser.add_argument("--samples-per-candidate", type=int, default=256)
    parser.add_argument("--window-duration-ms", type=float, default=WINDOW_DURATION_MS)
    parser.add_argument("--edge-fraction", type=float, default=0.15)
    parser.add_argument("--normalization", choices=["none", "window_zscore", "robust_zscore"], default="window_zscore")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    summary = calibrate(build_parser().parse_args())
    print(json.dumps({"best_score": summary["best_score"], "best_params": summary["best_params"]}, sort_keys=True))


if __name__ == "__main__":
    main()
