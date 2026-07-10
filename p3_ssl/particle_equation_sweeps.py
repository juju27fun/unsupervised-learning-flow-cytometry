#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
from p3_ssl import backbone_benchmark as backbone
from p3_ssl.decimation import normalize_signal
from p3_ssl.pretrained_backbones import MOMENT_DEFAULT_ID, PATCHTST_DEFAULT_ID, ParticleEvent
from p3_ssl.signal_preprocessing import PREPROCESS_MODES, PREPROCESS_NONE, P1PreprocessConfig, preprocess_batch


CONV_MODEL_KEY = "conv1dgap_same_input_3class"
DEFAULT_CONV_CHECKPOINT = (
    REPO_ROOT
    / "artifacts"
    / "unsupervised-learning-flow-cytometry"
    / "pretrained_backbones"
    / "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap"
    / CONV_MODEL_KEY
    / "best_model.pt"
)
MODEL_KEYS = ("moment_official", "patchtst_pretrained", CONV_MODEL_KEY)
MODEL_DISPLAY = {
    "moment_official": "MOMENT frozen pretrained",
    "patchtst_pretrained": "PatchTST frozen pretrained",
    CONV_MODEL_KEY: "Conv1D-GAP-L supervised same-input",
}
WINDOW_DURATION_MS = 2.048
RASTERIZED_PDF_DPI = 450
PNG_DPI = 260
DEFAULT_PARTICLE_MANIFEST = (
    REPO_ROOT
    / "artifacts"
    / "particles2SNR-pipeline"
    / "runs"
    / "p0_c1_Particles2SNR_F"
    / "event_classification_dataset"
    / "event_manifest.csv"
)
DEFAULT_YEAST_RANGE_SUMMARY = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "yeast_particle_range_analysis_budding" / "yeast_two_component_range_summary.json"
DEFAULT_YEAST_EVENTS_METADATA = (
    REPO_ROOT
    / "artifacts"
    / "unsupervised-learning-flow-cytometry"
    / "pretrained_backbones-4096_20260701"
    / "yeast_passage_events_p3_4096"
    / "events_metadata.csv"
)
DEFAULT_YEAST_TEMPLATE_ROOT = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones-4096_20260701" / "yeast_passage_events_p3_4096"
DEFAULT_YEAST_TEMPLATE_CALIBRATION = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "yeast_template_generator_calibration" / "budding_template_calibration_summary.json"
GAUSSIAN_FWHM_TO_SIGMA = 2.355
SNR_SWEEP_QUANTILES_DB = {
    "q20": -7.809031394293547,
    "q50": 1.1707388192463757,
    "q80": 5.953416315296055,
}
SNR_SWEEP_SOURCE = "outputs/snr_metric_figures/particles visual_subset q20-q80"
REALISTIC_TAU_BASE_MS = 0.16
REALISTIC_TAU_SWEEP_MS = (0.05, 0.25)
YEAST_BUDDED_STYLES = ("original", "budding_intermediate", "budding_noisy", "budding_realistic", "template_budding")
YEAST_TEMPLATE_FALLBACK_PARAMS = {
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

SINGLE_BASE = {"A": 1.0, "fD": 16.0, "phi": 0.0, "t0": 0.5, "tau": 0.06}
PAPER_TABLE_SINGLE_BASE = {
    "A": 0.43,
    "fD": 23.91 * WINDOW_DURATION_MS,
    "phi": 0.0,
    "t0": 0.5,
    "tau": (0.73 / GAUSSIAN_FWHM_TO_SIGMA) / WINDOW_DURATION_MS,
}
PAPER_TABLE_PHASE_VISUAL_BASE = {
    "A": 0.70,
    "fD": 8.00 * WINDOW_DURATION_MS,
    "phi": 0.0,
    "t0": 0.5,
    "tau": (0.33 / GAUSSIAN_FWHM_TO_SIGMA) / WINDOW_DURATION_MS,
}
PAPER_TABLE_STATS = {
    "source": "User-provided paper Table 3.4 descriptive OFI feature statistics",
    "feature_units": {
        "doppler_frequency": "kHz",
        "amplitude_time": "mV",
        "passage_time": "ms",
    },
    "by_particle_size": {
        "2um": {
            "doppler_frequency_khz": {"mean": 22.06, "std": 7.53, "min": 8.00, "max": 36.80},
            "amplitude_time_mv": {"mean": 0.18, "std": 0.04, "min": 0.10, "max": 0.35},
            "passage_time_ms": {"mean": 0.46, "std": 0.15, "min": 0.33, "max": 0.73},
        },
        "4um": {
            "doppler_frequency_khz": {"mean": 30.37, "std": 5.35, "min": 12.00, "max": 37.60},
            "amplitude_time_mv": {"mean": 0.41, "std": 0.10, "min": 0.16, "max": 0.86},
            "passage_time_ms": {"mean": 0.78, "std": 0.19, "min": 0.45, "max": 0.976},
        },
        "10um": {
            "doppler_frequency_khz": {"mean": 19.30, "std": 4.09, "min": 8.80, "max": 29.60},
            "amplitude_time_mv": {"mean": 0.70, "std": 0.57, "min": 0.19, "max": 3.56},
            "passage_time_ms": {"mean": 0.95, "std": 0.31, "min": 0.61, "max": 1.10},
        },
    },
    "global_min_max": {
        "doppler_frequency_khz": {"min": 8.00, "max": 37.60},
        "amplitude_time_mv": {"min": 0.10, "max": 3.56},
        "passage_time_ms": {"min": 0.33, "max": 1.10},
    },
}
TWO_BASE = {
    "A": 1.0,
    "B": 1.0,
    "fDA": 16.0,
    "fDB": 16.0,
    "phiA": 0.0,
    "phiB": 0.0,
    "t0A": 0.5,
    "t0B": 0.5,
    "tauA": 0.06,
    "tauB": 0.06,
}
YEAST_BUDDED_FALLBACK_RANGES = {
    "amplitude_ratio_b_over_a": {"p05": 0.55, "p50": 0.95, "p95": 1.65},
    "delta_t0": {"p05": 0.012, "p50": 0.055, "p95": 0.180},
    "abs_delta_phi_rad": {"p05": 0.15, "p50": 1.20, "p95": 3.05},
    "tau_ratio_b_over_a": {"p05": 0.55, "p50": 1.00, "p95": 1.90},
    "component_a_amplitude": {"p05": 0.8, "p50": 1.8, "p95": 3.8},
    "component_a_tau": {"p05": 0.006, "p50": 0.025, "p95": 0.080},
    "component_b_tau": {"p05": 0.006, "p50": 0.025, "p95": 0.080},
    "width_ms": {"p05": 0.592, "p50": 0.976, "p95": 1.424},
    "doppler_peak_hz": {"p05": 7_812.5, "p50": 19_531.25, "p95": 23_437.5},
    "dominant_cycles_per_window": {"p05": 16.0, "p50": 36.0, "p95": 60.0},
    "snr_proxy": {"p05": 5.0, "p50": 20.0, "p95": 350.0},
    "background_edge_std": {"p05": 0.30, "p50": 0.51, "p95": 1.07},
}


@dataclass(frozen=True)
class ParticleEquationPanel:
    key: str
    title: str
    sweep_param: str
    signal: np.ndarray
    encoded_signal: np.ndarray
    color_value: np.ndarray
    params: dict[str, np.ndarray]
    window_duration_ms: float = WINDOW_DURATION_MS


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _linspace_param(low: float, high: float, n: int, rng: np.random.Generator, shuffle: bool) -> np.ndarray:
    values = np.linspace(low, high, n, dtype=np.float32)
    if shuffle:
        rng.shuffle(values)
    return values


def _noise(rng: np.random.Generator, shape: tuple[int, int], std: float) -> np.ndarray:
    if std <= 0.0:
        return np.zeros(shape, dtype=np.float32)
    return rng.normal(0.0, std, size=shape).astype(np.float32)


def particle_wave(
    t: np.ndarray,
    amplitude: np.ndarray,
    doppler: np.ndarray,
    phase: np.ndarray,
    center: np.ndarray,
    width: np.ndarray,
) -> np.ndarray:
    carrier = np.cos(2.0 * np.pi * doppler[:, None] * t[None, :] + phase[:, None])
    envelope = np.exp(-np.square(t[None, :] - center[:, None]) / (2.0 * np.square(width[:, None])))
    return (amplitude[:, None] * carrier * envelope).astype(np.float32)


def _normalize_batch(signals: np.ndarray, normalization: str) -> np.ndarray:
    return np.stack([normalize_signal(row, mode=normalization) for row in signals]).astype(np.float32)


def _single_panel(
    rng: np.random.Generator,
    key: str,
    title: str,
    sweep_param: str,
    values: np.ndarray,
    length: int,
    noise_std: float,
    normalization: str,
    base_params: dict[str, float] | None = None,
    window_duration_ms: float = WINDOW_DURATION_MS,
) -> ParticleEquationPanel:
    n = int(values.size)
    base = SINGLE_BASE if base_params is None else base_params
    params = {name: np.full(n, value, dtype=np.float32) for name, value in base.items()}
    params[sweep_param] = values.astype(np.float32, copy=False)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    signal = particle_wave(t, params["A"], params["fD"], params["phi"], params["t0"], params["tau"])
    signal = signal + _noise(rng, signal.shape, noise_std)
    encoded_signal = _normalize_batch(signal, normalization)
    return ParticleEquationPanel(key, title, sweep_param, signal.astype(np.float32), encoded_signal, values, params, window_duration_ms)


def _single_snr_panel(
    rng: np.random.Generator,
    values: np.ndarray,
    length: int,
    normalization: str,
    base_params: dict[str, float],
    window_duration_ms: float = WINDOW_DURATION_MS,
) -> ParticleEquationPanel:
    n = int(values.size)
    params = {name: np.full(n, value, dtype=np.float32) for name, value in base_params.items()}
    params["snr_db"] = values.astype(np.float32, copy=False)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    clean_signal = particle_wave(t, params["A"], params["fD"], params["phi"], params["t0"], params["tau"])
    clean_rms = np.sqrt(np.mean(np.square(clean_signal), axis=1)).astype(np.float32)
    noise_std = (clean_rms / np.power(10.0, params["snr_db"] / 20.0)).astype(np.float32)
    params["snr_noise_std"] = noise_std
    noise = rng.normal(0.0, noise_std[:, None], size=clean_signal.shape).astype(np.float32)
    signal = clean_signal + noise
    encoded_signal = _normalize_batch(signal, normalization)
    return ParticleEquationPanel(
        "snr_db",
        "SNR",
        "snr_db",
        signal.astype(np.float32),
        encoded_signal,
        values.astype(np.float32, copy=False),
        params,
        window_duration_ms,
    )


def _paper_table_single_base(window_duration_ms: float, realistic_figure_based_sweeps: bool = False) -> dict[str, float]:
    tau_ms = REALISTIC_TAU_BASE_MS if realistic_figure_based_sweeps else 0.73 / GAUSSIAN_FWHM_TO_SIGMA
    return {
        "A": 0.43,
        "fD": 23.91 * window_duration_ms,
        "phi": 0.0,
        "t0": 0.5,
        "tau": tau_ms / window_duration_ms,
    }


def _paper_table_phase_visual_base(window_duration_ms: float) -> dict[str, float]:
    return {
        "A": 0.70,
        "fD": 8.00 * window_duration_ms,
        "phi": 0.0,
        "t0": 0.5,
        "tau": (0.33 / GAUSSIAN_FWHM_TO_SIGMA) / window_duration_ms,
    }


def _paper_table_single_specs(
    window_duration_ms: float = WINDOW_DURATION_MS,
    realistic_figure_based_sweeps: bool = False,
) -> list[tuple[str, str, str, float, float]]:
    ranges = PAPER_TABLE_STATS["global_min_max"]
    if realistic_figure_based_sweeps:
        tau_low = REALISTIC_TAU_SWEEP_MS[0] / window_duration_ms
        tau_high = REALISTIC_TAU_SWEEP_MS[1] / window_duration_ms
    else:
        passage = ranges["passage_time_ms"]
        tau_low = (float(passage["min"]) / GAUSSIAN_FWHM_TO_SIGMA) / window_duration_ms
        tau_high = (float(passage["max"]) / GAUSSIAN_FWHM_TO_SIGMA) / window_duration_ms
    return [
        ("amplitude_A", r"$A$", "A", 0.10, 3.56),
        ("doppler_fD", r"$f_D$", "fD", 8.00 * window_duration_ms, 37.60 * window_duration_ms),
        ("phase_phi", r"$\phi$", "phi", 0.0, 2.0 * np.pi),
        ("center_t0", r"$t_0$", "t0", 0.2, 0.8),
        ("width_tau", r"$\tau$", "tau", tau_low, tau_high),
    ]


def _legacy_single_specs() -> list[tuple[str, str, str, float, float]]:
    return [
        ("amplitude_A", r"$A$", "A", 0.25, 2.0),
        ("doppler_fD", r"$f_D$", "fD", 2.0, 64.0),
        ("phase_phi", r"$\phi$", "phi", 0.0, 2.0 * np.pi),
        ("center_t0", r"$t_0$", "t0", 0.2, 0.8),
        ("width_tau", r"$\tau$", "tau", 0.02, 0.15),
    ]


def generate_single_particle_panels(
    n_per_panel: int,
    length: int,
    seed: int,
    noise_std: float = 0.02,
    normalization: str = "window_zscore",
    shuffle: bool = True,
    sweep_source: str = "legacy",
    phase_profile: str = "table_mean",
    signal_window_duration_ms: float = WINDOW_DURATION_MS,
    realistic_figure_based_sweeps: bool = False,
) -> list[ParticleEquationPanel]:
    rng = np.random.default_rng(seed)
    if sweep_source == "paper_table":
        specs = _paper_table_single_specs(signal_window_duration_ms, realistic_figure_based_sweeps)
        base_params = _paper_table_single_base(signal_window_duration_ms, realistic_figure_based_sweeps)
    elif sweep_source == "legacy":
        specs = _legacy_single_specs()
        base_params = SINGLE_BASE
    else:
        raise ValueError(f"Unsupported single-particle sweep source: {sweep_source}")
    if phase_profile not in {"table_mean", "visual_low_cycles"}:
        raise ValueError(f"Unsupported phase profile: {phase_profile}")
    panels = [
        _single_panel(
            rng=rng,
            key=key,
            title=title,
            sweep_param=sweep_param,
            values=_linspace_param(low, high, n_per_panel, rng, shuffle),
            length=length,
            noise_std=noise_std,
            normalization=normalization,
            base_params=(
                _paper_table_phase_visual_base(signal_window_duration_ms)
                if sweep_source == "paper_table" and phase_profile == "visual_low_cycles" and key == "phase_phi"
                else base_params
            ),
            window_duration_ms=signal_window_duration_ms,
        )
        for key, title, sweep_param, low, high in specs
    ]
    panels.append(
        _single_snr_panel(
            rng=rng,
            values=_linspace_param(
                SNR_SWEEP_QUANTILES_DB["q20"],
                SNR_SWEEP_QUANTILES_DB["q80"],
                n_per_panel,
                rng,
                shuffle,
            ),
            length=length,
            normalization=normalization,
            base_params=base_params,
            window_duration_ms=signal_window_duration_ms,
        )
    )
    return panels


def _two_panel(
    rng: np.random.Generator,
    key: str,
    title: str,
    sweep_param: str,
    values: np.ndarray,
    length: int,
    noise_std: float,
    normalization: str,
) -> ParticleEquationPanel:
    n = int(values.size)
    params = {name: np.full(n, value, dtype=np.float32) for name, value in TWO_BASE.items()}
    if sweep_param == "separation_dt":
        params["t0A"] = (0.5 - values / 2.0).astype(np.float32)
        params["t0B"] = (0.5 + values / 2.0).astype(np.float32)
    elif sweep_param == "amplitude_ratio":
        params["B"] = values.astype(np.float32)
    elif sweep_param == "frequency_delta":
        params["fDB"] = (params["fDA"] + values).astype(np.float32)
    elif sweep_param == "phase_delta":
        params["phiB"] = (params["phiA"] + values).astype(np.float32)
    elif sweep_param == "width_ratio":
        params["tauB"] = (params["tauA"] * values).astype(np.float32)
    else:
        raise ValueError(f"Unsupported two-particle sweep: {sweep_param}")

    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    signal_a = particle_wave(t, params["A"], params["fDA"], params["phiA"], params["t0A"], params["tauA"])
    signal_b = particle_wave(t, params["B"], params["fDB"], params["phiB"], params["t0B"], params["tauB"])
    signal = signal_a + signal_b + _noise(rng, signal_a.shape, noise_std)
    encoded_signal = _normalize_batch(signal, normalization)
    return ParticleEquationPanel(key, title, sweep_param, signal.astype(np.float32), encoded_signal, values, params)


def generate_two_particle_panels(
    n_per_panel: int,
    length: int,
    seed: int,
    noise_std: float = 0.02,
    normalization: str = "window_zscore",
    shuffle: bool = True,
) -> list[ParticleEquationPanel]:
    rng = np.random.default_rng(seed)
    specs = [
        ("separation_dt", r"$\Delta t_0$", "separation_dt", 0.0, 3.0 * TWO_BASE["tauA"]),
        ("amplitude_ratio", r"$B/A$", "amplitude_ratio", 0.25, 2.0),
        ("frequency_delta", r"$f_{DB} - f_{DA}$", "frequency_delta", -16.0, 16.0),
        ("phase_delta", r"$\phi_B - \phi_A$", "phase_delta", 0.0, 2.0 * np.pi),
        ("width_ratio", r"$\tau_B/\tau_A$", "width_ratio", 0.5, 2.0),
    ]
    return [
        _two_panel(
            rng=rng,
            key=key,
            title=title,
            sweep_param=sweep_param,
            values=_linspace_param(low, high, n_per_panel, rng, shuffle),
            length=length,
            noise_std=noise_std,
            normalization=normalization,
        )
        for key, title, sweep_param, low, high in specs
    ]


def _load_yeast_range_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "found": False,
            "path": str(path),
            "note": "Range summary not found; using conservative fallback ranges.",
            "two_peak_only": YEAST_BUDDED_FALLBACK_RANGES,
        }
    with path.open() as f:
        payload = json.load(f)
    payload["found"] = True
    payload["path"] = str(path)
    return payload


def _summary_stat(summary: dict[str, Any], name: str, quantile: str, fallback: float) -> float:
    stats = summary.get("two_peak_only", {}).get(name, {})
    value = stats.get(quantile)
    if value is None or not math.isfinite(float(value)):
        value = YEAST_BUDDED_FALLBACK_RANGES.get(name, {}).get(quantile, fallback)
    if value is None or not math.isfinite(float(value)):
        return float(fallback)
    return float(value)


def _load_yeast_template_params(path: Path) -> dict[str, float]:
    params = dict(YEAST_TEMPLATE_FALLBACK_PARAMS)
    if not path.is_file():
        return params
    with path.open() as f:
        payload = json.load(f)
    best = payload.get("best_params", {})
    for key, fallback in YEAST_TEMPLATE_FALLBACK_PARAMS.items():
        value = best.get(key, fallback)
        if value is not None and math.isfinite(float(value)):
            params[key] = float(value)
    return params


def _load_yeast_template_signals(template_root: Path, length: int, source_group: str = "budding") -> tuple[np.ndarray, np.ndarray]:
    metadata_path = template_root / "events_metadata.csv"
    signals_path = template_root / "aligned_inputs.npz"
    if not metadata_path.is_file() or not signals_path.is_file():
        raise FileNotFoundError(f"Missing yeast template bank files under {template_root}")
    with metadata_path.open() as f:
        events = list(csv.DictReader(f))
    with np.load(signals_path, allow_pickle=True) as data:
        signals = data["signals"].astype(np.float32, copy=False)
    if len(events) != int(signals.shape[0]):
        raise ValueError(f"events/signals length mismatch in template bank: {len(events)} vs {signals.shape[0]}")

    indices = [
        i
        for i, row in enumerate(events)
        if row.get("source_group", "") == source_group and row.get("quality", "strict") == "strict"
    ]
    if not indices:
        indices = [i for i, row in enumerate(events) if row.get("source_group", "") == source_group]
    if not indices:
        raise ValueError(f"No yeast template events found for source_group={source_group!r} in {metadata_path}")
    selected = signals[np.asarray(indices, dtype=np.int64)]
    if selected.shape[1] != int(length):
        x_old = np.linspace(0.0, 1.0, selected.shape[1], dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, int(length), dtype=np.float32)
        selected = np.stack([np.interp(x_new, x_old, row).astype(np.float32) for row in selected])
    return selected.astype(np.float32, copy=False), np.asarray(indices, dtype=np.int32)


def _smooth_template_envelope(signal: np.ndarray) -> np.ndarray:
    envelope = np.abs(hilbert(np.asarray(signal, dtype=np.float64))).astype(np.float32)
    sigma = max(2.0, float(signal.size) * 0.020)
    radius = max(3, int(round(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * np.square(x / sigma)).astype(np.float32)
    kernel /= max(float(np.sum(kernel)), 1.0e-6)
    smoothed = np.convolve(envelope, kernel, mode="same").astype(np.float32)
    return smoothed / max(float(np.max(smoothed)), 1.0e-6)


def _smooth_template_signal(signal: np.ndarray, sigma_fraction: float = 0.045) -> np.ndarray:
    sigma = max(2.0, float(signal.size) * float(sigma_fraction))
    radius = max(3, int(round(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * np.square(x / sigma)).astype(np.float32)
    kernel /= max(float(np.sum(kernel)), 1.0e-6)
    return np.convolve(np.asarray(signal, dtype=np.float32), kernel, mode="same").astype(np.float32)


def _yeast_budded_base(summary: dict[str, Any], window_duration_ms: float, style: str) -> dict[str, float]:
    amp_a = _summary_stat(summary, "component_a_amplitude", "p50", 1.8)
    amp_ratio = np.clip(_summary_stat(summary, "amplitude_ratio_b_over_a", "p50", 0.95), 0.05, 4.0)
    delta_t0 = np.clip(_summary_stat(summary, "delta_t0", "p50", 0.055), 0.0, 0.45)
    if style == "original":
        tau_a = np.clip(_summary_stat(summary, "component_a_tau", "p50", 0.025), 0.002, 0.20)
        tau_b = np.clip(_summary_stat(summary, "component_b_tau", "p50", tau_a), 0.002, 0.20)
    elif style in {"budding_realistic", "template_budding"}:
        event_width_ms = _summary_stat(summary, "width_ms", "p50", 0.976)
        tau_a = np.clip((event_width_ms / 4.2) / window_duration_ms, 0.030, 0.220)
        tau_b = tau_a
    else:
        event_width_ms = _summary_stat(summary, "width_ms", "p50", 0.976)
        tau_a = np.clip((event_width_ms / GAUSSIAN_FWHM_TO_SIGMA) / window_duration_ms, 0.030, 0.330)
        tau_b = tau_a
    cycles = _summary_stat(summary, "doppler_peak_hz", "p50", 17_500.0) / 1000.0 * window_duration_ms
    phase_delta = _summary_stat(summary, "abs_delta_phi_rad", "p50", 1.2)
    return {
        "A": float(amp_a),
        "B": float(amp_a * amp_ratio),
        "fDA": float(cycles),
        "fDB": float(cycles),
        "phiA": 0.0,
        "phiB": float(phase_delta),
        "t0A": float(0.5 - delta_t0 / 2.0),
        "t0B": float(0.5 + delta_t0 / 2.0),
        "tauA": float(tau_a),
        "tauB": float(tau_b),
    }


def _yeast_realistic_wave(
    t: np.ndarray,
    amplitude: np.ndarray,
    doppler: np.ndarray,
    phase: np.ndarray,
    center: np.ndarray,
    width: np.ndarray,
) -> np.ndarray:
    tau = np.maximum(width[:, None], 1.0e-4)
    tt = t[None, :]
    center_col = center[:, None]
    main = np.exp(-np.square(tt - center_col) / (2.0 * np.square(tau)))
    pre = np.exp(-np.square(tt - (center_col - 1.45 * tau)) / (2.0 * np.square(1.25 * tau)))
    post = np.exp(-np.square(tt - (center_col + 1.95 * tau)) / (2.0 * np.square(1.45 * tau)))
    skew = 1.0 + 0.16 * np.tanh((tt - center_col) / tau)
    envelope = np.clip(main * skew + 0.16 * pre + 0.11 * post, 0.0, None)
    carrier = np.cos(2.0 * np.pi * doppler[:, None] * tt + phase[:, None])
    secondary = 0.18 * np.cos(2.0 * np.pi * (1.32 * doppler[:, None]) * tt + phase[:, None] + 0.85)
    return (amplitude[:, None] * (carrier + secondary) * envelope).astype(np.float32)


def _yeast_background_noise(
    rng: np.random.Generator,
    noise_std: np.ndarray,
    shape: tuple[int, int],
    correlated: bool,
) -> np.ndarray:
    white = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    if not correlated:
        return (white * noise_std[:, None]).astype(np.float32)
    x = np.linspace(-3.0, 3.0, 61, dtype=np.float32)
    kernel = np.exp(-0.5 * np.square(x)).astype(np.float32)
    kernel = kernel / float(np.sum(kernel))
    colored = np.empty_like(white)
    for i in range(shape[0]):
        colored[i] = np.convolve(white[i], kernel, mode="same")
    mixed = 0.08 * white + 0.92 * colored
    mixed_std = np.std(mixed, axis=1, keepdims=True).astype(np.float32)
    mixed = mixed / np.maximum(mixed_std, 1.0e-6)
    return (mixed * noise_std[:, None]).astype(np.float32)


def _yeast_budded_panel(
    rng: np.random.Generator,
    key: str,
    title: str,
    sweep_param: str,
    values: np.ndarray,
    length: int,
    normalization: str,
    base_params: dict[str, float],
    window_duration_ms: float,
    background_noise_std: float,
    style: str,
    template_signals: np.ndarray | None = None,
    template_source_indices: np.ndarray | None = None,
    template_params: dict[str, float] | None = None,
) -> ParticleEquationPanel:
    n = int(values.size)
    params = {name: np.full(n, value, dtype=np.float32) for name, value in base_params.items()}
    if sweep_param == "delta_t0":
        params["t0A"] = (0.5 - values / 2.0).astype(np.float32)
        params["t0B"] = (0.5 + values / 2.0).astype(np.float32)
    elif sweep_param == "amplitude_ratio":
        params["B"] = (params["A"] * values).astype(np.float32)
    elif sweep_param == "delta_fD":
        params["fDA"] = (base_params["fDA"] - values / 2.0).astype(np.float32)
        params["fDB"] = (base_params["fDA"] + values / 2.0).astype(np.float32)
    elif sweep_param == "delta_phi":
        params["phiB"] = (params["phiA"] + values).astype(np.float32)
    elif sweep_param == "tau_ratio":
        params["tauB"] = (params["tauA"] * values).astype(np.float32)
    elif sweep_param == "snr_proxy":
        params["snr_proxy"] = values.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported yeast budded sweep: {sweep_param}")

    if style == "template_budding":
        template_params = dict(YEAST_TEMPLATE_FALLBACK_PARAMS if template_params is None else template_params)
        common_jitter = float(template_params.get("common_t0_jitter", YEAST_TEMPLATE_FALLBACK_PARAMS["common_t0_jitter"]))
        common_shift = np.clip(rng.normal(0.0, common_jitter, size=n), -0.085, 0.085).astype(np.float32)
        params["t0A"] = (params["t0A"] + common_shift).astype(np.float32)
        params["t0B"] = (params["t0B"] + common_shift).astype(np.float32)
        params["template_common_t0_shift"] = common_shift

    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    wave_fn = _yeast_realistic_wave if style in {"budding_realistic", "template_budding"} else particle_wave
    signal_a = wave_fn(t, params["A"], params["fDA"], params["phiA"], params["t0A"], params["tauA"])
    signal_b = wave_fn(t, params["B"], params["fDB"], params["phiB"], params["t0B"], params["tauB"])
    clean_signal = signal_a + signal_b
    params["yeast_background_noise_std"] = np.full(n, float(background_noise_std), dtype=np.float32)
    if style == "template_budding":
        if template_signals is None or template_signals.size == 0:
            raise ValueError("template_budding style requires non-empty yeast template signals")
        template_params = dict(YEAST_TEMPLATE_FALLBACK_PARAMS if template_params is None else template_params)
        bank_indices = rng.integers(0, template_signals.shape[0], size=n)
        template_rows = template_signals[bank_indices].astype(np.float32, copy=False)
        template_norm = _normalize_batch(template_rows, "window_zscore")
        template_env = np.stack([_smooth_template_envelope(row) for row in template_norm]).astype(np.float32)
        template_low = _normalize_batch(
            np.stack([_smooth_template_signal(row) for row in template_norm]).astype(np.float32),
            "window_zscore",
        )
        env_strength = float(template_params.get("template_envelope_strength", YEAST_TEMPLATE_FALLBACK_PARAMS["template_envelope_strength"]))
        texture_strength = float(template_params.get("texture_strength", YEAST_TEMPLATE_FALLBACK_PARAMS["texture_strength"]))
        low_frequency_strength = float(
            template_params.get("low_frequency_template_strength", YEAST_TEMPLATE_FALLBACK_PARAMS["low_frequency_template_strength"])
        )
        clean_signal = clean_signal * ((1.0 - env_strength) + env_strength * template_env)
        texture_scale = np.std(clean_signal, axis=1, keepdims=True).astype(np.float32)
        template_mix = texture_strength * template_norm + low_frequency_strength * template_low
        clean_signal = clean_signal + np.maximum(texture_scale, 1.0e-6) * template_mix
        params["template_bank_row"] = bank_indices.astype(np.float32)
        if template_source_indices is not None:
            params["template_source_index"] = template_source_indices[bank_indices].astype(np.float32)
        params["template_envelope_strength"] = np.full(n, env_strength, dtype=np.float32)
        params["template_texture_strength"] = np.full(n, texture_strength, dtype=np.float32)
        params["template_low_frequency_strength"] = np.full(n, low_frequency_strength, dtype=np.float32)
        params["template_noise_scale"] = np.full(
            n,
            float(template_params.get("noise_scale", YEAST_TEMPLATE_FALLBACK_PARAMS["noise_scale"])),
            dtype=np.float32,
        )
        params["template_doppler_scale"] = np.full(
            n,
            float(template_params.get("doppler_scale", YEAST_TEMPLATE_FALLBACK_PARAMS["doppler_scale"])),
            dtype=np.float32,
        )
        params["template_tau_width_divisor"] = np.full(
            n,
            float(template_params.get("tau_width_divisor", YEAST_TEMPLATE_FALLBACK_PARAMS["tau_width_divisor"])),
            dtype=np.float32,
        )
    if sweep_param == "snr_proxy":
        clean_rms = np.sqrt(np.mean(np.square(clean_signal), axis=1)).astype(np.float32)
        variable_noise_std = clean_rms / np.maximum(values.astype(np.float32), 1.0e-3)
        noise_std = np.sqrt(np.square(variable_noise_std) + float(background_noise_std) ** 2).astype(np.float32)
        params["snr_noise_std"] = variable_noise_std
        params["total_noise_std"] = noise_std
    else:
        noise_std = np.full(n, float(background_noise_std), dtype=np.float32)
        params["total_noise_std"] = noise_std
    if style == "template_budding":
        noise_std = noise_std * float((template_params or YEAST_TEMPLATE_FALLBACK_PARAMS).get("noise_scale", YEAST_TEMPLATE_FALLBACK_PARAMS["noise_scale"]))
        params["total_noise_std"] = noise_std.astype(np.float32)
    signal = clean_signal + _yeast_background_noise(
        rng,
        noise_std=noise_std,
        shape=clean_signal.shape,
        correlated=style in {"budding_realistic", "template_budding"},
    )
    encoded_signal = _normalize_batch(signal, normalization)
    return ParticleEquationPanel(key, title, sweep_param, signal.astype(np.float32), encoded_signal, values, params, window_duration_ms)


def generate_yeast_budded_two_particle_panels(
    n_per_panel: int,
    length: int,
    seed: int,
    normalization: str = "window_zscore",
    shuffle: bool = True,
    range_summary_path: Path = DEFAULT_YEAST_RANGE_SUMMARY,
    signal_window_duration_ms: float = WINDOW_DURATION_MS,
    style: str = "budding_noisy",
    background_noise_scale: float = -1.0,
    template_root: Path = DEFAULT_YEAST_TEMPLATE_ROOT,
    template_calibration: Path = DEFAULT_YEAST_TEMPLATE_CALIBRATION,
) -> list[ParticleEquationPanel]:
    rng = np.random.default_rng(seed)
    summary = _load_yeast_range_summary(range_summary_path)
    if style not in set(YEAST_BUDDED_STYLES):
        raise ValueError(f"Unsupported yeast budded style: {style}")
    template_params: dict[str, float] | None = None
    template_signals: np.ndarray | None = None
    template_source_indices: np.ndarray | None = None
    if style == "template_budding":
        template_params = _load_yeast_template_params(template_calibration)
        template_signals, template_source_indices = _load_yeast_template_signals(template_root, length, source_group="budding")
    base_params = _yeast_budded_base(summary, signal_window_duration_ms, style)
    if style == "template_budding" and template_params is not None:
        doppler_scale = float(template_params.get("doppler_scale", YEAST_TEMPLATE_FALLBACK_PARAMS["doppler_scale"]))
        base_params["fDA"] = float(base_params["fDA"] * doppler_scale)
        base_params["fDB"] = float(base_params["fDB"] * doppler_scale)
        event_width_ms = _summary_stat(summary, "width_ms", "p50", 0.976)
        tau = np.clip(
            (event_width_ms / float(template_params.get("tau_width_divisor", YEAST_TEMPLATE_FALLBACK_PARAMS["tau_width_divisor"])))
            / signal_window_duration_ms,
            0.030,
            0.260,
        )
        base_params["tauA"] = float(tau)
        base_params["tauB"] = float(tau)
    if background_noise_scale < 0.0:
        background_noise_scale = {
            "original": 0.0,
            "budding_intermediate": 0.5,
            "budding_noisy": 1.0,
            "budding_realistic": 1.0,
            "template_budding": 1.0,
        }[style]
    background_noise_std = _summary_stat(summary, "background_edge_std", "p50", 0.51) * float(background_noise_scale)

    amp_low = np.clip(_summary_stat(summary, "amplitude_ratio_b_over_a", "p05", 0.55), 0.05, 4.0)
    amp_high = np.clip(_summary_stat(summary, "amplitude_ratio_b_over_a", "p95", 1.65), amp_low + 1.0e-3, 4.0)
    dt_low = np.clip(_summary_stat(summary, "delta_t0", "p05", 0.012), 0.0, 0.45)
    phase_high = np.clip(_summary_stat(summary, "abs_delta_phi_rad", "p95", 3.05), 0.1, np.pi)
    if style == "original":
        dt_high = np.clip(_summary_stat(summary, "delta_t0", "p95", 0.18), dt_low + 1.0e-4, 0.45)
        tau_low = np.clip(_summary_stat(summary, "tau_ratio_b_over_a", "p05", 0.55), 0.10, 5.0)
        tau_high = np.clip(_summary_stat(summary, "tau_ratio_b_over_a", "p95", 1.90), tau_low + 1.0e-3, 5.0)
    else:
        dt_high = np.clip(_summary_stat(summary, "delta_t0", "p75", 0.067), dt_low + 1.0e-4, 0.20)
        tau_low = np.clip(_summary_stat(summary, "tau_ratio_b_over_a", "p25", 0.62), 0.45, 1.75)
        tau_high = np.clip(_summary_stat(summary, "tau_ratio_b_over_a", "p75", 1.40), tau_low + 1.0e-3, 1.75)
    cycles_p50 = (
        base_params["fDA"]
        if style == "template_budding"
        else _summary_stat(summary, "doppler_peak_hz", "p50", base_params["fDA"] / signal_window_duration_ms * 1000.0) / 1000.0 * signal_window_duration_ms
    )
    cycles_spread = max(
        2.0,
        0.5
        * (
            _summary_stat(summary, "doppler_peak_hz", "p95", (cycles_p50 + 8.0) / signal_window_duration_ms * 1000.0)
            / 1000.0
            * signal_window_duration_ms
            - _summary_stat(summary, "doppler_peak_hz", "p05", (cycles_p50 - 8.0) / signal_window_duration_ms * 1000.0)
            / 1000.0
            * signal_window_duration_ms
        ),
    )
    snr_low = max(1.0, _summary_stat(summary, "snr_proxy", "p05", 5.0))
    snr_high = max(snr_low + 1.0e-3, _summary_stat(summary, "snr_proxy", "p95", 350.0))

    specs = [
        ("delta_t0", r"$\Delta t_0$", "delta_t0", dt_low, dt_high),
        ("amplitude_ratio", r"$B/A$", "amplitude_ratio", amp_low, amp_high),
        ("delta_fD", r"$\Delta f_D$", "delta_fD", -cycles_spread, cycles_spread),
        ("delta_phi", r"$\Delta \phi$", "delta_phi", -phase_high, phase_high),
        ("tau_ratio", r"$\tau_B/\tau_A$", "tau_ratio", tau_low, tau_high),
        ("snr_proxy", "SNR proxy", "snr_proxy", snr_low, snr_high),
    ]
    return [
        _yeast_budded_panel(
            rng=rng,
            key=key,
            title=title,
            sweep_param=sweep_param,
            values=_linspace_param(low, high, n_per_panel, rng, shuffle),
            length=length,
            normalization=normalization,
            base_params=base_params,
            window_duration_ms=signal_window_duration_ms,
            background_noise_std=background_noise_std,
            style=style,
            template_signals=template_signals,
            template_source_indices=template_source_indices,
            template_params=template_params,
        )
        for key, title, sweep_param, low, high in specs
    ]


def generate_panels(
    scenario: str,
    n_per_panel: int,
    length: int,
    seed: int,
    noise_std: float,
    normalization: str,
    single_sweep_source: str = "legacy",
    phase_profile: str = "table_mean",
    signal_window_duration_ms: float = WINDOW_DURATION_MS,
    realistic_figure_based_sweeps: bool = False,
    yeast_range_summary: Path = DEFAULT_YEAST_RANGE_SUMMARY,
    yeast_budded_style: str = "budding_noisy",
    yeast_background_noise_scale: float = -1.0,
    yeast_template_root: Path = DEFAULT_YEAST_TEMPLATE_ROOT,
    yeast_template_calibration: Path = DEFAULT_YEAST_TEMPLATE_CALIBRATION,
) -> list[ParticleEquationPanel]:
    if scenario == "single_particle":
        return generate_single_particle_panels(
            n_per_panel,
            length,
            seed,
            noise_std,
            normalization,
            sweep_source=single_sweep_source,
            phase_profile=phase_profile,
            signal_window_duration_ms=signal_window_duration_ms,
            realistic_figure_based_sweeps=realistic_figure_based_sweeps,
        )
    if scenario == "two_particles":
        return generate_two_particle_panels(n_per_panel, length, seed, noise_std, normalization)
    if scenario == "yeast_budded_two_particles":
        return generate_yeast_budded_two_particle_panels(
            n_per_panel=n_per_panel,
            length=length,
            seed=seed,
            normalization=normalization,
            shuffle=True,
            range_summary_path=yeast_range_summary,
            signal_window_duration_ms=signal_window_duration_ms,
            style=yeast_budded_style,
            background_noise_scale=yeast_background_noise_scale,
            template_root=yeast_template_root,
            template_calibration=yeast_template_calibration,
        )
    raise ValueError(f"Unsupported scenario: {scenario}")


def make_synthetic_events(panels: list[ParticleEquationPanel], scenario: str) -> list[ParticleEvent]:
    events: list[ParticleEvent] = []
    for panel in panels:
        for index in range(panel.signal.shape[0]):
            events.append(
                ParticleEvent(
                    event_id=f"{scenario}/{panel.key}/{index:06d}",
                    sample_id=f"{panel.key}_{index:06d}",
                    split="synthetic",
                    signal_path="",
                    label_path="",
                    class_id=0,
                    class_name=panel.key,
                    center_norm=float(panel.params.get("t0", panel.params.get("t0A"))[index]),
                    width_norm=float(panel.params.get("tau", panel.params.get("tauA"))[index]),
                    center_index=-1,
                    crop_start=0,
                    crop_end=int(panel.signal.shape[1]),
                )
            )
    return events


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def summarize_particle_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {"path": str(manifest_path), "found": False}
    by_class: dict[str, dict[str, list[float]]] = {}
    global_values: dict[str, list[float]] = {"frequency_khz": [], "width_ms": [], "center_fraction": [], "snr_db": []}
    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = str(row.get("class_name", "unknown"))
            by_class.setdefault(class_name, {key: [] for key in global_values})
            raw_values = {
                "frequency_khz": float(row["frequency"]) / 1000.0,
                "width_ms": float(row["width_ms"]),
                "center_fraction": float(row["center"]),
                "snr_db": float(row["snr_db"]),
            }
            for key, value in raw_values.items():
                if np.isfinite(value):
                    global_values[key].append(value)
                    by_class[class_name][key].append(value)
    return {
        "path": str(manifest_path),
        "found": True,
        "feature_units": {"frequency_khz": "kHz", "width_ms": "ms", "center_fraction": "fraction of full raw signal", "snr_db": "dB"},
        "global": {key: _quantiles(values) for key, values in global_values.items()},
        "by_particle_size": {class_name: {key: _quantiles(values) for key, values in values_by_key.items()} for class_name, values_by_key in sorted(by_class.items())},
    }


def save_signal_outputs(output_dir: Path, panels: list[ParticleEquationPanel], scenario: str, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_payload: dict[str, np.ndarray] = {}
    encoded_payload: dict[str, np.ndarray] = {}
    for panel in panels:
        raw_payload[f"{panel.key}_signals"] = panel.signal.astype(np.float32)
        encoded_payload[f"{panel.key}_signals"] = panel.encoded_signal.astype(np.float32)
        raw_payload[f"{panel.key}_color"] = panel.color_value.astype(np.float32)
        encoded_payload[f"{panel.key}_color"] = panel.color_value.astype(np.float32)
        for name, values in panel.params.items():
            raw_payload[f"{panel.key}_{name}"] = values.astype(np.float32)
            encoded_payload[f"{panel.key}_{name}"] = values.astype(np.float32)
    preprocessing_id = str(
        getattr(args, "preprocessing_summary", {})
        .get("preprocessing", {})
        .get("preprocessing_id", PREPROCESS_NONE)
    )
    raw_payload["preprocessing_id"] = np.asarray(PREPROCESS_NONE)
    encoded_payload["preprocessing_id"] = np.asarray(preprocessing_id)
    np.savez_compressed(output_dir / "synthetic_signals_raw.npz", **raw_payload)
    np.savez_compressed(output_dir / "synthetic_signals_encoded.npz", **encoded_payload)

    fieldnames = ["scenario", "panel", "index", "sweep_param", "color_value"]
    for panel in panels:
        for key in panel.params:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "synthetic_metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for panel in panels:
            for i in range(panel.signal.shape[0]):
                row: dict[str, Any] = {
                    "scenario": scenario,
                    "panel": panel.key,
                    "index": i,
                    "sweep_param": panel.sweep_param,
                    "color_value": float(panel.color_value[i]),
                }
                row.update({name: float(values[i]) for name, values in panel.params.items()})
                writer.writerow(row)

    with (output_dir / "run_config.json").open("w") as f:
        json.dump(
            {
                "scenario": scenario,
                "n_per_panel": int(args.n_per_panel),
                "input_length": int(args.input_length),
                "seed": int(args.seed),
                "noise_std": float(args.noise_std),
                "normalization": args.normalization,
                "time_domain": "normalized [0, 1]",
                "signal_window_duration_ms": float(args.signal_window_duration_ms),
                "realistic_figure_based_sweeps": bool(args.realistic_figure_based_sweeps),
                "realistic_tau_base_ms": REALISTIC_TAU_BASE_MS,
                "realistic_tau_sweep_ms": list(REALISTIC_TAU_SWEEP_MS),
                "single_sweep_source": args.single_sweep_source,
                "phase_profile": args.phase_profile,
                "phase_profile_note": (
                    "visual_low_cycles uses paper-table-valid low fD and short passage time only for the phi panel "
                    "to keep phase graphically legible."
                    if args.phase_profile == "visual_low_cycles"
                    else "table_mean keeps the same baseline parameters for every single-particle panel."
                ),
                "frequency_units": "cycles per synthetic window internally; paper_table source converts from kHz using signal_window_duration_ms",
                "amplitude_units": "mV for paper_table source; arbitrary units for legacy source",
                "tau_note": "paper_table tau is Gaussian sigma derived from passage_time_ms / 2.355",
                "paper_table_reference": PAPER_TABLE_STATS if args.single_sweep_source == "paper_table" else {},
                "particle_manifest_reference": summarize_particle_manifest(args.particle_manifest),
                "yeast_range_summary": _load_yeast_range_summary(args.yeast_range_summary),
                "yeast_events_metadata": str(args.yeast_events_metadata),
                "yeast_budded_style": args.yeast_budded_style,
                "yeast_background_noise_scale": float(args.yeast_background_noise_scale),
                "yeast_template_root": str(args.yeast_template_root),
                "yeast_template_calibration": str(args.yeast_template_calibration),
                "yeast_template_params": _load_yeast_template_params(args.yeast_template_calibration),
                "preprocessing": getattr(args, "preprocessing_summary", {"preprocessing": {"mode": PREPROCESS_NONE}}),
                "snr_sweep_source": SNR_SWEEP_SOURCE,
                "snr_sweep_quantiles_db": SNR_SWEEP_QUANTILES_DB,
                "snr_sweep_note": "The SNR panel injects per-sample raw Gaussian noise before configured normalization; global noise_std is not added again for this panel.",
                "yeast_budded_noise_note": (
                    "For yeast_budded_two_particles, every panel receives additive Gaussian background noise before normalization. "
                    "The default level is the p50 background_edge_std measured on selected yeast event crop edges; "
                    "the SNR proxy panel combines this background floor with variable clean_rms / snr_proxy noise."
                ),
                "display_units": {
                    "window_duration_ms": float(args.signal_window_duration_ms),
                    "A": "mV for paper_table source",
                    "fD": "kHz = cycles_per_window / signal_window_duration_ms",
                    "t0": "window fraction",
                    "tau": "ms = normalized_sigma * signal_window_duration_ms",
                    "snr_db": "dB",
                },
                "models": args.models,
                "skip_tsne": bool(args.skip_tsne),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def load_conv1dgap_same_input(checkpoint_path: Path, device: torch.device | str) -> nn.Module:
    from p0.models import create_model

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing Conv1D-GAP same-input checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model_name = str(state.get("model_name", "Conv1DGAP-L")) if isinstance(state, dict) else "Conv1DGAP-L"
    input_length = int(state.get("input_length", 512)) if isinstance(state, dict) else 512
    class_names = state.get("class_names", ("2um", "4um", "10um")) if isinstance(state, dict) else ("2um", "4um", "10um")
    num_classes = len(class_names)
    model = create_model(model_name, input_length=input_length, num_classes=num_classes)
    model_state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(model_state, strict=True)
    model.to(device).eval()
    return model


def encode_conv_features_all(model: nn.Module, signals: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, signals.shape[0], batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size]).float()
            features = backbone.encode_conv1dgap_features(model, batch, device=device)
            chunks.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def load_encoder_for_model(model_key: str, args: argparse.Namespace, device: torch.device, model_dir: Path):
    if model_key == "moment_official":
        args.event_length = int(args.input_length)
        return backbone.load_encoder(model_key, args.moment_model_id, args.cache_dir, device, model_dir, args)
    if model_key == "patchtst_pretrained":
        args.event_length = int(args.input_length)
        return backbone.load_encoder(model_key, args.patchtst_model_id, args.cache_dir, device, model_dir, args)
    if model_key == CONV_MODEL_KEY:
        model = load_conv1dgap_same_input(args.conv1dgap_checkpoint, device)
        return model, {
            "source_model_id": str(args.conv1dgap_checkpoint),
            "input_representation": f"same normalized {int(args.input_length)}-sample synthetic tensor as MOMENT/PatchTST",
            "supervised_same_input_checkpoint": True,
            "public_pretrained": False,
        }
    raise ValueError(f"Unsupported model: {model_key}")


def encode_panel_embeddings(
    model_key: str,
    encoder,
    panels: list[ParticleEquationPanel],
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    for panel in panels:
        if model_key == CONV_MODEL_KEY:
            emb = encode_conv_features_all(encoder, panel.encoded_signal, batch_size, device)
        else:
            emb = backbone.encode_all_events(model_key, encoder, panel.encoded_signal, batch_size, device)
        embeddings[panel.key] = emb.astype(np.float32)
    return embeddings


def reduce_panel_embeddings(embeddings: dict[str, np.ndarray], seed: int):
    reductions: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for key, emb in embeddings.items():
        pca, tsne, panel_metrics = backbone.reduce_embeddings(emb, seed=seed)
        reductions[key] = {"pca": pca, "tsne": tsne}
        metrics[key] = panel_metrics
    return reductions, metrics


def reduce_panel_embeddings_fast(embeddings: dict[str, np.ndarray], seed: int, skip_tsne: bool):
    if not skip_tsne:
        return reduce_panel_embeddings(embeddings, seed)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    reductions: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for key, emb in embeddings.items():
        x = StandardScaler().fit_transform(emb)
        pca_model = PCA(n_components=2, random_state=seed)
        pca = pca_model.fit_transform(x).astype(np.float32)
        panel_metrics = {
            "pca_explained_variance_ratio_sum": float(np.sum(pca_model.explained_variance_ratio_)),
            "trustworthiness": float("nan"),
            "tsne_perplexity": float("nan"),
            "tsne_skipped": 1.0,
        }
        reductions[key] = {"pca": pca, "tsne": pca.copy()}
        metrics[key] = panel_metrics
    return reductions, metrics


def save_model_outputs(
    output_dir: Path,
    panels: list[ParticleEquationPanel],
    embeddings: dict[str, np.ndarray],
    reductions: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, dict[str, float]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for panel in panels:
        payload[f"{panel.key}_embeddings"] = embeddings[panel.key].astype(np.float32)
        payload[f"{panel.key}_pca"] = reductions[panel.key]["pca"].astype(np.float32)
        payload[f"{panel.key}_tsne"] = reductions[panel.key]["tsne"].astype(np.float32)
        payload[f"{panel.key}_color"] = panel.color_value.astype(np.float32)
    np.savez_compressed(output_dir / "embeddings.npz", **payload)
    with (output_dir / "reduction_metrics.json").open("w") as f:
        json.dump({"reduction": metrics, "metadata": metadata}, f, indent=2, sort_keys=True)


def _add_colorbar(ax: plt.Axes, artist: plt.Collection, label: str | None = None) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    cb = ax.figure.colorbar(artist, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2)
    if label:
        cb.set_label(label, fontsize=6, labelpad=3)


def _set_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=6, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#555555")
    ax.grid(False)


def _single_param_display(panel: ParticleEquationPanel) -> tuple[np.ndarray, str, str, str]:
    value = panel.color_value.astype(np.float32, copy=False)
    if panel.sweep_param == "A":
        return value, "A [mV]", "A", "mV"
    if panel.sweep_param == "fD":
        return value / panel.window_duration_ms, "fD [kHz]", "fD", "kHz"
    if panel.sweep_param == "phi":
        return value, "phi [rad]", "phi", "rad"
    if panel.sweep_param == "t0":
        return value, "t0 [window fraction]", "t0", ""
    if panel.sweep_param == "tau":
        return value * panel.window_duration_ms, "tau [ms]", "tau", "ms"
    if panel.sweep_param == "snr_db":
        return value, "SNR [dB]", "SNR", "dB"
    if panel.sweep_param == "delta_t0":
        return value * panel.window_duration_ms, "Delta t0 [ms]", "Delta t0", "ms"
    if panel.sweep_param == "amplitude_ratio":
        return value, "B/A", "B/A", ""
    if panel.sweep_param in {"delta_fD", "frequency_delta"}:
        return value / panel.window_duration_ms, "Delta fD [kHz]", "Delta fD", "kHz"
    if panel.sweep_param in {"delta_phi", "phase_delta"}:
        return value, "Delta phi [rad]", "Delta phi", "rad"
    if panel.sweep_param in {"tau_ratio", "width_ratio"}:
        return value, "tauB/tauA", "tauB/tauA", ""
    if panel.sweep_param == "snr_proxy":
        return value, "SNR proxy", "SNR proxy", ""
    if panel.sweep_param == "separation_dt":
        return value * panel.window_duration_ms, "Delta t0 [ms]", "Delta t0", "ms"
    return value, panel.sweep_param, panel.sweep_param, ""


def _model_title(model_key: str) -> str:
    return MODEL_DISPLAY.get(model_key, model_key)


def _scenario_title(scenario: str) -> str:
    if scenario == "single_particle":
        return "single-particle signals"
    if scenario == "yeast_budded_two_particles":
        return "yeast budded two-particle signals"
    return "two-particle signals"


def _plot_reduction_panel(ax: plt.Axes, coords: np.ndarray, panel: ParticleEquationPanel) -> None:
    color_values, title, _, colorbar_label = _single_param_display(panel)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color_values,
        cmap="magma",
        s=9,
        alpha=0.88,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(title, pad=4)
    _set_axis_style(ax)
    _add_colorbar(ax, scatter, colorbar_label)


def _phase_profile_caption(panels: list[ParticleEquationPanel]) -> str:
    phase_panel = next((panel for panel in panels if panel.key == "phase_phi"), None)
    if phase_panel is None:
        return ""
    f_d_khz = float(np.median(phase_panel.params["fD"]) / phase_panel.window_duration_ms)
    tau_ms = float(np.median(phase_panel.params["tau"]) * phase_panel.window_duration_ms)
    visual_base = _paper_table_phase_visual_base(phase_panel.window_duration_ms)
    visual_f_d_khz = visual_base["fD"] / phase_panel.window_duration_ms
    visual_tau_ms = visual_base["tau"] * phase_panel.window_duration_ms
    if np.isclose(f_d_khz, visual_f_d_khz) and np.isclose(tau_ms, visual_tau_ms):
        return " The phi panel uses a table-valid low-cycle profile (fD=8.0 kHz, tau=0.140 ms) for graphical legibility."
    return ""


def plot_model_grid(
    panels: list[ParticleEquationPanel],
    reductions_by_model: dict[str, dict[str, dict[str, np.ndarray]]],
    output_pdf: Path,
    output_png: Path,
    scenario: str,
) -> None:
    backbone.apply_fig7_plot_style()
    model_keys = list(reductions_by_model)
    n_rows = 2 * len(model_keys)
    n_cols = len(panels)
    n_per_panel = panels[0].color_value.size if panels else 0
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(14.2, 3.55 * n_cols), max(6.3, 1.75 * n_rows)),
        constrained_layout=False,
        squeeze=False,
    )
    for model_idx, model_key in enumerate(model_keys):
        for reduction_idx, reduction_key in enumerate(("pca", "tsne")):
            row = 2 * model_idx + reduction_idx
            for col, panel in enumerate(panels):
                ax = axes[row, col]
                coords = reductions_by_model[model_key][panel.key][reduction_key]
                _plot_reduction_panel(ax, coords, panel)
                if row != 0:
                    ax.set_title("")
                if col == 0:
                    ax.set_ylabel(f"{_model_title(model_key)}\n{reduction_key.upper()}", fontsize=7)
    caption = (
        f"Latent spaces for {_scenario_title(scenario)}: PCA and t-SNE projections, "
        f"n={n_per_panel} points per swept variable. Points are rasterized in the PDF; axes and text remain vector graphics."
        f"{_phase_profile_caption(panels)}"
    )
    fig.suptitle("Latent representation comparison", fontsize=12, y=0.975)
    fig.subplots_adjust(left=0.075, right=0.958, top=0.92, bottom=0.10, wspace=0.44, hspace=0.46)
    fig.text(0.075, 0.035, caption, ha="left", va="bottom", fontsize=10, wrap=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=RASTERIZED_PDF_DPI)
    fig.savefig(output_png, dpi=PNG_DPI)
    plt.close(fig)


def plot_single_model_figure(
    panels: list[ParticleEquationPanel],
    reductions: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, dict[str, float]],
    model_key: str,
    output_pdf: Path,
    output_png: Path,
    scenario: str,
) -> None:
    del metrics
    backbone.apply_fig7_plot_style()
    n_cols = len(panels)
    n_per_panel = panels[0].color_value.size if panels else 0
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(max(14.2, 3.55 * n_cols), 6.3),
        constrained_layout=False,
        squeeze=False,
    )
    for col, panel in enumerate(panels):
        for row, reduction_key in enumerate(("pca", "tsne")):
            ax = axes[row, col]
            coords = reductions[panel.key][reduction_key]
            _plot_reduction_panel(ax, coords, panel)
            if row != 0:
                ax.set_title("")
            if col == 0:
                ax.set_ylabel(reduction_key.upper(), fontsize=8)
    caption = (
        f"{_scenario_title(scenario)} encoded with {_model_title(model_key)}. "
        f"n={n_per_panel} points per swept variable. "
        "Sweeps use physical OFI feature ranges when run with the paper-table source."
        f"{_phase_profile_caption(panels)}"
    )
    fig.suptitle(f"Latent spaces - {_model_title(model_key)}", fontsize=12, y=0.965)
    fig.subplots_adjust(left=0.055, right=0.958, top=0.86, bottom=0.16, wspace=0.44, hspace=0.34)
    fig.text(0.055, 0.045, caption, ha="left", va="bottom", fontsize=10, wrap=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=RASTERIZED_PDF_DPI)
    fig.savefig(output_png, dpi=PNG_DPI)
    plt.close(fig)


def _example_indices_by_sweep_value(panel: ParticleEquationPanel, count: int = 5) -> np.ndarray:
    order = np.argsort(panel.color_value)
    if order.size <= count:
        return order.astype(np.int64)
    positions = np.linspace(0, order.size - 1, count, dtype=np.int64)
    return order[positions].astype(np.int64)


def _single_particle_envelope(panel: ParticleEquationPanel, index: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, panel.signal.shape[1], dtype=np.float32)
    return panel.params["A"][index] * np.exp(-np.square(t - panel.params["t0"][index]) / (2.0 * panel.params["tau"][index] ** 2))


def _single_particle_display_signal(panel: ParticleEquationPanel, index: int) -> tuple[np.ndarray, np.ndarray]:
    return panel.signal[index].astype(np.float32), _single_particle_envelope(panel, index).astype(np.float32)


def plot_single_particle_signal_examples(
    panels: list[ParticleEquationPanel],
    output_pdf: Path,
    output_png: Path,
    examples_per_panel: int = 5,
) -> None:
    if not panels:
        return
    backbone.apply_fig7_plot_style()
    fig, axes = plt.subplots(
        len(panels),
        examples_per_panel,
        figsize=(15.4, max(10.4, 1.72 * len(panels))),
        constrained_layout=False,
        squeeze=False,
        sharex=True,
    )
    x_ms = np.linspace(0.0, panels[0].window_duration_ms, panels[0].signal.shape[1], dtype=np.float32)
    for row, panel in enumerate(panels):
        indices = _example_indices_by_sweep_value(panel, examples_per_panel)
        display_values, row_title, display_name, unit = _single_param_display(panel)
        for col, sample_idx in enumerate(indices.tolist()):
            ax = axes[row, col]
            display_signal, display_envelope = _single_particle_display_signal(panel, sample_idx)
            ax.plot(x_ms, display_signal, color="#1f77b4", linewidth=0.8)
            ax.plot(x_ms, display_envelope, color="#d95f02", linewidth=1.0, alpha=0.24)
            suffix = f" {unit}" if unit else ""
            title = f"{display_name} = {display_values[sample_idx]:.2f}{suffix}"
            ax.set_title(title, pad=3, fontsize=7)
            ax.tick_params(axis="both", labelsize=6, length=2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color("#555555")
            ax.grid(False)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=8)
            if row == len(panels) - 1:
                ax.set_xlabel("Time [ms]", fontsize=7)
    formula = r"$s(t)=A\cos(2\pi f_D t + \phi)\exp\left(-(t-t_0)^2/(2\tau^2)\right)+\epsilon$"
    caption = (
        "Informative analytic signals used to probe the latent spaces. "
        "The blue traces show the noised raw signals and the orange curve shows the Gaussian envelope. "
        "For paper-table sweeps, A is the peak time-domain amplitude in mV, fD is converted from kHz, "
        "and tau is Gaussian sigma derived from passage time / 2.355. "
        f"The plotted traces are the generated model inputs over a 0-{panels[0].window_duration_ms:.2f} ms window. "
        "The SNR row uses per-sample Gaussian noise over the particle visual-subset q20-q80 SNR range. "
        "The local manifest summary is stored in run_config.json for comparison against the paper ranges."
        f"{_phase_profile_caption(panels)}"
    )
    fig.suptitle("Tested formula and synthetic particle signals", fontsize=13, y=0.985)
    fig.text(0.5, 0.945, formula, ha="center", va="center", fontsize=13)
    fig.subplots_adjust(left=0.085, right=0.972, top=0.905, bottom=0.12, wspace=0.28, hspace=0.48)
    fig.text(0.085, 0.035, caption, ha="left", va="bottom", fontsize=9.5, wrap=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=RASTERIZED_PDF_DPI)
    fig.savefig(output_png, dpi=PNG_DPI)
    plt.close(fig)


def _two_particle_display_components(panel: ParticleEquationPanel, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, panel.signal.shape[1], dtype=np.float32)
    signal_a = particle_wave(
        t,
        panel.params["A"][index : index + 1],
        panel.params["fDA"][index : index + 1],
        panel.params["phiA"][index : index + 1],
        panel.params["t0A"][index : index + 1],
        panel.params["tauA"][index : index + 1],
    )[0]
    signal_b = particle_wave(
        t,
        panel.params["B"][index : index + 1],
        panel.params["fDB"][index : index + 1],
        panel.params["phiB"][index : index + 1],
        panel.params["t0B"][index : index + 1],
        panel.params["tauB"][index : index + 1],
    )[0]
    env_a = panel.params["A"][index] * np.exp(-np.square(t - panel.params["t0A"][index]) / (2.0 * panel.params["tauA"][index] ** 2))
    env_b = panel.params["B"][index] * np.exp(-np.square(t - panel.params["t0B"][index]) / (2.0 * panel.params["tauB"][index] ** 2))
    return signal_a.astype(np.float32), signal_b.astype(np.float32), env_a.astype(np.float32), env_b.astype(np.float32)


def plot_two_particle_signal_examples(
    panels: list[ParticleEquationPanel],
    output_pdf: Path,
    output_png: Path,
    scenario: str,
    examples_per_panel: int = 5,
) -> None:
    if not panels:
        return
    backbone.apply_fig7_plot_style()
    fig, axes = plt.subplots(
        len(panels),
        examples_per_panel,
        figsize=(15.4, max(10.4, 1.72 * len(panels))),
        constrained_layout=False,
        squeeze=False,
        sharex=True,
    )
    x_ms = np.linspace(0.0, panels[0].window_duration_ms, panels[0].signal.shape[1], dtype=np.float32)
    for row, panel in enumerate(panels):
        indices = _example_indices_by_sweep_value(panel, examples_per_panel)
        display_values, row_title, display_name, unit = _single_param_display(panel)
        for col, sample_idx in enumerate(indices.tolist()):
            ax = axes[row, col]
            signal_a, signal_b, env_a, env_b = _two_particle_display_components(panel, sample_idx)
            display_signal = panel.signal[sample_idx]
            display_clean = signal_a + signal_b
            display_env_a = env_a
            display_env_b = env_b
            reference_alpha = 0.45
            envelope_alpha = 0.24
            if "template_bank_row" in panel.params:
                display_signal = panel.encoded_signal[sample_idx]
                display_clean = normalize_signal(display_clean, mode="window_zscore")
                env_peak = max(float(np.max(np.abs(env_a + env_b))), 1.0e-6)
                env_scale = 0.42 * max(float(np.max(np.abs(display_signal))), 1.0e-6) / env_peak
                display_env_a = env_a * env_scale
                display_env_b = env_b * env_scale
                reference_alpha = 0.24
                envelope_alpha = 0.14
            ax.plot(x_ms, display_signal, color="#1f77b4", linewidth=0.8)
            ax.plot(x_ms, display_clean, color="#111111", linewidth=0.6, alpha=reference_alpha)
            ax.plot(x_ms, display_env_a, color="#d95f02", linewidth=0.9, alpha=envelope_alpha)
            ax.plot(x_ms, display_env_b, color="#009E73", linewidth=0.9, alpha=envelope_alpha)
            suffix = f" {unit}" if unit else ""
            ax.set_title(f"{display_name} = {display_values[sample_idx]:.2f}{suffix}", pad=3, fontsize=7)
            ax.tick_params(axis="both", labelsize=6, length=2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color("#555555")
            ax.grid(False)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=8)
            if row == len(panels) - 1:
                ax.set_xlabel("Time [ms]", fontsize=7)
    formula = (
        r"$s(t)=\sum_{i\in\{A,B\}} A_i\cos(2\pi f_{D,i}t+\phi_i)"
        r"\exp\left(-(t-t_{0,i})^2/(2\tau_i^2)\right)+\epsilon$"
    )
    caption = (
        f"Generated {_scenario_title(scenario)} used as a validation proof before latent-space encoding. "
        "Blue traces are model inputs, black traces are the underlying analytic two-component reference, and orange/green curves are the two Gaussian envelopes. "
        "The yeast budded scenario uses estimated close-component ranges from the yeast range analysis JSON when present; "
        "otherwise conservative fallback ranges are recorded in run_config.json. "
        "For template_budding, real budding crops also provide envelope modulation and texture/noise templates before normalization. "
        "The SNR proxy row injects per-sample Gaussian noise before normalization."
    )
    fig.suptitle("Two-component analytic yeast signal examples", fontsize=13, y=0.985)
    fig.text(0.5, 0.945, formula, ha="center", va="center", fontsize=12)
    fig.subplots_adjust(left=0.085, right=0.972, top=0.905, bottom=0.12, wspace=0.28, hspace=0.48)
    fig.text(0.085, 0.035, caption, ha="left", va="bottom", fontsize=9.5, wrap=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, dpi=RASTERIZED_PDF_DPI)
    fig.savefig(output_png, dpi=PNG_DPI)
    plt.close(fig)


def parse_models(raw: str) -> list[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    unsupported = [item for item in models if item not in MODEL_KEYS]
    if unsupported:
        raise ValueError(f"Unsupported models {unsupported}. Expected any of {MODEL_KEYS}")
    return models


def p1_preprocess_config_from_args(args: argparse.Namespace) -> P1PreprocessConfig:
    return P1PreprocessConfig(
        mode=str(getattr(args, "preprocess_mode", PREPROCESS_NONE)),
        sampling_frequency_hz=float(getattr(args, "preprocess_sampling_frequency_hz", 2_000_000.0)),
        low_khz=float(getattr(args, "preprocess_low_khz", 5.0)),
        high_khz_max=float(getattr(args, "preprocess_high_khz_max", 100.0)),
        saturation_fmin_hz=float(getattr(args, "saturation_fmin_hz", 7_000.0)),
        saturation_fmax_hz=float(getattr(args, "saturation_fmax_hz", 80_000.0)),
        saturation_min_flat=int(getattr(args, "saturation_min_flat", 500)),
        saturation_zero_threshold=float(getattr(args, "saturation_zero_threshold", 1.0e-4)),
        saturation_guard_before=int(getattr(args, "saturation_guard_before", 0)),
        saturation_guard_after=int(getattr(args, "saturation_guard_after", 0)),
        normalization=str(getattr(args, "normalization", "window_zscore")),
    )


def apply_panel_preprocessing(
    panels: list[ParticleEquationPanel],
    args: argparse.Namespace,
) -> tuple[list[ParticleEquationPanel], dict[str, Any]]:
    cfg = p1_preprocess_config_from_args(args)
    if cfg.mode == PREPROCESS_NONE:
        summary = {"preprocessing": cfg.to_dict(), "panels": {}, "rows": int(sum(panel.signal.shape[0] for panel in panels))}
        return panels, summary
    out: list[ParticleEquationPanel] = []
    panel_summaries: dict[str, Any] = {}
    for panel in panels:
        encoded, summary = preprocess_batch(
            panel.signal,
            cfg=cfg,
            reject_saturation=bool(getattr(args, "preprocess_reject_synthetic_saturation", False)),
        )
        if encoded.shape != panel.encoded_signal.shape:
            raise ValueError(
                f"Preprocessing changed synthetic row count for panel {panel.key}: "
                f"{encoded.shape} vs {panel.encoded_signal.shape}. Synthetic rows must stay aligned with metadata."
            )
        panel_summaries[panel.key] = summary
        out.append(
            ParticleEquationPanel(
                key=panel.key,
                title=panel.title,
                sweep_param=panel.sweep_param,
                signal=panel.signal,
                encoded_signal=encoded,
                color_value=panel.color_value,
                params=panel.params,
                window_duration_ms=panel.window_duration_ms,
            )
        )
    return out, {"preprocessing": cfg.to_dict(), "panels": panel_summaries}


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    panels = generate_panels(
        scenario=args.scenario,
        n_per_panel=args.n_per_panel,
        length=args.input_length,
        seed=args.seed,
        noise_std=args.noise_std,
        normalization=args.normalization,
        single_sweep_source=args.single_sweep_source,
        phase_profile=args.phase_profile,
        signal_window_duration_ms=args.signal_window_duration_ms,
        realistic_figure_based_sweeps=args.realistic_figure_based_sweeps,
        yeast_range_summary=args.yeast_range_summary,
        yeast_budded_style=args.yeast_budded_style,
        yeast_background_noise_scale=args.yeast_background_noise_scale,
        yeast_template_root=args.yeast_template_root,
        yeast_template_calibration=args.yeast_template_calibration,
    )
    panels, preprocessing_summary = apply_panel_preprocessing(panels, args)
    args.preprocessing_summary = preprocessing_summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_signal_outputs(args.output_dir, panels, args.scenario, args)
    with (args.output_dir / "preprocessing_summary.json").open("w") as f:
        json.dump(preprocessing_summary, f, indent=2, sort_keys=True)
    events = make_synthetic_events(panels, args.scenario)
    with (args.output_dir / "synthetic_events_metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0].__dataclass_fields__.keys()))
        writer.writeheader()
        for event in events:
            writer.writerow({key: getattr(event, key) for key in event.__dataclass_fields__})

    if args.scenario == "single_particle":
        plot_single_particle_signal_examples(
            panels=panels,
            output_pdf=args.output_dir / "single_particle_formula_signal_examples.pdf",
            output_png=args.output_dir / "single_particle_formula_signal_examples.png",
        )
    elif args.scenario in {"two_particles", "yeast_budded_two_particles"}:
        plot_two_particle_signal_examples(
            panels=panels,
            output_pdf=args.output_dir / f"{args.scenario}_formula_signal_examples.pdf",
            output_png=args.output_dir / f"{args.scenario}_formula_signal_examples.png",
            scenario=args.scenario,
        )

    if args.signals_only:
        print(f"Wrote particle-equation signal proof outputs to {args.output_dir}")
        return

    device = torch.device(args.device)
    reductions_by_model: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for model_key in parse_models(args.models):
        model_dir = args.output_dir / model_key
        encoder, metadata = load_encoder_for_model(model_key, args, device, model_dir)
        embeddings = encode_panel_embeddings(model_key, encoder, panels, args.batch_size, device)
        reductions, metrics = reduce_panel_embeddings_fast(embeddings, args.seed, args.skip_tsne)
        save_model_outputs(
            output_dir=model_dir,
            panels=panels,
            embeddings=embeddings,
            reductions=reductions,
            metrics=metrics,
            metadata={
                **metadata,
                "model_key": model_key,
                "display_name": MODEL_DISPLAY.get(model_key, model_key),
                "input_length": int(args.input_length),
                "normalization": args.normalization,
            },
        )
        reductions_by_model[model_key] = reductions
        plot_single_model_figure(
            panels=panels,
            reductions=reductions,
            metrics=metrics,
            model_key=model_key,
            output_pdf=args.output_dir / f"{model_key}_{args.scenario}_latent_sweeps_pca_tsne.pdf",
            output_png=args.output_dir / f"{model_key}_{args.scenario}_latent_sweeps_pca_tsne.png",
            scenario=args.scenario,
        )

    plot_model_grid(
        panels=panels,
        reductions_by_model=reductions_by_model,
        output_pdf=args.output_dir / f"{args.scenario}_latent_sweeps_pca_tsne.pdf",
        output_png=args.output_dir / f"{args.scenario}_latent_sweeps_pca_tsne.png",
        scenario=args.scenario,
    )
    print(f"Wrote particle-equation latent sweeps to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode analytic particle equations into MOMENT/PatchTST/Conv1D-GAP latent spaces.")
    parser.add_argument("--scenario", choices=["single_particle", "two_particles", "yeast_budded_two_particles"], default="single_particle")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "particle_equation_latent_sweeps")
    parser.add_argument("--models", default="moment_official,patchtst_pretrained,conv1dgap_same_input_3class")
    parser.add_argument("--moment-model-id", default=MOMENT_DEFAULT_ID)
    parser.add_argument("--patchtst-model-id", default=PATCHTST_DEFAULT_ID)
    parser.add_argument("--conv1dgap-checkpoint", type=Path, default=DEFAULT_CONV_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT.parent / ".cache" / "huggingface")
    parser.add_argument("--n-per-panel", type=int, default=1800)
    parser.add_argument("--input-length", type=int, default=4096)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--normalization", choices=["none", "window_zscore", "robust_zscore"], default="window_zscore")
    parser.add_argument("--preprocess-mode", choices=PREPROCESS_MODES, default=PREPROCESS_NONE)
    parser.add_argument("--preprocess-sampling-frequency-hz", type=float, default=2_000_000.0)
    parser.add_argument("--preprocess-low-khz", type=float, default=5.0)
    parser.add_argument("--preprocess-high-khz-max", type=float, default=100.0)
    parser.add_argument("--saturation-fmin-hz", type=float, default=7_000.0)
    parser.add_argument("--saturation-fmax-hz", type=float, default=80_000.0)
    parser.add_argument("--saturation-min-flat", type=int, default=500)
    parser.add_argument("--saturation-zero-threshold", type=float, default=1.0e-4)
    parser.add_argument("--saturation-guard-before", type=int, default=0)
    parser.add_argument("--saturation-guard-after", type=int, default=0)
    parser.add_argument(
        "--preprocess-reject-synthetic-saturation",
        action="store_true",
        help="Make P1 flat/saturation intervals fatal for synthetic rows instead of audit-only.",
    )
    parser.add_argument("--single-sweep-source", choices=["legacy", "paper_table"], default="legacy")
    parser.add_argument("--phase-profile", choices=["table_mean", "visual_low_cycles"], default="table_mean")
    parser.add_argument("--signal-window-duration-ms", type=float, default=WINDOW_DURATION_MS)
    parser.add_argument("--realistic-figure-based-sweeps", action="store_true")
    parser.add_argument("--particle-manifest", type=Path, default=DEFAULT_PARTICLE_MANIFEST)
    parser.add_argument("--yeast-range-summary", type=Path, default=DEFAULT_YEAST_RANGE_SUMMARY)
    parser.add_argument("--yeast-events-metadata", type=Path, default=DEFAULT_YEAST_EVENTS_METADATA)
    parser.add_argument("--yeast-template-root", type=Path, default=DEFAULT_YEAST_TEMPLATE_ROOT)
    parser.add_argument("--yeast-template-calibration", type=Path, default=DEFAULT_YEAST_TEMPLATE_CALIBRATION)
    parser.add_argument(
        "--yeast-budded-style",
        choices=list(YEAST_BUDDED_STYLES),
        default="budding_noisy",
    )
    parser.add_argument(
        "--yeast-background-noise-scale",
        type=float,
        default=-1.0,
        help="Scale for p50 measured yeast background noise; negative uses the style default.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-tsne", action="store_true", help="Use PCA coordinates in the t-SNE row for quick smoke tests.")
    parser.add_argument("--signals-only", action="store_true", help="Write generated signal NPZ/CSV and proof figure, then skip model encoding.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
