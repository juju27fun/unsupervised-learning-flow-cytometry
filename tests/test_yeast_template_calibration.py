from __future__ import annotations

import numpy as np

from p3_ssl.yeast_template_calibration import (
    compute_signal_metrics,
    generate_template_candidate_signals,
    score_metric_summary,
    summarize_metrics,
)


def test_template_calibration_metrics_and_score_are_finite() -> None:
    t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    env = np.exp(-np.square(t - 0.5) / (2.0 * 0.18**2)).astype(np.float32)
    signals = np.stack(
        [
            env * np.sin(2.0 * np.pi * 18.0 * t),
            0.8 * env * np.sin(2.0 * np.pi * 22.0 * t + 0.5),
            0.6 * env * np.sin(2.0 * np.pi * 15.0 * t + 1.0),
        ]
    ).astype(np.float32)

    summary = summarize_metrics(compute_signal_metrics(signals))
    assert set(summary) >= {"edge_std", "support_frac", "spectral_centroid"}
    assert all(np.isfinite(summary[key]["p50"]) for key in summary)
    assert np.isfinite(score_metric_summary(summary, summary))
    assert score_metric_summary(summary, summary) == 0.0


def test_template_candidate_generation_shapes_and_finiteness() -> None:
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    env = np.exp(-np.square(t - 0.5) / (2.0 * 0.18**2)).astype(np.float32)
    templates = np.stack(
        [
            env * np.sin(2.0 * np.pi * 18.0 * t),
            0.8 * env * np.sin(2.0 * np.pi * 22.0 * t + 0.5),
        ]
    ).astype(np.float32)
    range_summary = {
        "two_peak_only": {
            "background_edge_std": {"p50": 0.05},
            "width_ms": {"p50": 1.0},
            "component_a_amplitude": {"p50": 2.0},
            "amplitude_ratio_b_over_a": {"p05": 0.6, "p95": 1.4},
            "delta_t0": {"p05": 0.02, "p75": 0.08},
            "abs_delta_phi_rad": {"p95": 2.5},
            "doppler_peak_hz": {"p50": 15000.0},
        }
    }
    params = {
        "template_envelope_strength": 0.35,
        "texture_strength": 0.10,
        "noise_scale": 0.40,
        "tau_width_divisor": 4.0,
        "asymmetry_strength": 0.10,
        "secondary_doppler_strength": 0.10,
    }

    generated = generate_template_candidate_signals(
        rng=rng,
        templates=templates,
        range_summary=range_summary,
        params=params,
        n_samples=5,
        window_duration_ms=2.048,
        normalization="none",
    )

    assert generated.shape == (5, 128)
    assert np.isfinite(generated).all()
    assert float(np.std(generated)) > 0.0
