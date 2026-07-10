from __future__ import annotations

import numpy as np

from p3_ssl.signal_preprocessing import (
    PREPROCESS_P1,
    P1PreprocessConfig,
    adaptive_bandpass_decimate_np,
    detect_flat_saturation_intervals,
    preprocess_signal,
    signal_quality_report,
)


def test_detect_flat_saturation_intervals_finds_constant_band() -> None:
    signal = np.random.default_rng(1).normal(0, 0.1, size=2048).astype(np.float32)
    signal[700:1300] = 0.25

    intervals = detect_flat_saturation_intervals(signal, min_flat=400, zero_threshold=1.0e-8)

    assert intervals
    assert any(start <= 700 and end >= 1300 for start, end in intervals)


def test_p1_bandpass_preprocess_returns_finite_zscored_signal() -> None:
    fs = 2_000_000.0
    t = np.arange(4096, dtype=np.float32) / fs
    signal = np.sin(2.0 * np.pi * 25_000.0 * t).astype(np.float32)
    cfg = P1PreprocessConfig(mode=PREPROCESS_P1)

    processed = preprocess_signal(signal, output_length=4096, cfg=cfg)

    assert processed.shape == (4096,)
    assert np.isfinite(processed).all()
    assert abs(float(processed.mean())) < 1.0e-5
    assert 0.99 < float(processed.std()) < 1.01


def test_quality_report_rejects_near_constant_signal() -> None:
    cfg = P1PreprocessConfig(mode=PREPROCESS_P1)
    report = signal_quality_report(np.ones(1024, dtype=np.float32), cfg)

    assert report["ok"] is False
    assert report["reject_reason"] == "near_constant"


def test_quality_report_rejects_local_constant_band() -> None:
    signal = np.random.default_rng(3).normal(0, 0.02, size=4096).astype(np.float32)
    signal[1500:2200] = 0.25
    cfg = P1PreprocessConfig(mode=PREPROCESS_P1, saturation_zero_threshold=1.0e-8)

    report = signal_quality_report(signal, cfg)

    assert report["ok"] is False
    assert report["reject_reason"] == "flat_saturation_interval"
    assert report["n_raw_flat_intervals"] >= 1


def test_adaptive_bandpass_decimate_uses_target_length() -> None:
    signal = np.random.default_rng(2).normal(size=4096).astype(np.float32)
    decimated = adaptive_bandpass_decimate_np(signal, target_length=1024)

    assert decimated.shape == (1024,)
    assert np.isfinite(decimated).all()
