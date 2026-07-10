from __future__ import annotations

import csv
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.build_yeast_event_dataset import (
    YeastDetectionConfig,
    build_aligned_signal_at_center,
    build_aligned_512_signal_at_center,
    build_dataset,
    detect_yeast_passages,
)


def synthetic_multi_doppler_passage(length: int = 16384, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fs = 2_000_000.0
    idx = np.arange(length, dtype=np.float32)
    t = idx / fs
    center = length // 2
    envelope = np.exp(-0.5 * np.square((idx - center) / 420.0)).astype(np.float32)
    carrier_a = np.sin(2.0 * np.pi * 22_000.0 * t)
    carrier_b = 0.75 * np.sin(2.0 * np.pi * 34_000.0 * t + 0.45)
    baseline = 0.015 * rng.normal(size=length).astype(np.float32)
    return (baseline + envelope * (carrier_a + carrier_b)).astype(np.float32)


def loose_test_config() -> YeastDetectionConfig:
    return YeastDetectionConfig(
        active_snr_z=2.5,
        strict_min_snr=3.0,
        medium_min_snr=2.0,
        strict_min_concentration=0.08,
        medium_min_concentration=0.04,
        min_width_ms=0.05,
        max_width_ms=2.0,
        max_events_per_signal=3,
    )


def test_detect_yeast_passages_groups_multi_doppler_peaks() -> None:
    signal = synthetic_multi_doppler_passage()
    events, reason = detect_yeast_passages(
        signal,
        sample_id="synthetic",
        signal_path="/tmp/synthetic.npy",
        source_group="synthetic",
        split="test",
        config=loose_test_config(),
    )

    usable = [event for event in events if event.quality in {"strict", "medium"}]
    assert reason == ""
    assert len(usable) == 1
    event = usable[0]
    assert abs(event.center_index - signal.size // 2) < 512
    assert event.n_doppler_peaks >= 2
    assert event.doppler_low_hz < event.doppler_high_hz
    assert event.width_ms > 0.05


def test_build_aligned_signal_at_detected_center_defaults_to_p3_4096() -> None:
    signal = synthetic_multi_doppler_passage()
    aligned = build_aligned_signal_at_center(signal, center_index=signal.size // 2)

    assert aligned.shape == (4096,)
    assert np.isfinite(aligned).all()
    assert abs(float(aligned.mean())) < 1.0e-5
    assert 0.99 < float(aligned.std()) < 1.01


def test_build_aligned_512_signal_at_detected_center_compat_alias() -> None:
    signal = synthetic_multi_doppler_passage()
    aligned = build_aligned_512_signal_at_center(signal, center_index=signal.size // 2)

    assert aligned.shape == (512,)
    assert np.isfinite(aligned).all()
    assert abs(float(aligned.mean())) < 1.0e-5
    assert 0.99 < float(aligned.std()) < 1.01


def test_build_dataset_writes_p3_compatible_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "yeast" / "mix"
    input_dir.mkdir(parents=True)
    signal_path = input_dir / "synthetic.npy"
    np.save(signal_path, synthetic_multi_doppler_passage())
    output_dir = tmp_path / "out"

    args = Namespace(
        input_dir=tmp_path / "yeast",
        output_dir=output_dir,
        include_groups="mix",
        quality="strict",
        split="test",
        class_id=3,
        class_name="yeast",
        max_files=0,
        max_events_per_signal=3,
        max_plot_per_class=500,
        seed=42,
        sampling_frequency_hz=2_000_000.0,
        low_freq_hz=7_000.0,
        high_freq_hz=80_000.0,
        filter_order=4,
        stft_nperseg=512,
        stft_noverlap=384,
        smooth_frames=3,
        active_snr_z=2.5,
        boundary_snr_z=1.5,
        medium_min_snr=2.0,
        strict_min_snr=3.0,
        medium_min_concentration=0.04,
        strict_min_concentration=0.08,
        strict_min_phase_coherence=0.0,
        frequency_peak_height_frac=0.20,
        frequency_peak_prominence_frac=0.08,
        cluster_gap_ms=0.25,
        boundary_pad_ms=0.04,
        min_width_ms=0.05,
        max_width_ms=2.0,
        raw_crop_length=4096,
        output_length=4096,
        write_audit=False,
        audit_max_events=0,
    )

    summary = build_dataset(args)

    assert summary["kept_events"]["n"] == 1
    with np.load(output_dir / "aligned_inputs.npz", allow_pickle=True) as data:
        assert data["signals"].shape == (1, 4096)
        assert data["labels"].tolist() == [3]
        assert data["split"].astype(str).tolist() == ["test"]
        assert data["event_id"].astype(str).tolist()[0].startswith("test/yeast/")
    with (output_dir / "events_metadata.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["class_name"] == "yeast"
