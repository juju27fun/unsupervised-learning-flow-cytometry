from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from p3_ssl.bead_ssl_v2 import (
    Z8AsymmetricSyntheticDataset,
    Z8RealValidationDataset,
    fixed_class_proportional_monitor_indices,
    load_bead_ssl_v2_config,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs/bead_ssl_z8_v5_v2.yaml"


def _write_synthetic(root: Path) -> None:
    signals = np.arange(3 * 4096, dtype=np.float32).reshape(3, 4096)
    np.save(root / "signals_raw_4096.npy", signals)
    fields = (
        "sample_id",
        "class_name",
        "noise_source_split",
        "t0_fraction",
        "tau_ms",
        "waveform_asymmetry",
    )
    with (root / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                dict(sample_id="a", class_name="2um", noise_source_split="train", t0_fraction=0.5, tau_ms=0.1, waveform_asymmetry=0.0),
                dict(sample_id="b", class_name="4um", noise_source_split="val", t0_fraction=0.5, tau_ms=0.1, waveform_asymmetry=0.2),
                dict(sample_id="c", class_name="10um", noise_source_split="train", t0_fraction=0.5, tau_ms=0.1, waveform_asymmetry=-0.2),
            ]
        )


def test_v2_config_freezes_registered_development_inputs() -> None:
    config = load_bead_ssl_v2_config(CONFIG)
    assert config["study"]["protocol"] == "bead-ssl-comparison-v2"
    assert config["training"]["checkpoint_selection"] == "fixed_final"
    assert "test" in config["study"]["forbidden_splits"]
    assert config["training"]["matched_monitoring"]["samples_per_split"] == 2048


def test_monitor_selection_is_exact_class_proportional_and_deterministic(
    tmp_path: Path,
) -> None:
    signals = np.zeros((10, 4096), dtype=np.float32)
    np.save(tmp_path / "signals_raw_4096.npy", signals)
    fields = (
        "sample_id",
        "class_name",
        "noise_source_split",
        "t0_fraction",
        "tau_ms",
        "waveform_asymmetry",
    )
    classes = ["2um"] * 5 + ["4um"] * 3 + ["10um"] * 2
    with (tmp_path / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, class_name in enumerate(classes):
            writer.writerow(
                dict(
                    sample_id=f"sample-{index}",
                    class_name=class_name,
                    noise_source_split="train",
                    t0_fraction=0.5,
                    tau_ms=0.1,
                    waveform_asymmetry=0.0,
                )
            )
    dataset = Z8AsymmetricSyntheticDataset(
        tmp_path, split="train", normalization="window_zscore"
    )
    first, metadata = fixed_class_proportional_monitor_indices(
        dataset, max_samples=5, seed=42, split_tag="train"
    )
    second, repeated = fixed_class_proportional_monitor_indices(
        dataset, max_samples=5, seed=42, split_tag="train"
    )
    assert first == second
    assert metadata == repeated
    assert metadata["class_counts"] == {"2um": 3, "4um": 1, "10um": 1}
    assert len(first) == 5


def test_v5_loader_splits_by_carrier_and_builds_asymmetric_support(
    tmp_path: Path,
) -> None:
    _write_synthetic(tmp_path)
    train = Z8AsymmetricSyntheticDataset(
        tmp_path, split="train", normalization="window_zscore"
    )
    val = Z8AsymmetricSyntheticDataset(
        tmp_path, split="val", normalization="window_zscore"
    )
    assert len(train) == 2
    assert len(val) == 1
    assert train[0]["sample_id"] == "a"
    assert float(train[0]["signal"].mean()) == pytest.approx(0.0, abs=1e-5)
    mask = val[0]["event_mask"].numpy()
    center = int(round(0.5 * 4095))
    left = center - int(np.flatnonzero(mask)[0])
    right = int(np.flatnonzero(mask)[-1]) - center
    assert right > left


def test_real_loader_excludes_unclear_and_shifts_edge_crop(tmp_path: Path) -> None:
    events = tmp_path / "events"
    signals = tmp_path / "signals"
    events.mkdir()
    (signals / "val/signals").mkdir(parents=True)
    np.save(signals / "val/signals/a.npy", np.arange(16384, dtype=np.float32))
    fields = (
        "event_id",
        "split",
        "class_name",
        "source_filename",
        "source_signal_relative_path",
        "start_sample",
        "end_sample",
        "center_sample",
    )
    with (events / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                dict(event_id="physical", split="val", class_name="2um", source_filename="a.npy", source_signal_relative_path="val/signals/a.npy", start_sample=80, end_sample=300, center_sample=190),
                dict(event_id="unclear", split="val", class_name="unclear", source_filename="a.npy", source_signal_relative_path="val/signals/a.npy", start_sample=400, end_sample=500, center_sample=450),
            ]
        )
    dataset = Z8RealValidationDataset(
        events, signals, split="val", normalization="window_zscore"
    )
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["sample_id"] == "physical"
    assert sample["signal"].shape == (1, 4096)
    assert int(sample["event_mask"].sum()) == 220
