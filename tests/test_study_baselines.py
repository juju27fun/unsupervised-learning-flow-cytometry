from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from p3_ssl.study_baselines import (
    handcrafted_features,
    load_baseline_data,
    rms_features,
    sample_record_groups,
)


def _dataset(root: Path) -> None:
    signals = np.stack([np.sin(np.arange(64) / (index + 2)) for index in range(24)]).astype(np.float32)
    np.save(root / "signals.npy", signals)
    fields = ["signal_row", "source_group", "development_split", "record_id"]
    with (root / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(24):
            writer.writerow(
                {
                    "signal_row": index,
                    "source_group": "a" if index % 2 == 0 else "b",
                    "development_split": "development_train" if index < 16 else "development_validation",
                    "record_id": f"record-{index // 2}",
                }
            )


def test_features_are_finite_and_rms_is_one_dimensional() -> None:
    signals = np.stack([np.sin(np.arange(128) / 4), np.cos(np.arange(128) / 7)]).astype(np.float32)
    assert rms_features(signals).shape == (2, 1)
    features = handcrafted_features(signals)
    assert features.shape[0] == 2
    assert features.shape[1] > 20
    assert np.isfinite(features).all()


def test_group_sampling_and_loader_never_use_sealed_split(tmp_path: Path) -> None:
    _dataset(tmp_path)
    data = load_baseline_data(tmp_path, max_per_class=4, seed=3)
    selected = sample_record_groups(data.rows, data.labels, data.train_indices, 0.5, seed=4)
    assert selected.size > 0
    assert {data.rows[int(index)]["development_split"] for index in selected} == {
        "development_train"
    }
    assert set(data.labels[selected]) == {0, 1}
