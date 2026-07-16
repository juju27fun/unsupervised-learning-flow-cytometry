from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from p3_ssl.followup_features import (
    extract_feature_families,
    feature_matrix,
    fit_probe,
    load_followup_development,
)


def _rows(count: int) -> list[dict[str, str]]:
    return [
        {
            "width_ms": "1.0",
            "snr_proxy": str(2 + index),
            "energy_concentration": "0.8",
            "phase_coherence": "0.7",
            "n_doppler_peaks": "1",
        }
        for index in range(count)
    ]


def test_feature_families_are_finite_disjoint_and_fusable() -> None:
    rng = np.random.default_rng(4)
    signals = rng.normal(size=(8, 256)).astype(np.float32)
    families, names = extract_feature_families(signals, _rows(len(signals)))
    assert set(families) == set(names)
    assert all(values.shape[0] == len(signals) for values in families.values())
    assert feature_matrix(families).shape[1] == sum(len(value) for value in names.values())
    with pytest.raises(ValueError, match="Unknown"):
        feature_matrix(families, include=("missing",))


def test_linear_probe_detects_deliberate_separation() -> None:
    rng = np.random.default_rng(7)
    train_labels = np.repeat((0, 1), 50)
    validation_labels = np.repeat((0, 1), 40)
    train = np.column_stack(
        [train_labels * 8.0 + rng.normal(scale=0.1, size=100), rng.normal(size=(100, 3))]
    )
    validation = np.column_stack(
        [validation_labels * 8.0 + rng.normal(scale=0.1, size=80), rng.normal(size=(80, 3))]
    )
    metrics, _ = fit_probe(
        train,
        train_labels,
        validation,
        validation_labels,
        probe="linear",
        seed=1,
        class_names=["a", "b"],
    )
    assert metrics["macro_f1"] > 0.95
    assert metrics["converged"] is True


def test_development_loader_does_not_return_followup_test(tmp_path: Path) -> None:
    rows = []
    for index, split in enumerate(("followup_train", "followup_validation", "followup_test")):
        rows.append(
            {
                "signal_row": str(index),
                "development_split": split,
                "source_group": "a",
                "record_id": f"r-{index}",
            }
        )
    with (tmp_path / "development_events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows[:2])
    np.save(tmp_path / "signals.npy", np.arange(12, dtype=np.float32).reshape(3, 4))
    loaded = load_followup_development(tmp_path)
    assert len(loaded.rows) == 2
    assert {row["development_split"] for row in loaded.rows} == {
        "followup_train",
        "followup_validation",
    }


def test_development_loader_rejects_missing_physical_partition(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="physically separated"):
        load_followup_development(tmp_path)
