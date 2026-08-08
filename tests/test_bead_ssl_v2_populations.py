from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from p3_ssl.bead_ssl_v2_populations import load_v5_population


def test_v5_population_uses_carrier_split_and_tau_frequency_targets(tmp_path: Path) -> None:
    rows = [
        {
            "sample_id": "a",
            "noise_source_split": "train",
            "noise_source_relative_path": "train/a.npy",
            "tau_ms": "0.2",
            "frequency_khz": "20",
        },
        {
            "sample_id": "b",
            "noise_source_split": "val",
            "noise_source_relative_path": "val/b.npy",
            "tau_ms": "0.3",
            "frequency_khz": "30",
        },
    ]
    with (tmp_path / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.save(tmp_path / "signals_raw_4096.npy", np.stack([np.arange(8), np.arange(8) + 1]).astype(np.float32))
    population = load_v5_population(tmp_path, split="val")
    assert population.ids.tolist() == ["b"]
    assert population.groups.tolist() == ["val/b.npy"]
    assert population.labels.tolist() == [[0.3, 30.0]]
