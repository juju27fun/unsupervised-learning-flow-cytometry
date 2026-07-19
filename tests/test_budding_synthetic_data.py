from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from p3_ssl.budding_synthetic_data import (
    BuddingSimulationDataset,
    budding_event_support_mask,
    validate_budding_simulation_dataset,
)


def _row(**updates: str) -> dict[str, str]:
    values = {
        "component1_center_ms": "1.5",
        "component2_center_ms": "2.5",
        "sigma1_left_ms": "0.1",
        "sigma1_right_ms": "0.1",
        "sigma2_left_ms": "0.1",
        "sigma2_right_ms": "0.1",
        "shape1": "2.0",
        "shape2": "2.0",
        "relative_amplitude": "0.5",
    }
    values.update(updates)
    return values


def test_budding_event_support_mask_covers_both_components() -> None:
    support = budding_event_support_mask(_row())
    assert support.dtype == np.bool_
    assert support.shape == (4096,)
    assert support[1500]
    assert support[2500]
    assert not support[0]
    assert not support[-1]


def _build_dataset(root: Path, dataset_id: str) -> None:
    root.mkdir()
    np.save(root / "signals.npy", np.zeros((14_000, 4096), dtype=np.float32))
    (root / "dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "generator_id": "test-generator",
                "input_contract": "yeast-event-8192to4096-bandpass-global-v1",
                "views_per_latent": 2,
                "split_signal_counts": {
                    "train": 10_000,
                    "validation": 2_000,
                    "test": 2_000,
                },
            }
        ),
        encoding="utf-8",
    )
    fieldnames = [
        "signal_row",
        "latent_id",
        "view_index",
        "split",
        "generator_model",
        "resolved",
        "mother_radius_relative",
        "bud_radius_ratio",
        "orientation_cosine",
        *list(_row()),
    ]
    with (root / "simulation_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for view in range(2):
            writer.writerow(
                {
                    "signal_row": view,
                    "latent_id": "train-0000000",
                    "view_index": view,
                    "split": "train",
                    "generator_model": "test-generator",
                    "resolved": "True",
                    "mother_radius_relative": "1.0",
                    "bud_radius_ratio": "0.5",
                    "orientation_cosine": "-0.2",
                    **_row(),
                }
            )


def test_dataset_validates_identity_and_returns_paired_views(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    dataset_id = "yeast-budding-simulations-biophysics@v1"
    _build_dataset(root, dataset_id)
    contract = validate_budding_simulation_dataset(root, expected_dataset_id=dataset_id)
    assert contract.signal_shape == (14_000, 4096)
    dataset = BuddingSimulationDataset(root, "train", expected_dataset_id=dataset_id)
    sample = dataset[0]
    assert sample["signals"].shape == (2, 1, 4096)
    assert sample["event_masks"].shape == (2, 4096)
    assert sample["geometry_valid"].tolist() == [True, True, True]
    assert sample["latent_id"] == "train-0000000"
