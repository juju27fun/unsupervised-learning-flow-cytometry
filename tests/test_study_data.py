from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from p3_ssl.study_data import RealEventDataset, SimulatedLatentDataset, validate_study_dataset_contracts


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_registered_array_contract_loaders(tmp_path: Path) -> None:
    real = tmp_path / "real"
    simulation = tmp_path / "simulation"
    real.mkdir()
    simulation.mkdir()
    np.save(real / "signals.npy", np.ones((1, 4096), dtype=np.float32))
    np.save(simulation / "signals.npy", np.ones((2, 4096), dtype=np.float32))
    (real / "input_contract.json").write_text(
        json.dumps({"contract_id": "yeast-event-8192to4096-bandpass-global-v1", "output_length": 4096})
    )
    (simulation / "dataset_summary.json").write_text(
        json.dumps({"input_contract": "compatible-4096", "generator_id": "test"})
    )
    _write_csv(
        real / "events.csv",
        [
            {
                "signal_row": 0,
                "development_split": "development_train",
                "event_start_input_index": 100,
                "event_end_input_index": 200,
                "event_id": "event",
                "record_id": "record",
                "source_group": "budding",
                "condition_id": "condition",
                "acquisition_id": "session",
                "quality": "strict",
            }
        ],
    )
    simulation_rows = []
    for view in range(2):
        simulation_rows.append(
            {
                "signal_row": view,
                "latent_id": "latent",
                "view_index": view,
                "split": "train",
                "generator_variant": "base",
                "duration_ms": 0.8,
                "doppler_khz": 18.0,
                "component_count": 1,
                "component_separation_ms": 0.0,
                "relative_component_amplitude": 0.0,
                "frequency_separation_khz": 0.0,
                "event_position_fraction": 0.5,
            }
        )
    _write_csv(simulation / "simulation_metadata.csv", simulation_rows)

    contract = validate_study_dataset_contracts(real, simulation)
    real_dataset = RealEventDataset(real, "development_train")
    simulation_dataset = SimulatedLatentDataset(simulation, "train")
    assert contract["valid"] is True
    assert real_dataset[0]["signal"].shape == (1, 4096)
    assert int(real_dataset[0]["event_mask"].sum()) == 100
    assert simulation_dataset[0]["signals"].shape == (2, 1, 4096)
    assert simulation_dataset[0]["continuous_valid"].tolist() == [True, True, False, False, False]


def test_real_dataset_requires_explicit_final_open_for_sealed_split(tmp_path: Path) -> None:
    root = tmp_path / "real"
    root.mkdir()
    np.save(root / "signals.npy", np.ones((1, 4096), dtype=np.float32))
    _write_csv(
        root / "events.csv",
        [
            {
                "signal_row": 0,
                "development_split": "sealed_acquisition_test",
                "event_start_input_index": 100,
                "event_end_input_index": 200,
                "event_id": "event",
                "record_id": "record",
                "source_group": "budding",
                "condition_id": "condition",
                "acquisition_id": "session-ood",
                "acquisition_role": "sealed_ood_test",
                "quality": "strict",
            }
        ],
    )

    with pytest.raises(PermissionError, match="sealed"):
        RealEventDataset(root, "sealed_acquisition_test")
    opened = RealEventDataset(
        root,
        "sealed_acquisition_test",
        allow_sealed_split=True,
    )
    assert opened[0]["acquisition_role"] == "sealed_ood_test"
