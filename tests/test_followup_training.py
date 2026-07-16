from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from p3_ssl.config import load_config
from p3_ssl.followup_evaluation import evaluate_followup_checkpoints
from p3_ssl.followup_training import (
    convergence_diagnostics,
    train_followup_cell,
    validate_followup_config,
    validate_followup_dataset_contracts,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs/yeast_ssl_followup_week2_v1.yaml"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _small_contract(tmp_path: Path) -> tuple[Path, Path, dict]:
    real = tmp_path / "real"
    simulation = tmp_path / "simulation"
    real.mkdir()
    simulation.mkdir()
    config = copy.deepcopy(load_config(CONFIG_PATH))
    config["data"]["input_length"] = 1024
    config["model"].update(
        {"d_model": 16, "n_heads": 4, "n_layers": 1, "dim_feedforward": 32, "max_tokens": 64}
    )
    config["training"]["profiles"]["smoke"].update(
        {"batch_size": 2, "max_real_events": 2, "max_simulation_latents": 2}
    )
    signals = np.stack(
        [np.sin(np.arange(1024, dtype=np.float32) / (15.0 + index)) for index in range(4)]
    )
    np.save(real / "signals.npy", signals)
    (real / "input_contract.json").write_text(
        json.dumps(
            {
                "contract_id": "yeast-event-4096-followup-train-normalized-v2",
                "output_length": 1024,
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for index, split in enumerate(("followup_train", "followup_train", "followup_validation")):
        rows.append(
            {
                "signal_row": index,
                "development_split": split,
                "event_start_input_index": 300,
                "event_end_input_index": 700,
                "event_id": f"event-{index}",
                "record_id": f"record-{index}",
                "source_group": "mix" if index else "budding",
                "condition_id": "condition",
                "acquisition_id": "session",
                "acquisition_role": "",
                "quality": "strict",
                "capture_block_id": f"block-{index}",
                "width_ms": 0.4,
                "snr_proxy": 12.0 + index,
                "energy_concentration": 0.8,
                "phase_coherence": 0.7,
                "n_doppler_peaks": 1,
            }
        )
    _write_csv(real / "development_events.csv", rows)
    _write_csv(
        real / "sealed_followup_test_events.csv",
        [{**rows[-1], "signal_row": 3, "development_split": "followup_test"}],
    )

    simulation_signals = np.stack(
        [np.sin(np.arange(1024, dtype=np.float32) / (12.0 + index)) for index in range(8)]
    )
    np.save(simulation / "signals.npy", simulation_signals)
    (simulation / "dataset_summary.json").write_text(
        json.dumps({"input_contract": "compatible-4096", "generator_id": "test"}),
        encoding="utf-8",
    )
    simulation_rows = []
    for latent in range(4):
        for view in range(2):
            simulation_rows.append(
                {
                    "signal_row": latent * 2 + view,
                    "latent_id": f"latent-{latent}",
                    "view_index": view,
                    "split": "train" if latent < 2 else "validation",
                    "generator_variant": "base",
                    "duration_ms": 0.8,
                    "doppler_khz": 18.0,
                    "component_count": 1 + (latent % 2),
                    "component_separation_ms": 0.2,
                    "relative_component_amplitude": 0.7,
                    "frequency_separation_khz": 2.0,
                    "event_position_fraction": 0.5,
                    "phase_rad": 0.1 * view,
                    "snr_db": 10.0 + latent,
                    "target_rms": 1.0,
                    "baseline_drift": 0.1,
                    "sensor_response": 1.0,
                }
            )
    _write_csv(simulation / "simulation_metadata.csv", simulation_rows)
    return real, simulation, config


def test_week2_config_preserves_frozen_cells_and_seeds() -> None:
    config = load_config(CONFIG_PATH)
    validate_followup_config(config)
    assert config["study"]["pre_training_addendum"]["outcome_data_seen"] is False
    assert config["training"]["representation_seeds"] == [42, 43, 44]
    assert config["training"]["checkpoint_selection"] == "fixed_final_epoch_no_early_stopping"


def test_followup_contract_never_reads_sealed_metadata(tmp_path: Path) -> None:
    real, simulation, config = _small_contract(tmp_path)
    result = validate_followup_dataset_contracts(real, simulation, config)
    assert result["valid"] is True
    assert result["sealed_metadata_opened"] is False
    rows = list(csv.DictReader((real / "development_events.csv").open(encoding="utf-8")))
    rows.append({**rows[-1], "development_split": "followup_test"})
    _write_csv(real / "development_events.csv", rows)
    contaminated = validate_followup_dataset_contracts(real, simulation, config)
    assert contaminated["valid"] is False
    assert "development metadata contains a final split" in contaminated["errors"]


def test_convergence_requires_complete_finite_decreasing_phases() -> None:
    history = []
    for phase in ("synthetic_pretraining", "real_adaptation"):
        history.extend(
            {"phase": phase, "epoch": index + 1, "epoch_loss": 4.0 - index}
            for index in range(3)
        )
    result = convergence_diagnostics(history, 3, 3)
    assert result["converged"] is True
    history[-1]["epoch_loss"] = float("nan")
    assert convergence_diagnostics(history, 3, 3)["converged"] is False


@pytest.mark.parametrize("cell", ["R0", "R3"])
def test_followup_training_smoke_is_finite_and_never_opens_final(
    tmp_path: Path, cell: str
) -> None:
    real, simulation, config = _small_contract(tmp_path)
    output = tmp_path / f"run-{cell}"
    result = train_followup_cell(
        cell=cell,
        seed=42,
        config=config,
        real_root=real,
        simulation_root=simulation,
        output_dir=output,
        profile="smoke",
        device=torch.device("cpu"),
    )
    assert result["convergence"]["converged"] is True
    assert result["sealed_splits_used"] == []
    assert result["contract"]["sealed_metadata_opened"] is False
    assert torch.load(output / "checkpoint.pt", weights_only=False)["cell"] == cell


def test_followup_evaluation_smoke_reports_health_physics_and_probes(tmp_path: Path) -> None:
    real, simulation, config = _small_contract(tmp_path)
    training_output = tmp_path / "training"
    train_followup_cell(
        cell="R0",
        seed=42,
        config=config,
        real_root=real,
        simulation_root=simulation,
        output_dir=training_output,
        profile="smoke",
        device=torch.device("cpu"),
    )
    payload = evaluate_followup_checkpoints(
        checkpoints=[("r0_s42", training_output / "checkpoint.pt")],
        config=config,
        real_root=real,
        simulation_root=simulation,
        profile="smoke",
        device=torch.device("cpu"),
        output_dir=tmp_path / "evaluation",
    )
    result = payload["checkpoint_results"]["r0_s42"]
    assert result["real_validation_embedding_health"]["effective_rank"] >= 1.0
    assert "retained_factor_linear_probes" in result["physical_retention"]
    assert {row["method"] for row in payload["probe_results"]} == {
        "learned",
        "handcrafted",
        "handcrafted_plus_learned",
    }
    assert payload["sealed_splits_used"] == []
