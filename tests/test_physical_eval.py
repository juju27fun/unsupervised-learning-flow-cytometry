from __future__ import annotations

import csv
import json

import numpy as np

from p3_ssl.physical_eval import (
    evaluate_encoder_on_sweep_directory,
    evaluate_sweep_directory,
    load_sweep_physics_by_panel,
    merge_reference_and_candidate_rankings,
    write_physical_evaluation_report,
)


def _write_metadata(path) -> None:
    rows = []
    for panel in ("amplitude_A", "doppler_fD"):
        for i, value in enumerate(np.linspace(0.0, 1.0, 6, dtype=np.float32)):
            rows.append(
                {
                    "scenario": "single_particle",
                    "panel": panel,
                    "index": str(i),
                    "sweep_param": "A" if panel == "amplitude_A" else "fD",
                    "color_value": str(value),
                    "A": str(1.0 + value if panel == "amplitude_A" else 1.0),
                    "fD": str(10.0 + 10.0 * value if panel == "doppler_fD" else 10.0),
                    "phi": "0.25",
                    "t0": "0.5",
                    "tau": "0.1",
                }
            )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_sweep_physics_converts_units(tmp_path) -> None:
    metadata = tmp_path / "synthetic_metadata.csv"
    _write_metadata(metadata)
    physics = load_sweep_physics_by_panel(metadata)
    assert set(physics) == {"amplitude_A", "doppler_fD"}
    assert np.isclose(physics["amplitude_A"][0, 1], 10.0 / 2.048)
    assert np.isclose(physics["amplitude_A"][0, 4], 0.1 * 2.048)
    assert np.isnan(physics["amplitude_A"][0, 5])


def test_evaluate_sweep_directory_ranks_models_and_writes_report(tmp_path) -> None:
    _write_metadata(tmp_path / "synthetic_metadata.csv")
    signal_payload = {}
    good_payload = {}
    bad_payload = {}
    for panel in ("amplitude_A", "doppler_fD"):
        values = np.linspace(0.0, 1.0, 6, dtype=np.float32)
        signal_payload[f"{panel}_signals"] = np.stack([values, values**2], axis=1)
        good_payload[f"{panel}_embeddings"] = np.stack([values, values**2, values**3], axis=1)
        bad_payload[f"{panel}_embeddings"] = np.ones((6, 3), dtype=np.float32)
    np.savez_compressed(tmp_path / "synthetic_signals_encoded.npz", **signal_payload)
    (tmp_path / "good_model").mkdir()
    (tmp_path / "bad_model").mkdir()
    np.savez_compressed(tmp_path / "good_model" / "embeddings.npz", **good_payload)
    np.savez_compressed(tmp_path / "bad_model" / "embeddings.npz", **bad_payload)

    metrics = evaluate_sweep_directory(
        tmp_path,
        model_names=["good_model", "bad_model"],
        include_raw=True,
        include_random=True,
        k_neighbors=2,
        max_combined_samples=8,
        pass_threshold=0.25,
    )
    ranked = [row["model"] for row in metrics["ranking"]]
    assert "good_model" in ranked
    assert "raw_signal" in ranked
    assert "random_embedding" in ranked
    assert metrics["models"]["good_model"]["combined_samples"] == 8
    assert metrics["models"]["good_model"]["combined"]["pass_threshold"] == 0.25
    assert metrics["models"]["good_model"]["combined"]["physical_score"] > metrics["models"]["bad_model"]["combined"]["physical_score"]

    output_dir = tmp_path / "report"
    write_physical_evaluation_report(metrics, output_dir)
    assert (output_dir / "physical_metrics.json").is_file()
    assert (output_dir / "physical_ranking.md").is_file()
    saved = json.loads((output_dir / "physical_metrics.json").read_text())
    assert saved["ranking"]


def test_evaluate_encoder_on_sweep_directory_and_merge(tmp_path) -> None:
    _write_metadata(tmp_path / "synthetic_metadata.csv")
    signal_payload = {}
    model_payload = {}
    for panel in ("amplitude_A", "doppler_fD"):
        values = np.linspace(0.0, 1.0, 6, dtype=np.float32)
        signal_payload[f"{panel}_signals"] = np.stack([values, values**2], axis=1)
        model_payload[f"{panel}_embeddings"] = np.ones((6, 3), dtype=np.float32)
    np.savez_compressed(tmp_path / "synthetic_signals_encoded.npz", **signal_payload)
    (tmp_path / "reference_model").mkdir()
    np.savez_compressed(tmp_path / "reference_model" / "embeddings.npz", **model_payload)

    reference = evaluate_sweep_directory(
        tmp_path,
        model_names=["reference_model"],
        include_raw=True,
        include_random=False,
        k_neighbors=2,
        max_combined_samples=8,
    )

    def encode(signals: np.ndarray) -> np.ndarray:
        values = signals[:, 0]
        return np.stack([values, values**2, values**3], axis=1).astype(np.float32)

    candidate = evaluate_encoder_on_sweep_directory(
        tmp_path,
        encode_panel=encode,
        model_name="candidate",
        input_length=2,
        k_neighbors=2,
        max_combined_samples=8,
    )
    merged = merge_reference_and_candidate_rankings(reference, candidate)
    assert "candidate" in merged["models"]
    assert any(row["model"] == "candidate" for row in merged["ranking"])
