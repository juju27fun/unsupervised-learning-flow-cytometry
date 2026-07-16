from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from p3_ssl.config import load_config
from p3_ssl.followup_week3 import evaluate_week3, plot_week3, write_week3_markdown


CONFIG_PATH = Path(__file__).parents[1] / "configs/yeast_ssl_followup_week3_v1.yaml"


def _signals(count: int, *, seed: int, frequency_shift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(4096, dtype=np.float64) / 1_000_000.0
    values = []
    for index in range(count):
        frequency = 14_000.0 + frequency_shift + 200.0 * (index % 5)
        width = 0.00035 + 0.00002 * (index % 4)
        envelope = np.exp(-0.5 * np.square((time - 0.002) / width))
        signal = envelope * np.cos(2.0 * np.pi * frequency * time + rng.uniform(0, 2 * np.pi))
        signal += 0.08 * rng.normal(size=len(time))
        values.append(signal.astype(np.float32))
    return np.asarray(values)


def _write_real(root: Path, *, include_final: bool = False) -> None:
    root.mkdir()
    signals = _signals(24, seed=1)
    np.save(root / "signals.npy", signals)
    rows = []
    for index in range(len(signals)):
        split = "followup_train" if index < 12 else "followup_validation"
        if include_final and index == len(signals) - 1:
            split = "followup_test"
        rows.append(
            {
                "signal_row": index,
                "development_split": split,
                "source_group": "group_a" if index % 2 == 0 else "group_b",
                "capture_block_id": f"block-{index // 2:03d}",
            }
        )
    with (root / "development_events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_simulation(root: Path, *, seed: int, frequency_shift: float) -> None:
    root.mkdir()
    signals = _signals(24, seed=seed, frequency_shift=frequency_shift)
    np.save(root / "signals.npy", signals)
    rows = []
    for index in range(len(signals)):
        split = "train" if index < 12 else "validation"
        rows.append(
            {
                "signal_row": index,
                "split": split,
                "view_index": 0,
                "latent_id": f"{split}-{index:04d}",
            }
        )
    with (root / "simulation_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _smoke_config() -> dict:
    config = load_config(CONFIG_PATH)
    config["evaluation"]["max_train_per_domain"] = 12
    config["evaluation"]["max_validation_per_domain"] = 12
    config["evaluation"]["sensitivity_calipers"] = [2.0]
    config["evaluation"]["domain_probe"]["models"] = ["linear"]
    config["evaluation"]["domain_probe"]["grouped_bootstrap_repetitions"] = 5
    return config


def test_week3_evaluation_is_complete_and_reportable(tmp_path: Path) -> None:
    real = tmp_path / "real"
    baseline = tmp_path / "baseline"
    corrected = tmp_path / "corrected"
    _write_real(real)
    _write_simulation(baseline, seed=2, frequency_shift=300.0)
    _write_simulation(corrected, seed=3, frequency_shift=0.0)
    payload = evaluate_week3(
        real_root=real,
        baseline_root=baseline,
        corrected_root=corrected,
        config=_smoke_config(),
    )
    assert payload["sealed_splits_used"] == []
    assert len(payload["probe_results"]) == 12
    assert all(row["converged"] for row in payload["probe_results"])
    assert payload["decision"]["representation_training_authorized"] is False
    outputs = plot_week3(payload, tmp_path / "comparison")
    report = tmp_path / "decision.md"
    write_week3_markdown(report, payload)
    assert {path.suffix for path in outputs} == {".png", ".pdf"}
    assert "Representation retraining" in report.read_text(encoding="utf-8")


def test_week3_evaluation_rejects_final_real_split(tmp_path: Path) -> None:
    real = tmp_path / "real"
    baseline = tmp_path / "baseline"
    corrected = tmp_path / "corrected"
    _write_real(real, include_final=True)
    _write_simulation(baseline, seed=2, frequency_shift=300.0)
    _write_simulation(corrected, seed=3, frequency_shift=0.0)
    try:
        evaluate_week3(
            real_root=real,
            baseline_root=baseline,
            corrected_root=corrected,
            config=_smoke_config(),
        )
    except PermissionError as error:
        assert "final splits" in str(error)
    else:
        raise AssertionError("Week 3 must reject final-split metadata")
