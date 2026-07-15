from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from p3_ssl.study_baselines import load_baseline_data
from p3_ssl.study_evaluation import (
    calibration_metrics,
    cross_recording_retrieval,
    evaluate_linear_probe,
    label_efficiency_auc,
    perturb_signals,
    physical_embedding_diagnostics,
    real_variability_summary,
    robustness_metrics,
)


def _dataset(root: Path) -> None:
    rng = np.random.default_rng(4)
    signals = rng.normal(size=(48, 16)).astype(np.float32)
    np.save(root / "signals.npy", signals)
    fields = [
        "signal_row",
        "source_group",
        "development_split",
        "record_id",
        "capture_block_id",
        "acquisition_id",
        "quality",
        "width_ms",
        "snr_proxy",
        "crop_8192_pad_left",
        "crop_8192_pad_right",
    ]
    with (root / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(48):
            writer.writerow(
                {
                    "signal_row": index,
                    "source_group": "a" if index % 2 == 0 else "b",
                    "development_split": "development_train" if index < 32 else "development_validation",
                    "record_id": f"record-{index}",
                    "capture_block_id": f"block-{index // 2}",
                    "acquisition_id": "session-1",
                    "quality": "strict" if index % 3 else "medium",
                    "width_ms": 0.4 + index / 100.0,
                    "snr_proxy": 2.0 + index,
                    "crop_8192_pad_left": 1 if index % 4 == 0 else 0,
                    "crop_8192_pad_right": 0,
                }
            )


def test_rich_probe_reports_calibration_bootstrap_and_subgroups(tmp_path: Path) -> None:
    _dataset(tmp_path)
    data = load_baseline_data(tmp_path)
    metrics, _ = evaluate_linear_probe(
        np.asarray(data.signals), data, fraction=1.0, seed=3, bootstrap_repeats=20
    )
    assert metrics["grouped_bootstrap"]["status"] == "ok"
    assert len(metrics["grouped_bootstrap"]["metrics"]["macro_f1"]["replicates"]) == 20
    assert set(metrics["subgroups"]["strata"]) == {
        "quality",
        "duration_tertile",
        "snr_tertile",
        "crop_edge_status",
    }
    assert 0.0 <= metrics["calibration"]["expected_calibration_error"] <= 1.0
    assert metrics["probe_optimization"]["converged"] is True
    assert metrics["probe_optimization"]["max_iter"] == 5000
    variability = real_variability_summary(data)
    assert variability["split"] == "development_train"
    assert variability["window_rms_quantiles"]["p95"] >= variability["window_rms_quantiles"]["p05"]


def test_cross_record_retrieval_excludes_same_record() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]])
    rows = [
        {"record_id": f"r{i}", "acquisition_id": "a", "quality": "strict"}
        for i in range(4)
    ]
    result = cross_recording_retrieval(embeddings, rows, np.asarray([0, 0, 1, 1]), neighbors=1)
    assert result["same_record_neighbors"] == 0
    assert result["top1_label_purity"] == 1.0


def test_perturbation_and_robustness_metrics_are_deterministic() -> None:
    signals = np.ones((3, 16), dtype=np.float32)
    first = perturb_signals(
        signals, {"kind": "noise_fraction_signal_std", "value": 0.1}, seed=7
    )
    second = perturb_signals(
        signals, {"kind": "noise_fraction_signal_std", "value": 0.1}, seed=7
    )
    assert np.array_equal(first, second)
    probabilities = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    result = robustness_metrics(signals, signals, probabilities, probabilities)
    assert result["embedding_cosine_distance_mean"] == 0.0
    assert result["prediction_agreement"] == 1.0


def test_calibration_and_label_efficiency_auc() -> None:
    calibration = calibration_metrics(
        np.asarray([0, 1]), np.asarray([[0.8, 0.2], [0.1, 0.9]])
    )
    assert calibration["negative_log_likelihood"] > 0.0
    rows = [
        {"method": "a", "seed": 1, "label_fraction": 0.1, "macro_f1": 0.2},
        {"method": "a", "seed": 1, "label_fraction": 1.0, "macro_f1": 0.4},
    ]
    auc = label_efficiency_auc(rows, "method")
    assert len(auc) == 1
    assert np.isclose(auc[0]["normalized_area"], 0.3)


def test_physical_diagnostics_exclude_paired_latent_neighbors() -> None:
    rng = np.random.default_rng(8)

    def rows(prefix: str, count: int) -> list[dict[str, str]]:
        result = []
        for index in range(count):
            component_count = 1 + index % 2
            result.append(
                {
                    "latent_id": f"{prefix}-{index // 2}",
                    "component_count": str(component_count),
                    "duration_ms": str(0.5 + 0.03 * index),
                    "doppler_khz": str(8.0 + 0.4 * index),
                    "component_separation_ms": str(0.1 + 0.01 * index if component_count == 2 else 0.0),
                    "relative_component_amplitude": str(0.5 + 0.01 * index if component_count == 2 else 0.0),
                    "frequency_separation_khz": str(1.0 + 0.1 * index if component_count == 2 else 0.0),
                    "phase_rad": str(0.1 * index),
                    "event_position_fraction": str(0.3 + 0.01 * index),
                    "snr_db": str(15.0 + index),
                    "target_rms": str(0.8 + 0.01 * index),
                    "baseline_drift": str(-0.1 + 0.01 * index),
                    "sensor_response": str(0.9 + 0.01 * index),
                }
            )
        return result

    train_rows = rows("train", 20)
    validation_rows = rows("validation", 12)
    result = physical_embedding_diagnostics(
        rng.normal(size=(20, 6)),
        rng.normal(size=(12, 6)),
        train_rows,
        validation_rows,
        neighbors=2,
    )
    assert result["scope"].endswith("remains sealed")
    assert result["cross_latent_neighborhood_continuity"]["neighbors"] == 2
    assert set(result["retained_factor_linear_probes"]) >= {"duration_ms", "doppler_khz"}
    assert result["component_count_probe"]["optimization"]["converged"] is True
