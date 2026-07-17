from __future__ import annotations

import csv
import hashlib
import json

import pytest

from p3_ssl.local_spectral_reporting import (
    audit_registered_dataset,
    summarize_local_spectral_result,
)


def _metrics() -> tuple[dict, dict]:
    region = {
        name: {part: {"mse": value} for part in ("all", "event", "boundary", "background")}
        for name, value in (("model", 0.06), ("feature_of_interpolation", 0.02))
    }
    s1 = {
        "history": [
            {"local_spectral_prediction": 0.3},
            {"local_spectral_prediction": 0.075},
        ],
        "decision": "end_objective_rescue_negative",
        "gates": {
            "beats_zero": True,
            "beats_train_constant": True,
            "beats_feature_of_interpolation": False,
            "effective_rank": True,
            "mean_pairwise_cosine": True,
        },
        "validation_local_spectral_controls": {
            "model_masked_feature_mse": 0.06,
            "zero_masked_feature_mse": 0.45,
            "train_constant_masked_feature_mse": 0.25,
            "feature_of_interpolation_masked_feature_mse": 0.02,
            "model_output_rms_fraction_of_target": 0.94,
            "regions": region,
        },
        "validation_embedding_health": {
            "effective_rank": 14.8,
            "mean_off_diagonal_cosine_similarity": 0.70,
        },
    }
    c1 = {
        "validation_embedding_health": {
            "real": {
                "effective_rank": 9.6,
                "mean_off_diagonal_cosine_similarity": 0.935,
            }
        }
    }
    return s1, c1


def test_summary_distinguishes_optimization_from_interpolation_gate() -> None:
    summary = summarize_local_spectral_result(*_metrics())
    assert summary["training"]["relative_reduction"] == pytest.approx(0.75)
    assert summary["controls"]["s1_to_interpolation_ratio"] == pytest.approx(3.0)
    assert summary["regions"]["event"]["s1_to_interpolation_ratio"] == pytest.approx(3.0)
    assert summary["interpretation"] == {
        "optimization_worked": True,
        "beats_zero_and_constant": True,
        "beats_interpolation": False,
        "geometry_passed": True,
        "utility_authorized": False,
    }


def test_dataset_audit_hashes_manifest_and_rejects_split_overlap(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    events = dataset / "events.csv"
    with events.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("development_split", "record_id", "capture_block_id"),
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "development_split": "development_train",
                    "record_id": "r1",
                    "capture_block_id": "b1",
                },
                {
                    "development_split": "development_validation",
                    "record_id": "r2",
                    "capture_block_id": "b2",
                },
            ]
        )
    digest = hashlib.sha256(events.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"path": "events.csv", "sha256": digest, "size": events.stat().st_size}) + "\n"
    )
    audit = audit_registered_dataset(dataset, manifest)
    assert audit["event_count"] == 2
    assert (
        audit["cross_split_overlaps"]["record_id"][
            "development_train__development_validation"
        ]
        == 0
    )

    text = events.read_text().replace("r2", "r1")
    events.write_text(text)
    manifest.write_text(
        json.dumps(
            {
                "path": "events.csv",
                "sha256": hashlib.sha256(events.read_bytes()).hexdigest(),
                "size": events.stat().st_size,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="overlap"):
        audit_registered_dataset(dataset, manifest)
