from __future__ import annotations

import csv

import numpy as np

from p3_ssl.data import SSLManifestDataset, parse_yolo_1d_labels
from p3_ssl.hybrid_physics import (
    build_hybrid_manifest,
    generate_synthetic_manifest,
    load_particles2snr_event_estimates,
    summarize_hybrid_manifest,
)


def test_generate_synthetic_manifest_has_physics_columns_and_masks(tmp_path) -> None:
    manifest = generate_synthetic_manifest(
        tmp_path,
        n_samples=12,
        input_length=128,
        seed=7,
        normalization="none",
        include_two_particle=True,
    )
    with manifest.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12
    required = {"signal_path", "label_path", "A", "fD_khz", "phi_rad", "t0_fraction", "tau_ms", "snr_db"}
    assert required <= set(rows[0])
    assert {row["source"] for row in rows} == {"synthetic"}
    assert any(row["scenario"] == "two_particles" for row in rows)
    single = next(row for row in rows if row["scenario"] == "single_particle")
    two = next(row for row in rows if row["scenario"] == "two_particles")
    assert parse_yolo_1d_labels(single["label_path"]).shape[0] == int(single["particle_count"])
    assert parse_yolo_1d_labels(two["label_path"]).shape[0] == int(two["particle_count"]) == 2
    signal = np.load(rows[0]["signal_path"])
    assert signal.shape == (128,)

    ds = SSLManifestDataset(
        manifest,
        split="train",
        input_length_raw=128,
        decimation_factor=1,
        input_length_ssl=128,
        patch_size=4,
        patch_stride=4,
        mask_ratio=0.2,
        min_block_length=4,
        max_block_length=16,
        seed=1,
    )
    item = ds[0]
    assert item["event_mask"].any()
    assert item["physics_params"].shape == (6,)
    assert bool(item["has_physics_params"])


def test_build_hybrid_manifest_caps_real_rows_across_splits(tmp_path) -> None:
    synthetic = generate_synthetic_manifest(
        tmp_path / "synthetic",
        n_samples=6,
        input_length=64,
        seed=3,
        normalization="none",
        include_two_particle=False,
    )
    real_manifest = tmp_path / "real.csv"
    fieldnames = ["split", "id", "signal_path", "label_path", "source_kind", "n_labels"]
    with real_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "val", "test"):
            for idx in range(4):
                writer.writerow(
                    {
                        "split": split,
                        "id": f"{split}_{idx}",
                        "signal_path": f"{split}_{idx}.npy",
                        "label_path": "",
                        "source_kind": "particle",
                        "n_labels": "1",
                    }
                )
    hybrid = build_hybrid_manifest(synthetic, real_manifest, tmp_path / "hybrid.csv", max_real_rows=6)
    with hybrid.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    real_rows = [row for row in rows if row["source"] == "real"]
    assert len(real_rows) == 6
    assert {row["split"] for row in real_rows} == {"train", "val", "test"}


def test_build_hybrid_manifest_enriches_real_rows_from_particles2snr_events(tmp_path) -> None:
    synthetic = generate_synthetic_manifest(
        tmp_path / "synthetic",
        n_samples=4,
        input_length=64,
        seed=4,
        normalization="none",
        include_two_particle=False,
    )
    real_manifest = tmp_path / "real.csv"
    with real_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "id", "signal_path", "label_path", "source_kind", "n_labels"])
        writer.writeheader()
        writer.writerow(
            {
                "split": "val",
                "id": "sample_a",
                "signal_path": "signals/sample_a.npy",
                "label_path": "",
                "source_kind": "particle",
                "n_labels": "1",
            }
        )
    event_manifest = tmp_path / "event_manifest.csv"
    fields = ["source_filename", "snr_db", "frequency", "center", "passage_time_ms", "width_ms"]
    with event_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "source_filename": "sample_a.npy",
                "snr_db": "2.0",
                "frequency": "12000.0",
                "center": "0.25",
                "passage_time_ms": "0.40",
                "width_ms": "0.80",
            }
        )
        writer.writerow(
            {
                "source_filename": "sample_a.npy",
                "snr_db": "5.0",
                "frequency": "24000.0",
                "center": "0.50",
                "passage_time_ms": "0.30",
                "width_ms": "0.60",
            }
        )
    estimates = load_particles2snr_event_estimates(event_manifest)
    assert estimates["sample_a"]["particle_count"] == "2"

    hybrid = build_hybrid_manifest(
        synthetic,
        real_manifest,
        tmp_path / "hybrid_enriched.csv",
        particles2snr_event_manifest=event_manifest,
    )
    with hybrid.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    real = next(row for row in rows if row["source"] == "real")
    assert real["physics_param_source"] == "particles2snr_event_manifest"
    assert real["particle_count"] == "2"
    assert float(real["fD_khz"]) == 24.0
    assert float(real["t0_fraction"]) == 0.5
    assert float(real["tau_ms"]) == 0.3
    assert float(real["snr_db"]) == 5.0
    assert real["A"] == ""
    assert real["phi_rad"] == ""


def test_summarize_hybrid_manifest_reports_physics_coverage(tmp_path) -> None:
    synthetic = generate_synthetic_manifest(
        tmp_path / "synthetic",
        n_samples=4,
        input_length=64,
        seed=5,
        normalization="none",
        include_two_particle=False,
    )
    real_manifest = tmp_path / "real.csv"
    with real_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "id", "signal_path", "label_path", "source_kind", "n_labels"])
        writer.writeheader()
        writer.writerow(
            {
                "split": "val",
                "id": "sample_b",
                "signal_path": "signals/sample_b.npy",
                "label_path": "",
                "source_kind": "particle",
                "n_labels": "1",
            }
        )
    event_manifest = tmp_path / "event_manifest.csv"
    fields = ["source_filename", "snr_db", "frequency", "center", "passage_time_ms", "width_ms"]
    with event_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "source_filename": "sample_b.npy",
                "snr_db": "7.0",
                "frequency": "30000.0",
                "center": "0.40",
                "passage_time_ms": "0.50",
                "width_ms": "0.50",
            }
        )
    hybrid = build_hybrid_manifest(
        synthetic,
        real_manifest,
        tmp_path / "hybrid_summary.csv",
        particles2snr_event_manifest=event_manifest,
    )

    summary = summarize_hybrid_manifest(hybrid)
    assert summary["total_rows"] == 5
    assert summary["by_source"] == {"synthetic": 4, "real": 1}
    assert summary["by_physics_param_source"] == {
        "synthetic_internal": 4,
        "particles2snr_event_manifest": 1,
    }
    assert summary["rows_with_any_physics_param"] == 5
    assert summary["rows_with_all_physics_params"] == 4
    assert summary["physics_param_coverage"]["A"] == 4
    assert summary["physics_param_coverage"]["phi_rad"] == 4
    assert summary["physics_param_coverage"]["fD_khz"] == 5
