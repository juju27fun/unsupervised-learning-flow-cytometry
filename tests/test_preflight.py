from __future__ import annotations

import csv

from p3_ssl.preflight import preflight_hybrid_run


def _write_csv(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_ready_config(tmp_path) -> tuple[dict, object]:
    real_manifest = tmp_path / "real_manifest.csv"
    _write_csv(
        real_manifest,
        [
            {"split": "train", "id": "a", "signal_path": "a.npy", "label_path": "a.txt"},
            {"split": "val", "id": "b", "signal_path": "b.npy", "label_path": "b.txt"},
            {"split": "test", "id": "c", "signal_path": "c.npy", "label_path": "c.txt"},
        ],
    )
    event_manifest = tmp_path / "particles2snr" / "event_manifest.csv"
    _write_csv(event_manifest, [{"source_filename": "a.npy", "snr_db": "1"}])
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    (sweep / "synthetic_metadata.csv").write_text("panel,A,fD,phi,t0,tau\n")
    (sweep / "synthetic_signals_encoded.npz").write_bytes(b"placeholder")
    for model in ("moment_official", "patchtst_pretrained", "conv1dgap_same_input_3class"):
        (sweep / model).mkdir()
        (sweep / model / "embeddings.npz").write_bytes(b"placeholder")
    checkpoint = tmp_path / "reference" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    reference = tmp_path / "reference"
    (reference / "eval_val").mkdir()
    (reference / "eval_test").mkdir()
    (reference / "eval_val" / "metrics.json").write_text("{}")
    (reference / "eval_test" / "metrics.json").write_text("{}")
    config = {
        "simulation": {
            "n_synthetic": {"full": 3000},
            "source": "internal",
            "particles2snr_event_manifest": str(event_manifest),
        },
        "hybrid_sampling": {"max_real_rows": {"full": None}},
        "contrastive_loss": {},
        "reconstruction_loss": {},
        "physics_metrics": {},
        "data": {"split_train": "train", "split_val": "val", "split_test": "test"},
        "baseline_assessment": {
            "sweep_dir": {"full": str(sweep)},
            "max_combined_samples": {"full": 2000},
            "reconstruction_only_checkpoint": str(checkpoint),
        },
        "classic_assessment": {
            "max_real_rows": {"full": 1000},
            "label_fractions": {"full": [0.1, 1.0]},
        },
        "real_adaptation": {
            "epochs": {"full": 5},
            "reconstruction_reference_root": str(reference),
        },
        "training": {
            "epochs": {"full": 20},
            "batch_size": {"full": 32},
        },
        "paths": {"output_root": str(tmp_path / "runs")},
    }
    return config, real_manifest


def test_preflight_hybrid_run_passes_when_required_inputs_exist(tmp_path) -> None:
    config, real_manifest = _make_ready_config(tmp_path)
    report = preflight_hybrid_run(
        config=config,
        real_manifest=real_manifest,
        profile="full",
        device="cuda",
        simulation_source="internal,particles2snr_pipeline",
        cuda_available=True,
    )
    assert report["run_ready"] is True
    assert not report["required_failures"]


def test_preflight_hybrid_run_reports_missing_cuda(tmp_path) -> None:
    config, real_manifest = _make_ready_config(tmp_path)
    report = preflight_hybrid_run(
        config=config,
        real_manifest=real_manifest,
        profile="full",
        device="cuda",
        simulation_source="internal,particles2snr_pipeline",
        cuda_available=False,
    )
    assert report["run_ready"] is False
    failures = {item["name"] for item in report["required_failures"]}
    assert "requested_device_available" in failures


def test_preflight_hybrid_run_reports_missing_particles2snr_manifest(tmp_path) -> None:
    config, real_manifest = _make_ready_config(tmp_path)
    config["simulation"]["particles2snr_event_manifest"] = str(tmp_path / "missing.csv")
    report = preflight_hybrid_run(
        config=config,
        real_manifest=real_manifest,
        profile="full",
        device="cpu",
        simulation_source="internal,particles2snr_pipeline",
        cuda_available=False,
    )
    failures = {item["name"] for item in report["required_failures"]}
    assert "particles2snr_event_manifest_present" in failures
