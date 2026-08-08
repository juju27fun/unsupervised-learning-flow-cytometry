#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.visual_review_store import ReviewStore
from p3_ssl.yeast_4class_classifier import encode_signals, load_checkpoint
from p3_ssl.yeast_budding_m2_sweep import (
    CLASS_NAMES,
    METHOD_EVIDENCE_ID,
    PARAMETERS,
    analyze_embeddings,
    build_sweep_bank,
    computation_fingerprint,
    load_parameter_grid,
    read_csv,
    select_real_carriers,
    sha256_file,
    write_csv,
)


DEFAULT_V1_DATASET = "yeast-budding-mix-shmoo-background-classification@v1"
DEFAULT_V2_DATASET = "yeast-budding-mix-shmoo-background-classification@v2"


def _record(workspace: Workspace, dataset_id: str) -> dict[str, Any]:
    matches = [
        record.payload
        for record in load_records(workspace)
        if f"{record.payload['id']}@{record.payload['version']}" == dataset_id
    ]
    if len(matches) != 1 or matches[0]["status"] not in {"active", "reference"}:
        raise ValueError(f"Dataset is not a unique usable reference: {dataset_id}")
    return matches[0]


def _git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _verify_method(review_dir: Path) -> str:
    receipt = ReviewStore(review_dir).verify_receipt()
    decisions_path = review_dir / str(receipt["decisions_file"])
    decisions = json.loads(decisions_path.read_text())
    decision = decisions["decisions"]["yeast-budding-m2-resnet-stft-latent-sweep-method"]["decision"]
    if decision != "approved":
        raise PermissionError("M2 latent sweep method is not approved")
    return str(receipt["decisions_sha256"])


def _checkpoint_contract(run_dir: Path, expected_model: str) -> tuple[Path, dict[str, Any]]:
    run = json.loads((run_dir / "run.json").read_text())
    checkpoint = run_dir / "best_model.pt"
    if run["status"] != "complete" or run["seed"] != 42:
        raise ValueError(f"Checkpoint run is not the frozen seed-42 result: {run_dir}")
    if run["config"]["model_name"] != expected_model:
        raise ValueError(f"Unexpected model in {run_dir}")
    if sha256_file(checkpoint) != run["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {run_dir}")
    return checkpoint, run


def _save_encoded(path: Path, sample_ids: np.ndarray, encoded: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, sample_ids=sample_ids, **encoded)


def _pca_payload(embeddings_l2: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pca = PCA(n_components=2, svd_solver="full", random_state=seed)
    coordinates = pca.fit_transform(embeddings_l2).astype(np.float32)
    payload = {
        "components": pca.components_.astype(np.float32),
        "mean": pca.mean_.astype(np.float32),
        "explained_variance": pca.explained_variance_.astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
        "singular_values": pca.singular_values_.astype(np.float32),
    }
    return coordinates, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the approved paired M2 ResNet/STFT latent sweep.")
    parser.add_argument("--input-dataset", default=DEFAULT_V1_DATASET)
    parser.add_argument("--domain-dataset", default=DEFAULT_V2_DATASET)
    parser.add_argument("--domain-run-dir", type=Path, required=True)
    parser.add_argument("--resnet-run-dir", type=Path, required=True)
    parser.add_argument("--stft-run-dir", type=Path, required=True)
    parser.add_argument("--method-review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    workspace = Workspace.load()
    receipt_sha256 = _verify_method(args.method_review_dir)
    v1_record = _record(workspace, args.input_dataset)
    v2_record = _record(workspace, args.domain_dataset)
    v1_root = workspace.datasets_root / v1_record["path"]
    v2_root = workspace.datasets_root / v2_record["path"]
    v1_contract = json.loads((v1_root / "dataset-contract.json").read_text())
    v2_contract = json.loads((v2_root / "dataset-contract.json").read_text())
    if v1_contract["input_contract"] != v2_contract["input_contract"]:
        raise ValueError("v1/v2 signal preprocessing contracts differ")
    if v1_contract["output_length"] != 4096 or v1_contract["bandpass_hz"] != [5000.0, 100000.0]:
        raise ValueError("Unexpected classifier input contract")

    domain_manifest = json.loads((args.domain_run_dir / "metrics_manifest.json").read_text())
    if domain_manifest["analysis_run_id"] != args.domain_run_dir.name:
        raise ValueError("Domain run identity mismatch")
    anchor = json.loads((args.domain_run_dir / "anchor_event.json").read_text())
    if anchor["event_id"] != "09f788a7473797b794f6:00" or not anchor["fit_valid"]:
        raise ValueError("Unexpected M2 anchor")
    grids = load_parameter_grid(args.domain_run_dir / "parameter_quantile_grid.csv")

    source_rows = read_csv(v1_root / "samples.csv")
    source_signals = np.load(v1_root / "signals.npy", mmap_mode="r")
    carriers = select_real_carriers(source_rows, source_signals)
    amplitude_scale = (
        float(v2_contract["normalization"]["std"])
        / float(v1_contract["normalization"]["std"])
    )
    bank, metadata = build_sweep_bank(
        anchor=anchor,
        grids=grids,
        carriers=carriers,
        amplitude_v2_to_v1=amplitude_scale,
        parameter_subset=("log_A_A", "delta_phi_rad") if args.smoke else PARAMETERS,
        quantile_indices=(0, 15, 30) if args.smoke else tuple(range(31)),
        phases=(0.0,) if args.smoke else (0.0, np.pi / 2.0),
        positions=(0.5,) if args.smoke else (0.40, 0.60),
    )
    expected_rows = 14 if args.smoke else 2240
    if bank.shape != (expected_rows, 4096):
        raise ValueError(f"Unexpected generated bank shape: {bank.shape}")

    resnet_checkpoint, resnet_run = _checkpoint_contract(args.resnet_run_dir, "ResNet1D-XS")
    stft_checkpoint, stft_run = _checkpoint_contract(args.stft_run_dir, "STFT-CNN")
    resnet_model, resnet_payload = load_checkpoint(resnet_checkpoint, args.device)
    stft_model, stft_payload = load_checkpoint(stft_checkpoint, args.device)
    if resnet_payload["input_contract"] != stft_payload["input_contract"]:
        raise ValueError("Frozen checkpoints do not share an input contract")
    if resnet_payload["input_contract"] != v1_contract:
        raise ValueError("Generated signal contract differs from checkpoint contract")

    signal_memory_sha256 = hashlib.sha256(bank.tobytes()).hexdigest()
    resnet_encoded = encode_signals(resnet_model, bank, device=args.device, batch_size=args.batch_size)
    after_resnet_sha256 = hashlib.sha256(bank.tobytes()).hexdigest()
    stft_encoded = encode_signals(stft_model, bank, device=args.device, batch_size=args.batch_size)
    after_stft_sha256 = hashlib.sha256(bank.tobytes()).hexdigest()
    if len({signal_memory_sha256, after_resnet_sha256, after_stft_sha256}) != 1:
        raise AssertionError("Signal bank changed between encoders")
    sample_ids = np.asarray([row["sample_id"] for row in metadata])

    resnet_points, resnet_parameters = analyze_embeddings(
        model_name="ResNet1D-XS",
        metadata=metadata,
        embeddings_l2=resnet_encoded["embeddings_l2"],
        probabilities=resnet_encoded["probabilities"],
    )
    stft_points, stft_parameters = analyze_embeddings(
        model_name="STFT-CNN",
        metadata=metadata,
        embeddings_l2=stft_encoded["embeddings_l2"],
        probabilities=stft_encoded["probabilities"],
    )
    resnet_coordinates, resnet_pca = _pca_payload(resnet_encoded["embeddings_l2"], args.seed)
    stft_coordinates, stft_pca = _pca_payload(stft_encoded["embeddings_l2"], args.seed)
    for rows, coordinates in ((resnet_points, resnet_coordinates), (stft_points, stft_coordinates)):
        for row, coordinate in zip(rows, coordinates, strict=True):
            row["pca_1"] = float(coordinate[0])
            row["pca_2"] = float(coordinate[1])

    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "sweep_signals.npy", bank, allow_pickle=False)
    write_csv(args.output_dir / "sweep_metadata.csv", metadata)
    _save_encoded(args.output_dir / "resnet_embeddings.npz", sample_ids, resnet_encoded)
    _save_encoded(args.output_dir / "stft_embeddings.npz", sample_ids, stft_encoded)
    np.savez_compressed(args.output_dir / "resnet_pca.npz", **resnet_pca)
    np.savez_compressed(args.output_dir / "stft_pca.npz", **stft_pca)
    write_csv(args.output_dir / "per_point_metrics.csv", [*resnet_points, *stft_points])
    write_csv(args.output_dir / "per_parameter_metrics.csv", [*resnet_parameters, *stft_parameters])
    carriers_payload = [
        {
            "carrier_id": carrier.carrier_id,
            "record_id": carrier.record_id,
            "signal_row": carrier.signal_row,
            "energy_quantile": carrier.energy_quantile,
            "normalized_values_sha256": __import__("hashlib").sha256(carrier.values.tobytes()).hexdigest(),
        }
        for carrier in carriers
    ]
    (args.output_dir / "carriers.json").write_text(json.dumps(carriers_payload, indent=2, sort_keys=True) + "\n")

    aggregate = {}
    for model_name, rows in (("ResNet1D-XS", resnet_parameters), ("STFT-CNN", stft_parameters)):
        aggregate[model_name] = {
            metric: float(np.median([row[metric] for row in rows]))
            for metric in (
                "excursion_median",
                "path_efficiency_median",
                "jump_ratio_median",
                "monotonicity_median",
                "nuisance_dispersion_median",
            )
        }
    metrics = {
        "schema_version": 1,
        "method_evidence_id": METHOD_EVIDENCE_ID,
        "run_id": args.run_id,
        "smoke": args.smoke,
        "generated_signals": int(bank.shape[0]),
        "sweep_signals": int(sum(row["sample_kind"] == "sweep" for row in metadata)),
        "context_anchors": int(sum(row["sample_kind"] == "anchor" for row in metadata)),
        "parameters": list(PARAMETERS if not args.smoke else ("log_A_A", "delta_phi_rad")),
        "contexts": len({row["context_id"] for row in metadata}),
        "signal_sha256": sha256_file(args.output_dir / "sweep_signals.npy"),
        "signal_memory_sha256": signal_memory_sha256,
        "identical_signal_bank_for_both_encoders": True,
        "embedding_dimension": 512,
        "class_names": list(CLASS_NAMES),
        "aggregate": aggregate,
        "sealed_test_accessed": False,
        "automatic_winner_selected": False,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    provenance = {
        "code": {
            "runner": "scripts/run_yeast_budding_m2_latent_sweep.py",
            "runner_sha256": sha256_file(Path(__file__)),
            "library": "p3_ssl/yeast_budding_m2_sweep.py",
            "library_sha256": sha256_file(Path(__file__).resolve().parents[1] / "p3_ssl" / "yeast_budding_m2_sweep.py"),
        },
        "datasets": {
            args.input_dataset: v1_record["manifest_sha256"],
            args.domain_dataset: v2_record["manifest_sha256"],
        },
        "inputs": {
            "domain_metrics_manifest_sha256": sha256_file(args.domain_run_dir / "metrics_manifest.json"),
            "method_decisions_sha256": receipt_sha256,
            "resnet_checkpoint_sha256": resnet_run["checkpoint_sha256"],
            "stft_checkpoint_sha256": stft_run["checkpoint_sha256"],
        },
        "metric_definitions": {
            "excursion": "Maximum cosine distance from the matching context anchor.",
            "path_efficiency": "Euclidean endpoint distance divided by cumulative adjacent-step length in L2-normalized 512-D.",
            "jump_ratio": "Maximum adjacent step divided by the median adjacent step.",
            "monotonicity": "Absolute Spearman correlation between quantile order and projection on the endpoint direction.",
            "nuisance_dispersion": "Cosine distance to the same-quantile centroid across the eight contexts.",
        },
        "parameters": {
            "seed": args.seed,
            "signal_contract": "raw M2 at 2MHz -> order-4 zero-phase 5-100kHz -> resample 8192-to-4096 -> v1 normalized domain",
            "common_phase_offsets_rad": [0.0, np.pi / 2.0] if not args.smoke else [0.0],
            "position_fractions": [0.40, 0.60] if not args.smoke else [0.5],
            "carrier_policy": "budding development_train backgrounds nearest q50 and q75 energy, distinct record_id",
            "amplitude_v2_to_v1": amplitude_scale,
        },
        "git_revision": {
            "workspace": _git_revision(workspace.root),
            "unsupervised-learning-flow-cytometry": _git_revision(workspace.root / "unsupervised-learning-flow-cytometry"),
            "particles2SNR-pipeline": _git_revision(workspace.root / "particles2SNR-pipeline"),
        },
    }
    fingerprint = computation_fingerprint(provenance)
    metric_files = (
        "metrics.json",
        "per_point_metrics.csv",
        "per_parameter_metrics.csv",
        "resnet_pca.npz",
        "stft_pca.npz",
    )
    metrics_manifest = {
        "schema_version": 1,
        "analysis_run_id": args.run_id,
        "computation_fingerprint": fingerprint,
        "computation_provenance": provenance,
        "metrics": [
            {"path": name, "sha256": sha256_file(args.output_dir / name), "computation_fingerprint": fingerprint}
            for name in metric_files
        ],
    }
    (args.output_dir / "metrics_manifest.json").write_text(json.dumps(metrics_manifest, indent=2, sort_keys=True) + "\n")
    output_files = (
        "sweep_signals.npy",
        "sweep_metadata.csv",
        "resnet_embeddings.npz",
        "stft_embeddings.npz",
        "resnet_pca.npz",
        "stft_pca.npz",
        "per_point_metrics.csv",
        "per_parameter_metrics.csv",
        "carriers.json",
        "metrics.json",
        "metrics_manifest.json",
    )
    artifact_manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "files": [{"path": name, "sha256": sha256_file(args.output_dir / name)} for name in output_files],
    }
    (args.output_dir / "artifact_manifest.json").write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n")
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method_evidence_id": METHOD_EVIDENCE_ID,
        "source_run_ids": [args.domain_run_dir.name, args.resnet_run_dir.name, args.stft_run_dir.name],
        "dataset_ids": [args.input_dataset, args.domain_dataset],
        "smoke": args.smoke,
        "device": args.device,
        "sealed_test_accessed": False,
        "outputs": [*output_files, "artifact_manifest.json"],
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
