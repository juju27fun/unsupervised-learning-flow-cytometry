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
    DENSE_METHOD_EVIDENCE_ID,
    PARAMETERS,
    build_sweep_bank,
    interpolate_parameter_grids,
    load_parameter_grid,
    read_csv,
    select_real_carriers,
    sha256_file,
    write_csv,
)


DEFAULT_V1_DATASET = "yeast-budding-mix-shmoo-background-classification@v1"
DEFAULT_V2_DATASET = "yeast-budding-mix-shmoo-background-classification@v2"
DENSE_QUANTILES = 225


def _record(workspace: Workspace, dataset_id: str) -> dict[str, Any]:
    matches = [
        record.payload
        for record in load_records(workspace)
        if f"{record.payload['id']}@{record.payload['version']}" == dataset_id
    ]
    if len(matches) != 1 or matches[0]["status"] not in {"active", "reference"}:
        raise ValueError(f"Dataset is not a unique usable reference: {dataset_id}")
    return matches[0]


def _verify_method(review_dir: Path) -> str:
    receipt = ReviewStore(review_dir).verify_receipt()
    decisions = json.loads((review_dir / str(receipt["decisions_file"])).read_text())
    decision = decisions["decisions"]["yeast-budding-m2-dense-atlas-method"]["decision"]
    if decision != "approved":
        raise PermissionError("Dense M2 atlas method is not approved")
    return str(receipt["decisions_sha256"])


def _checkpoint_contract(run_dir: Path, expected_model: str) -> tuple[Path, dict[str, Any]]:
    run = json.loads((run_dir / "run.json").read_text())
    checkpoint = run_dir / "best_model.pt"
    if run["status"] != "complete" or run["seed"] != 42:
        raise ValueError(f"Checkpoint run is not the frozen seed-42 result: {run_dir}")
    if run["config"]["model_name"] != expected_model or sha256_file(checkpoint) != run["checkpoint_sha256"]:
        raise ValueError(f"Frozen checkpoint contract mismatch: {run_dir}")
    return checkpoint, run


def _pca(embeddings: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model = PCA(n_components=2, svd_solver="full", random_state=seed)
    coordinates = model.fit_transform(embeddings).astype(np.float32)
    return coordinates, {
        "components": model.components_.astype(np.float32),
        "mean": model.mean_.astype(np.float32),
        "explained_variance": model.explained_variance_.astype(np.float32),
        "explained_variance_ratio": model.explained_variance_ratio_.astype(np.float32),
        "singular_values": model.singular_values_.astype(np.float32),
    }


def _git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the approved 1,800-point-per-parameter M2 atlas bank.")
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

    workspace = Workspace.load()
    decisions_sha256 = _verify_method(args.method_review_dir)
    v1_record = _record(workspace, args.input_dataset)
    v2_record = _record(workspace, args.domain_dataset)
    v1_root = workspace.datasets_root / v1_record["path"]
    v2_root = workspace.datasets_root / v2_record["path"]
    v1_contract = json.loads((v1_root / "dataset-contract.json").read_text())
    v2_contract = json.loads((v2_root / "dataset-contract.json").read_text())
    if v1_contract["input_contract"] != v2_contract["input_contract"]:
        raise ValueError("v1/v2 signal contracts differ")

    anchor = json.loads((args.domain_run_dir / "anchor_event.json").read_text())
    if anchor["event_id"] != "09f788a7473797b794f6:00" or not anchor["fit_valid"]:
        raise ValueError("Unexpected M2 anchor")
    source_grids = load_parameter_grid(args.domain_run_dir / "parameter_quantile_grid.csv")
    quantile_count = 5 if args.smoke else DENSE_QUANTILES
    dense_grids, probabilities = interpolate_parameter_grids(source_grids, count=quantile_count)

    source_rows = read_csv(v1_root / "samples.csv")
    source_signals = np.load(v1_root / "signals.npy", mmap_mode="r")
    carriers = select_real_carriers(source_rows, source_signals)
    amplitude_scale = float(v2_contract["normalization"]["std"]) / float(v1_contract["normalization"]["std"])
    bank, metadata = build_sweep_bank(
        anchor=anchor,
        grids=dense_grids,
        carriers=carriers,
        amplitude_v2_to_v1=amplitude_scale,
        parameter_subset=("log_A_A", "delta_phi_rad") if args.smoke else PARAMETERS,
        quantile_indices=tuple(range(quantile_count)),
        quantile_probabilities=tuple(float(value) for value in probabilities),
        phases=(0.0,) if args.smoke else (0.0, np.pi / 2.0),
        positions=(0.5,) if args.smoke else (0.40, 0.60),
    )
    expected_rows = 22 if args.smoke else 16_208
    if bank.shape != (expected_rows, 4096):
        raise ValueError(f"Unexpected dense bank shape: {bank.shape}")

    resnet_checkpoint, resnet_run = _checkpoint_contract(args.resnet_run_dir, "ResNet1D-XS")
    stft_checkpoint, stft_run = _checkpoint_contract(args.stft_run_dir, "STFT-CNN")
    resnet_model, resnet_payload = load_checkpoint(resnet_checkpoint, args.device)
    stft_model, stft_payload = load_checkpoint(stft_checkpoint, args.device)
    if resnet_payload["input_contract"] != stft_payload["input_contract"] or resnet_payload["input_contract"] != v1_contract:
        raise ValueError("Frozen encoders do not share the generated-signal contract")

    memory_sha256 = hashlib.sha256(bank.tobytes()).hexdigest()
    resnet = encode_signals(resnet_model, bank, device=args.device, batch_size=args.batch_size)
    after_resnet = hashlib.sha256(bank.tobytes()).hexdigest()
    stft = encode_signals(stft_model, bank, device=args.device, batch_size=args.batch_size)
    after_stft = hashlib.sha256(bank.tobytes()).hexdigest()
    if len({memory_sha256, after_resnet, after_stft}) != 1:
        raise AssertionError("Dense signal bank changed between encoders")

    resnet_coordinates, resnet_pca = _pca(resnet["embeddings_l2"], args.seed)
    stft_coordinates, stft_pca = _pca(stft["embeddings_l2"], args.seed)
    sample_ids = np.asarray([row["sample_id"] for row in metadata])
    pca_rows: list[dict[str, Any]] = []
    for model_name, coordinates in (("ResNet1D-XS", resnet_coordinates), ("STFT-CNN", stft_coordinates)):
        pca_rows.extend(
            {
                "model": model_name,
                "sample_id": row["sample_id"],
                "sample_kind": row["sample_kind"],
                "sweep_parameter": row["sweep_parameter"],
                "context_id": row["context_id"],
                "quantile_index": row["quantile_index"],
                "quantile_probability": row["quantile_probability"],
                "pca_1": float(coordinate[0]),
                "pca_2": float(coordinate[1]),
            }
            for row, coordinate in zip(metadata, coordinates, strict=True)
        )

    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "sweep_signals.npy", bank, allow_pickle=False)
    write_csv(args.output_dir / "sweep_metadata.csv", metadata)
    np.savez_compressed(args.output_dir / "resnet_embeddings.npz", sample_ids=sample_ids, **resnet)
    np.savez_compressed(args.output_dir / "stft_embeddings.npz", sample_ids=sample_ids, **stft)
    np.savez_compressed(args.output_dir / "resnet_pca.npz", **resnet_pca)
    np.savez_compressed(args.output_dir / "stft_pca.npz", **stft_pca)
    write_csv(args.output_dir / "dense_pca_points.csv", pca_rows)

    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "method_evidence_id": DENSE_METHOD_EVIDENCE_ID,
        "smoke": args.smoke,
        "quantiles_per_parameter": quantile_count,
        "contexts": len({row["context_id"] for row in metadata}),
        "parameters": 2 if args.smoke else 9,
        "sweep_signals": sum(row["sample_kind"] == "sweep" for row in metadata),
        "context_anchors": sum(row["sample_kind"] == "anchor" for row in metadata),
        "points_per_parameter": quantile_count * len({row["context_id"] for row in metadata}),
        "signal_memory_sha256": memory_sha256,
        "identical_signal_bank_for_both_encoders": True,
        "resnet_pca_explained_variance_ratio_sum": float(np.sum(resnet_pca["explained_variance_ratio"])),
        "stft_pca_explained_variance_ratio_sum": float(np.sum(stft_pca["explained_variance_ratio"])),
        "quantitative_metrics_recomputed": False,
        "sealed_test_accessed": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    provenance = {
        "method_decisions_sha256": decisions_sha256,
        "datasets": {args.input_dataset: v1_record["manifest_sha256"], args.domain_dataset: v2_record["manifest_sha256"]},
        "checkpoints": {"resnet": resnet_run["checkpoint_sha256"], "stft": stft_run["checkpoint_sha256"]},
        "source_domain_manifest_sha256": sha256_file(args.domain_run_dir / "metrics_manifest.json"),
        "dense_probabilities": {"count": quantile_count, "minimum": 0.01, "maximum": 0.99, "interpolation": "linear empirical quantile function"},
        "git_revision": {
            "workspace": _git_revision(workspace.root),
            "unsupervised-learning-flow-cytometry": _git_revision(workspace.root / "unsupervised-learning-flow-cytometry"),
            "particles2SNR-pipeline": _git_revision(workspace.root / "particles2SNR-pipeline"),
        },
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    output_files = (
        "sweep_signals.npy", "sweep_metadata.csv", "resnet_embeddings.npz", "stft_embeddings.npz",
        "resnet_pca.npz", "stft_pca.npz", "dense_pca_points.csv", "summary.json", "provenance.json",
    )
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "files": [{"path": name, "sha256": sha256_file(args.output_dir / name)} for name in output_files],
    }
    (args.output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method_evidence_id": DENSE_METHOD_EVIDENCE_ID,
        "source_run_ids": [args.domain_run_dir.name, args.resnet_run_dir.name, args.stft_run_dir.name],
        "dataset_ids": [args.input_dataset, args.domain_dataset],
        "smoke": args.smoke,
        "device": args.device,
        "sealed_test_accessed": False,
        "outputs": [*output_files, "artifact_manifest.json"],
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
