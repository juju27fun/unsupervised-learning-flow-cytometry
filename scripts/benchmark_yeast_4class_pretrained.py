#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from p3_ssl.study_baselines import LOGISTIC_MAX_ITER, fit_logistic_with_diagnostics, public_encoder_features
from p3_ssl.yeast_4class_classifier import CLASS_NAMES, classification_metrics, load_dataset, sha256_file


def stratified_screening_indices(labels: np.ndarray, indices: np.ndarray, *, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in range(len(CLASS_NAMES)):
        candidates = indices[labels[indices] == class_id]
        if max_per_class > 0 and candidates.size > max_per_class:
            candidates = rng.choice(candidates, size=max_per_class, replace=False)
        selected.extend(int(value) for value in candidates)
    return np.asarray(sorted(selected), dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen MOMENT/PatchTST probe on the yeast four-class dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--method", choices=("moment", "patchtst"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-per-class", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dir", type=Path, default=Path("../.cache/huggingface"))
    parser.add_argument("--method-evidence-id", default="yeast-4class-event-balanced-benchmark-method-r1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")

    data = load_dataset(args.dataset_root)
    train = stratified_screening_indices(
        data.labels,
        data.train_indices,
        max_per_class=args.max_train_per_class,
        seed=args.seed,
    )
    selected = np.concatenate([train, data.validation_indices])
    signals = np.asarray(data.signals[selected], dtype=np.float32)
    features, encoder_metadata = public_encoder_features(
        args.method,
        signals,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        cache_dir=args.cache_dir,
    )
    train_features = features[: train.size]
    validation_features = features[train.size :]
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=LOGISTIC_MAX_ITER,
            class_weight="balanced",
            random_state=args.seed,
        ),
    )
    optimization = fit_logistic_with_diagnostics(probe, train_features, data.labels[train])
    probabilities = probe.predict_proba(validation_features).astype(np.float32)
    metrics = classification_metrics(data.labels[data.validation_indices], probabilities)
    metrics["probe_optimization"] = optimization
    metrics["encoder"] = encoder_metadata
    metrics["n_train_screening"] = int(train.size)
    metrics["n_validation"] = int(data.validation_indices.size)
    metrics["max_train_per_class"] = int(args.max_train_per_class)

    args.output_dir.mkdir(parents=True)
    np.savez_compressed(
        args.output_dir / "validation_embeddings.npz",
        indices=data.validation_indices,
        labels=data.labels[data.validation_indices],
        embeddings=validation_features.astype(np.float32),
        probabilities=probabilities,
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    dataset_manifest = args.dataset_root / "dataset-manifest.json"
    run = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": data.contract["dataset_id"],
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "method_evidence_id": args.method_evidence_id,
        "method": args.method,
        "seed": args.seed,
        "sealed_holdout_accessed": False,
        "config": {
            "transfer": "frozen_encoder_linear_probe",
            "max_train_per_class": args.max_train_per_class,
            "batch_size": args.batch_size,
            "selection_metric": "validation.event_only.macro_f1",
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": args.device,
        },
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"run_id": args.run_id, "event_only": metrics["event_only"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
