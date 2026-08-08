#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from aeon.classification.convolution_based import MiniRocketClassifier

from p3_ssl.yeast_4class_classifier import (
    CLASS_NAMES,
    classification_metrics,
    load_dataset,
    load_frozen_split,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen yeast MiniRocket separability diagnostic.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-kernels", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-train-rows", type=int, default=0)
    args = parser.parse_args()

    data = load_dataset(args.dataset_root)
    split = load_frozen_split(args.split_manifest, data)
    train_indices = split.train_indices
    if args.max_train_rows:
        per_class = max(1, args.max_train_rows // len(CLASS_NAMES))
        train_indices = np.concatenate(
            [train_indices[data.labels[train_indices] == class_id][:per_class] for class_id in range(len(CLASS_NAMES))]
        )
    train_signals = np.asarray(data.signals[train_indices], dtype=np.float32)[:, None, :]
    validation_signals = np.asarray(data.signals[split.validation_indices], dtype=np.float32)[:, None, :]
    model = MiniRocketClassifier(
        n_kernels=args.n_kernels,
        class_weight="balanced",
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    started = time.perf_counter()
    model.fit(train_signals, data.labels[train_indices])
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = np.asarray(model.predict_proba(validation_signals), dtype=np.float64)
    inference_seconds = time.perf_counter() - started
    metrics = classification_metrics(data.labels[split.validation_indices], probabilities)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_path = args.output_dir / "model.joblib"
    joblib.dump(model, model_path)
    with (args.output_dir / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "record_id", "class_name", "class_id", "predicted_class", *[f"p_{name}" for name in CLASS_NAMES]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for position, source_index in enumerate(split.validation_indices):
            row = data.rows[int(source_index)]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "record_id": row["record_id"],
                    "class_name": row["class_name"],
                    "class_id": int(data.labels[int(source_index)]),
                    "predicted_class": CLASS_NAMES[int(probabilities[position].argmax())],
                    **{f"p_{name}": float(probabilities[position, class_id]) for class_id, name in enumerate(CLASS_NAMES)},
                }
            )
    metric_payload = {
        "validation": metrics,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metric_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": data.contract["dataset_id"],
        "split_id": split.manifest["split_id"],
        "method_evidence_id": "yeast-4class-separability-80-20-method-r1",
        "sealed_holdout_accessed": False,
        "config": {"n_kernels": args.n_kernels, "n_jobs": args.n_jobs, "seed": args.seed},
        "model_sha256": sha256_file(model_path),
        "split_manifest_sha256": sha256_file(args.split_manifest),
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metric_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
