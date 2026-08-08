#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from p3_ssl.yeast_4class_classifier import CLASS_NAMES, load_dataset, load_frozen_split, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source-disjoint OOF folds inside the frozen yeast 80% training split.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="yeast-4class-logit-stacker-oof-splits-r1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--method-evidence-id", default="yeast-4class-logit-stacker-method-r1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Immutable output already exists: {args.output_dir}")

    data = load_dataset(args.dataset_root)
    base = load_frozen_split(args.base_split_manifest, data)
    indices = base.train_indices
    labels = data.labels[indices]
    groups = np.asarray([data.rows[int(index)]["record_id"] for index in indices])
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_payloads: list[dict[str, object]] = []
    held_out: list[int] = []
    args.output_dir.mkdir(parents=True)
    for fold, (train_local, validation_local) in enumerate(splitter.split(indices, labels, groups)):
        train_indices = np.sort(indices[train_local])
        validation_indices = np.sort(indices[validation_local])
        train_groups = {data.rows[int(index)]["record_id"] for index in train_indices}
        validation_groups = {data.rows[int(index)]["record_id"] for index in validation_indices}
        if train_groups & validation_groups:
            raise ValueError(f"Source leakage in fold {fold}")
        if set(data.labels[train_indices]) != set(range(len(CLASS_NAMES))) or set(data.labels[validation_indices]) != set(range(len(CLASS_NAMES))):
            raise ValueError(f"Fold {fold} does not contain every class")
        held_out.extend(int(value) for value in validation_indices)
        fold_id = f"yeast-4class-logit-stacker-oof-fold{fold}-s{args.seed}-r1"
        payload = {
            "schema_version": 1,
            "split_id": fold_id,
            "parent_split_id": base.manifest["split_id"],
            "fold": fold,
            "fold_count": args.folds,
            "seed": args.seed,
            "group_key": "record_id",
            "source_partition": "development_train",
            "sealed_holdout_accessed": False,
            "dataset_id": data.contract["dataset_id"],
            "dataset_manifest_sha256": sha256_file(args.dataset_root / "dataset-manifest.json"),
            "assignments": {
                "train": [data.rows[int(index)]["sample_id"] for index in train_indices],
                "validation": [data.rows[int(index)]["sample_id"] for index in validation_indices],
            },
            "counts": {
                "train": {name: int(np.sum(data.labels[train_indices] == class_id)) for class_id, name in enumerate(CLASS_NAMES)},
                "validation": {name: int(np.sum(data.labels[validation_indices] == class_id)) for class_id, name in enumerate(CLASS_NAMES)},
            },
        }
        path = args.output_dir / f"fold_{fold}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fold_payloads.append({"fold": fold, "split_id": fold_id, "manifest": path.name, "sha256": sha256_file(path)})

    if sorted(held_out) != sorted(indices.tolist()) or len(held_out) != len(set(held_out)):
        raise ValueError("OOF validation folds must cover the frozen 80% exactly once")
    validation_ids = set(base.manifest["assignments"]["validation"])
    oof_ids = {data.rows[index]["sample_id"] for index in held_out}
    if validation_ids & oof_ids:
        raise ValueError("Frozen 20% leaked into OOF folds")

    with (args.output_dir / "fold_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "record_id", "class_name", "fold"))
        writer.writeheader()
        fold_by_index = {index: fold for fold, (_, validation_local) in enumerate(splitter.split(indices, labels, groups)) for index in indices[validation_local]}
        for index in sorted(fold_by_index):
            row = data.rows[int(index)]
            writer.writerow({"sample_id": row["sample_id"], "record_id": row["record_id"], "class_name": row["class_name"], "fold": fold_by_index[index]})
    run = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": data.contract["dataset_id"],
        "method_evidence_id": args.method_evidence_id,
        "parent_split_id": base.manifest["split_id"],
        "fold_count": args.folds,
        "seed": args.seed,
        "oof_rows": len(held_out),
        "sealed_holdout_accessed": False,
        "folds": fold_payloads,
        "outputs": ["fold_assignments.csv", *[item["manifest"] for item in fold_payloads]],
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(run, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
