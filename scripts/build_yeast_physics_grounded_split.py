#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from p3_ssl.yeast_4class_classifier import load_dataset, sha256_file
from p3_ssl.yeast_physics_grounded import build_capture_block_80_20_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the @v2 capture-block-disjoint physics-grounded split."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--candidates", type=int, default=4096)
    args = parser.parse_args()

    data = load_dataset(args.dataset_root)
    if data.contract["dataset_id"] != "yeast-budding-mix-shmoo-background-classification@v2":
        raise ValueError("This frozen split is defined only for the @v2 classification dataset")
    split = build_capture_block_80_20_split(
        data.rows,
        data.labels,
        data.train_indices,
        seed=args.seed,
        candidates=args.candidates,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)

    payload = dict(split.manifest)
    payload["dataset_id"] = data.contract["dataset_id"]
    payload["dataset_manifest_sha256"] = sha256_file(
        args.dataset_root / "dataset-manifest.json"
    )
    payload["assignments"] = {
        "train_core": [data.rows[int(index)]["sample_id"] for index in split.train_core_indices],
        "model_selection": [
            data.rows[int(index)]["sample_id"] for index in split.model_selection_indices
        ],
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    assignments_path = args.output_dir / "split_assignments.csv"
    with assignments_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "sample_id",
            "record_id",
            "capture_block_id",
            "class_name",
            "source_group_original",
            "role",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for role, indices in (
            ("train_core", split.train_core_indices),
            ("model_selection", split.model_selection_indices),
        ):
            for index in indices:
                row = data.rows[int(index)]
                writer.writerow({field: row[field] if field != "role" else role for field in fieldnames})

    run = {
        "schema_version": 1,
        "run_id": payload["split_id"],
        "run_kind": "data_split",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": (
            "build_yeast_physics_grounded_split.py "
            "--dataset-root yeast-budding-mix-shmoo-background-classification@v2 "
            f"--seed {args.seed} --candidates {args.candidates}"
        ),
        "project": "unsupervised-learning-flow-cytometry",
        "repositories": {
            "unsupervised-learning-flow-cytometry": "d1d4dc07343404d350f11b7a2cb68af11adabfb2",
            "workspace": "aa9784d60c91c85bed81dbfabaa302ae07ca588e",
        },
        "dataset": payload["dataset_id"],
        "datasets": [payload["dataset_id"]],
        "source_partition": "development_train",
        "external_holdout_status": "closed",
        "sealed_holdout_accessed": False,
        "method_evidence_id": "yeast-physics-grounded-classifier-method-r1",
        "downstream_blocked_until_method_approval": True,
        "outputs": {
            "split_manifest": "split_manifest.json",
            "split_manifest_sha256": sha256_file(manifest_path),
            "assignments": "split_assignments.csv",
            "assignments_sha256": sha256_file(assignments_path),
        },
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "assignments"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
