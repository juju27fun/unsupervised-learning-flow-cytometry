#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from p3_ssl.yeast_4class_classifier import (
    build_source_disjoint_80_20_split,
    load_dataset,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the yeast source-disjoint 80/20 development split.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    data = load_dataset(args.dataset_root)
    split = build_source_disjoint_80_20_split(
        data.rows,
        data.labels,
        data.train_indices,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = dict(split.manifest)
    payload["dataset_id"] = data.contract["dataset_id"]
    payload["dataset_manifest_sha256"] = sha256_file(args.dataset_root / "dataset-manifest.json")
    payload["assignments"] = {
        "train": [data.rows[int(index)]["sample_id"] for index in split.train_indices],
        "validation": [data.rows[int(index)]["sample_id"] for index in split.validation_indices],
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (args.output_dir / "split_assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "record_id", "class_name", "split"))
        writer.writeheader()
        for split_name, indices in (("train", split.train_indices), ("validation", split.validation_indices)):
            for index in indices:
                row = data.rows[int(index)]
                writer.writerow(
                    {
                        "sample_id": row["sample_id"],
                        "record_id": row["record_id"],
                        "class_name": row["class_name"],
                        "split": split_name,
                    }
                )

    run = {
        "schema_version": 1,
        "run_id": payload["split_id"],
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": payload["dataset_id"],
        "sealed_holdout_accessed": False,
        "method_evidence_id": "yeast-4class-separability-80-20-method-r1",
        "outputs": {
            "split_manifest": "split_manifest.json",
            "split_manifest_sha256": sha256_file(manifest_path),
            "assignments": "split_assignments.csv",
        },
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "assignments"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
