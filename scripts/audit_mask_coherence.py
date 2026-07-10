#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

from p3_ssl.config import load_config
from p3_ssl.masking import mask_coherence_summary
from p3_ssl.train_reconstruction import make_dataset


def parse_splits(raw: str) -> list[str]:
    splits = [item.strip() for item in raw.split(",") if item.strip()]
    if not splits:
        raise ValueError("At least one split is required")
    return splits


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"0": None, "0.25": None, "0.5": None, "0.75": None, "0.95": None, "1": None}
    arr = np.asarray(values, dtype=np.float64)
    return {str(q): float(np.quantile(arr, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)}


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        return {"n_masks": 0}
    event_rows = [row for row in rows if int(row["event_points"]) > 0]
    accepted = [float(row["mask_accepted"] == "1") for row in rows]
    fully_hidden = [float(row["fully_hidden_event_count"] != "0") for row in event_rows]
    return {
        "n_masks": len(rows),
        "event_masks": len(event_rows),
        "mask_acceptance_rate": float(np.mean(accepted)) if accepted else None,
        "fully_hidden_event_rate": float(np.mean(fully_hidden)) if fully_hidden else None,
        "event_target_fraction": quantiles([float(row["event_target_fraction"]) for row in event_rows]),
        "event_hidden_fraction": quantiles([float(row["event_hidden_fraction"]) for row in event_rows]),
        "target_event_fraction": quantiles([float(row["target_event_fraction"]) for row in event_rows]),
        "max_event_hidden_fraction": quantiles([float(row["max_event_hidden_fraction"]) for row in event_rows]),
    }


def audit_split(config: dict, manifest: Path, split: str, max_samples: int | None, masks_per_sample: int) -> list[dict[str, str]]:
    try:
        dataset = make_dataset(config, manifest, split)
    except ValueError:
        return []
    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    rows: list[dict[str, str]] = []
    for idx in range(limit):
        for draw in range(masks_per_sample):
            item = dataset[idx]
            summary = mask_coherence_summary(
                item["target_time_mask"].numpy(),
                item["token_time_mask"].numpy(),
                item["event_mask"].numpy(),
            )
            row = {
                "split": split,
                "dataset_index": str(idx),
                "draw": str(draw),
                "sample_id": str(item["sample_id"]),
                "source_kind": str(item["source_kind"]),
                "mask_attempts": str(int(item["mask_attempts"].item())),
                "mask_accepted": "1" if bool(item["mask_accepted"].item()) else "0",
            }
            for key, value in summary.items():
                row[key] = str(value)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SSL mask coherence against labeled particle event spans.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--masks-per-sample", type=int, default=1)
    args = parser.parse_args()

    if args.masks_per_sample <= 0:
        raise ValueError("--masks-per-sample must be positive")
    config = load_config(args.config)
    torch.manual_seed(int(config["experiment"].get("seed", 42)))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    by_split: dict[str, object] = {}
    for split in parse_splits(args.splits):
        rows = audit_split(config, args.manifest, split, args.max_samples, args.masks_per_sample)
        all_rows.extend(rows)
        by_split[split] = summarize_rows(rows)

    csv_path = args.output_dir / "mask_coherence_samples.csv"
    fieldnames = [
        "split",
        "dataset_index",
        "draw",
        "sample_id",
        "source_kind",
        "mask_attempts",
        "mask_accepted",
        "event_count",
        "event_points",
        "target_points",
        "hidden_points",
        "background_target_points",
        "event_target_points",
        "event_hidden_points",
        "event_target_fraction",
        "event_hidden_fraction",
        "target_event_fraction",
        "fully_hidden_event_count",
        "fully_targeted_event_count",
        "max_event_hidden_fraction",
        "max_event_target_fraction",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "config": str(args.config),
        "manifest": str(args.manifest),
        "splits": by_split,
        "overall": summarize_rows(all_rows),
    }
    with (args.output_dir / "mask_coherence_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
