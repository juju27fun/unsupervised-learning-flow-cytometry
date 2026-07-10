#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YEAST_ROOT = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones-4096_20260701" / "yeast_passage_events_p3_4096"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _stratified_split(labels: np.ndarray, train_frac: float, val_frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    split = np.full(labels.shape[0], "test", dtype=object)
    for class_id in sorted(set(int(v) for v in labels.tolist())):
        idx = np.flatnonzero(labels == class_id)
        rng.shuffle(idx)
        n = int(idx.size)
        n_train = max(1, int(round(n * train_frac))) if n >= 3 else max(0, n - 1)
        n_val = max(1, int(round(n * val_frac))) if n - n_train >= 2 else max(0, n - n_train - 1)
        split[idx[:n_train]] = "train"
        split[idx[n_train : n_train + n_val]] = "val"
        split[idx[n_train + n_val :]] = "test"
    return split.astype(str)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    events_path = args.yeast_root / "events_metadata.csv"
    inputs_path = args.yeast_root / "aligned_inputs.npz"
    rows = _read_rows(events_path)
    with np.load(inputs_path, allow_pickle=True) as data:
        signals = data["signals"].astype(np.float32, copy=False)
    if len(rows) != int(signals.shape[0]):
        raise ValueError(f"events/signals length mismatch: {len(rows)} vs {signals.shape[0]}")

    include = set(_parse_csv(args.include_labels))
    label_values = [row.get(args.label_column, "") for row in rows]
    if include:
        selected_idx = [i for i, value in enumerate(label_values) if value in include]
    else:
        selected_idx = [i for i, value in enumerate(label_values) if value]
    if not selected_idx:
        raise ValueError("No yeast events selected")

    counts: dict[str, int] = {}
    for i in selected_idx:
        counts[label_values[i]] = counts.get(label_values[i], 0) + 1
    kept_labels = [name for name, count in sorted(counts.items()) if count >= args.min_events_per_class]
    class_to_id = {name: idx for idx, name in enumerate(kept_labels)}
    selected_idx = [i for i in selected_idx if label_values[i] in class_to_id]
    if args.max_events_per_class > 0:
        rng = np.random.default_rng(args.seed)
        limited: list[int] = []
        for name in kept_labels:
            class_indices = np.asarray([i for i in selected_idx if label_values[i] == name], dtype=np.int64)
            if class_indices.size > args.max_events_per_class:
                class_indices = rng.choice(class_indices, size=args.max_events_per_class, replace=False)
            limited.extend(int(i) for i in class_indices.tolist())
        selected_idx = sorted(limited)
    if len(class_to_id) < 2:
        raise ValueError(f"Need at least two yeast labels for supervised Conv1D-GAP training, got {counts}")

    selected = np.asarray(selected_idx, dtype=np.int64)
    out_signals = signals[selected].astype(np.float32, copy=False)
    labels = np.asarray([class_to_id[label_values[i]] for i in selected_idx], dtype=np.int64)
    split = _stratified_split(labels, args.train_frac, args.val_frac, args.seed)
    out_rows = [rows[i] for i in selected_idx]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "yeast_conv1dgap_dataset.npz",
        signals=out_signals,
        labels=labels,
        split=split,
        event_id=np.asarray([row.get("event_id", "") for row in out_rows]),
        class_name=np.asarray([kept_labels[int(label)] for label in labels.tolist()]),
        class_names=np.asarray(kept_labels),
    )

    metadata_fields = [
        "event_id",
        "sample_id",
        "split",
        "class_id",
        "class_name",
        "source_group",
        "quality",
        "width_ms",
        "snr_proxy",
        "phase_coherence",
        "doppler_peak_hz",
        "signal_path",
    ]
    with (args.output_dir / "events_metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata_fields)
        writer.writeheader()
        for row, class_id, split_name in zip(out_rows, labels.tolist(), split.tolist()):
            writer.writerow(
                {
                    "event_id": row.get("event_id", ""),
                    "sample_id": row.get("sample_id", ""),
                    "split": split_name,
                    "class_id": int(class_id),
                    "class_name": kept_labels[int(class_id)],
                    "source_group": row.get("source_group", ""),
                    "quality": row.get("quality", ""),
                    "width_ms": row.get("width_ms", ""),
                    "snr_proxy": row.get("snr_proxy", ""),
                    "phase_coherence": row.get("phase_coherence", ""),
                    "doppler_peak_hz": row.get("doppler_peak_hz", ""),
                    "signal_path": row.get("signal_path", ""),
                }
            )

    summary = {
        "yeast_root": str(args.yeast_root),
        "output_dir": str(args.output_dir),
        "label_column": args.label_column,
        "class_names": kept_labels,
        "class_counts": {name: int(np.sum(labels == class_to_id[name])) for name in kept_labels},
        "split_counts": {name: int(np.sum(split == name)) for name in ["train", "val", "test"]},
        "input_length": int(out_signals.shape[1]),
        "note": "Each row is one isolated detected yeast passage crop. Labels default to source_group for a supervised yeast Conv1D-GAP control.",
    }
    with (args.output_dir / "dataset_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a supervised isolated-yeast dataset for Conv1D-GAP training.")
    parser.add_argument("--yeast-root", type=Path, default=DEFAULT_YEAST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "yeast_conv1dgap_dataset")
    parser.add_argument("--label-column", default="source_group")
    parser.add_argument("--include-labels", default="", help="Optional comma-separated labels to keep, e.g. budding,mix,shmoo,shmoo2.")
    parser.add_argument("--min-events-per-class", type=int, default=20)
    parser.add_argument("--max-events-per-class", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    summary = build_dataset(build_parser().parse_args())
    print(json.dumps({"output_dir": summary["output_dir"], "class_counts": summary["class_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
