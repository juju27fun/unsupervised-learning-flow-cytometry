#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from p3_ssl.data import labels_to_event_mask, parse_yolo_1d_labels
from p3_ssl.decimation import crop_or_pad, decimate_signal, ensure_1d_signal, normalize_signal
from p3_ssl.visualization import write_patch_stride_audit_pdf


DEFAULT_CONFIGS = [
    {"patch_size": 4, "patch_stride": 4},
    {"patch_size": 4, "patch_stride": 2},
    {"patch_size": 8, "patch_stride": 8},
    {"patch_size": 8, "patch_stride": 4},
    {"patch_size": 16, "patch_stride": 8},
]


def read_rows(manifest: Path, max_samples: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with manifest.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= max_samples:
                break
    return rows


def load_sample(row: dict[str, str], input_length_raw: int, decimation_factor: int, input_length_ssl: int) -> dict[str, np.ndarray | str]:
    signal_path = Path(row["signal_path"])
    signal = ensure_1d_signal(np.load(signal_path))
    signal = crop_or_pad(signal, input_length_raw, mode="center")
    signal = decimate_signal(signal, decimation_factor, method="mean")
    signal = crop_or_pad(signal, input_length_ssl, mode="center")
    signal = normalize_signal(signal, mode="window_zscore")
    labels = parse_yolo_1d_labels(row.get("label_path") or None)
    event_mask = labels_to_event_mask(labels, input_length_ssl)
    return {
        "sample_id": row.get("id", Path(row["signal_path"]).stem),
        "signal": signal,
        "event_mask": event_mask,
    }


def parse_configs(raw: str | None) -> list[dict[str, int]]:
    if not raw:
        return DEFAULT_CONFIGS
    configs = []
    for item in raw.split(","):
        patch, stride = item.split("/")
        configs.append({"patch_size": int(patch), "patch_stride": int(stride)})
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a P3_SSL patch/stride/masking audit PDF.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--configs", default=None, help="Comma-separated patch/stride pairs, e.g. 4/4,4/2,8/8")
    parser.add_argument("--input-length-raw", type=int, default=16384)
    parser.add_argument("--decimation-factor", type=int, default=4)
    parser.add_argument("--input-length-ssl", type=int, default=4096)
    parser.add_argument("--guard-points", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_rows(args.manifest, args.max_samples)
    samples = [
        load_sample(row, args.input_length_raw, args.decimation_factor, args.input_length_ssl)
        for row in rows
    ]
    write_patch_stride_audit_pdf(
        output_pdf=args.output,
        samples=samples,
        configs=parse_configs(args.configs),
        guard_points=args.guard_points,
        seed=args.seed,
    )
    print(f"Wrote audit PDF to {args.output}")


if __name__ == "__main__":
    main()
