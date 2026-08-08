#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from p3_ssl.yeast_4class_classifier import CLASS_NAMES, encode_signals, load_checkpoint, sha256_file


ABSOLUTE_PARAMETERS = ("log_A_A", "fD_A", "log_tau_A", "snr_db")
RELATIVE_PARAMETERS = ("log_B_over_A", "delta_t0", "delta_fD", "delta_phi", "log_tau_B_over_tau_A")
NUISANCE_COLUMNS = ("common_phase", "position", "carrier_class", "noise_id")
REQUIRED_METADATA = ("sample_id", *ABSOLUTE_PARAMETERS, *RELATIVE_PARAMETERS, *NUISANCE_COLUMNS)


def read_metadata(path: Path, expected_rows: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(REQUIRED_METADATA).issubset(reader.fieldnames):
            missing = sorted(set(REQUIRED_METADATA) - set(reader.fieldnames or []))
            raise ValueError(f"Two-particle metadata contract is missing: {missing}")
        rows = list(reader)
    if len(rows) != expected_rows or len({row["sample_id"] for row in rows}) != expected_rows:
        raise ValueError("Metadata must align one-to-one with signals and contain unique sample IDs")
    for row in rows:
        values = [float(row[name]) for name in (*ABSOLUTE_PARAMETERS, *RELATIVE_PARAMETERS, "common_phase", "position")]
        if not np.isfinite(values).all():
            raise ValueError("Two-particle parameters must be finite")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen 4-class yeast inference adapter for future two-particle sweeps.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True, help="Normalized float32 [N,4096] array from the historical generator.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    signals = np.load(args.signals, mmap_mode="r")
    rows = read_metadata(args.metadata, signals.shape[0])
    model, payload = load_checkpoint(args.checkpoint, args.device)
    encoded = encode_signals(model, signals, device=args.device, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, sample_ids=np.asarray([row["sample_id"] for row in rows]), **encoded)
    manifest = {
        "schema_version": 1,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "input_contract": payload["input_contract"],
        "class_names": list(CLASS_NAMES),
        "absolute_parameters": list(ABSOLUTE_PARAMETERS),
        "relative_parameters": list(RELATIVE_PARAMETERS),
        "controlled_nuisances": list(NUISANCE_COLUMNS),
        "input_shape": list(signals.shape),
        "output_shapes": {key: list(value.shape) for key, value in encoded.items()},
        "phase4_status": "interface_only_no_sweep_executed",
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
