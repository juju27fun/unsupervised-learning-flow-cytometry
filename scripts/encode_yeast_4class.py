#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from p3_ssl.yeast_4class_classifier import encode_signals, load_checkpoint, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode normalized 4096-sample signals with a frozen yeast classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--sample-ids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    signals = np.load(args.signals, mmap_mode="r")
    model, payload = load_checkpoint(args.checkpoint, args.device)
    encoded = encode_signals(model, signals, device=args.device, batch_size=args.batch_size)
    sample_ids = np.arange(signals.shape[0]).astype(str) if args.sample_ids is None else np.load(args.sample_ids, allow_pickle=False).astype(str)
    if sample_ids.shape != (signals.shape[0],):
        raise ValueError("sample IDs must align one-to-one with signals")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, sample_ids=sample_ids, **encoded)
    manifest = {
        "schema_version": 1, "checkpoint_sha256": sha256_file(args.checkpoint),
        "input_contract": payload["input_contract"], "class_names": payload["class_names"],
        "input_shape": list(signals.shape), "output_shapes": {key: list(value.shape) for key, value in encoded.items()},
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
