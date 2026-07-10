#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p3_ssl.official_moment import encode_with_official_moment, load_official_moment


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the official MOMENT HF model loader.")
    parser.add_argument("--model-id", default="AutonLab/MOMENT-1-large")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs" / "hf_cache")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=ROOT / "outputs" / "official_moment_smoke.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_official_moment(
        model_id=args.model_id,
        device=device,
        cache_dir=args.cache_dir,
        seq_len=args.seq_len,
    )

    x = np.random.default_rng(0).normal(0.0, 1.0, size=(args.batch_size, args.seq_len)).astype(np.float32)
    embeddings = encode_with_official_moment(
        model=model,
        signals=x,
        batch_size=args.batch_size,
        device=device,
        reduction="mean",
    )

    payload = {
        "model_id": args.model_id,
        "cache_dir": str(args.cache_dir),
        "device": str(device),
        "seq_len": int(args.seq_len),
        "batch_size": int(args.batch_size),
        "embedding_shape": list(embeddings.shape),
        "embedding_mean": float(np.mean(embeddings)),
        "embedding_std": float(np.std(embeddings)),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
