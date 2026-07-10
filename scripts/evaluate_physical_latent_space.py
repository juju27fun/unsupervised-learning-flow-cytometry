#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from p3_ssl.physical_eval import evaluate_sweep_directory, write_physical_evaluation_report


def parse_models(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank existing synthetic sweep embeddings by physical latent-space fidelity."
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Directory containing synthetic_metadata.csv and per-model embeddings.npz files.",
    )
    parser.add_argument("--models", default=None, help="Comma-separated model directories to evaluate.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--pass-threshold", type=float, default=0.05)
    parser.add_argument(
        "--max-combined-samples",
        type=int,
        default=2000,
        help="Deterministically subsample the all-panel combined score for dense sweeps; use 0 to disable.",
    )
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument("--no-random", action="store_true")
    parser.add_argument("--random-seed", type=int, default=123)
    args = parser.parse_args()

    metrics = evaluate_sweep_directory(
        sweep_dir=args.sweep_dir,
        model_names=parse_models(args.models),
        include_raw=not args.no_raw,
        include_random=not args.no_random,
        k_neighbors=args.k_neighbors,
        random_seed=args.random_seed,
        max_combined_samples=None if args.max_combined_samples == 0 else args.max_combined_samples,
        pass_threshold=args.pass_threshold,
    )
    write_physical_evaluation_report(metrics, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "ranking": metrics["ranking"]}, indent=2))


if __name__ == "__main__":
    main()
