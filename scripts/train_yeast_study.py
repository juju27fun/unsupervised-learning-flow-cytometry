#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from p3_ssl.config import load_config
from p3_ssl.study_training import train_study_cell


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one predeclared yeast SSL rebuild cell.")
    parser.add_argument("--cell", choices=("A1", "A2", "A3", "A4"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/yeast_ssl_rebuild_v1.yaml"))
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        help="Representation-training seed; must be predeclared in the study config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        allowed_seeds = {int(seed) for seed in config["training"]["representation_seeds"]}
        if args.seed not in allowed_seeds:
            raise ValueError(
                f"Seed {args.seed} is not predeclared; expected one of {sorted(allowed_seeds)}"
            )
        config = copy.deepcopy(config)
        config["training"]["seed"] = args.seed
    result = train_study_cell(
        cell=args.cell,
        config=config,
        real_root=args.real_root,
        simulation_root=args.simulation_root,
        output_dir=args.output_dir,
        profile=args.profile,
        device=torch.device(args.device),
        init_checkpoint=args.init_checkpoint,
    )
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": (
            f"{config['study']['real_dataset']} + "
            f"{config['study']['simulation_dataset']}"
        ),
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(workspace_root / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "cell": args.cell,
        "profile": args.profile,
        "seed": int(config["training"]["seed"]),
        "config_sha256": config_hash,
        "sealed_splits_used": result["sealed_splits_used"],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
