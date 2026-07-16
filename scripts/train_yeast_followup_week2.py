#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from p3_ssl.config import load_config
from p3_ssl.followup_training import train_followup_cell


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one frozen yeast Week 2 R0-R3 cell.")
    parser.add_argument("--cell", choices=("R0", "R1", "R2", "R3"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_followup_week2_v1.yaml")
    )
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    result = train_followup_cell(
        cell=args.cell,
        seed=args.seed,
        config=config,
        real_root=args.real_root,
        simulation_root=args.simulation_root,
        output_dir=args.output_dir,
        profile=args.profile,
        device=torch.device(args.device),
    )
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    metrics_path = args.output_dir / "metrics.json"
    checkpoint_path = args.output_dir / "checkpoint.pt"
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": f"{config['study']['real_dataset']} + {config['study']['simulation_dataset']}",
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(workspace_root / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "cell": args.cell,
        "profile": args.profile,
        "seed": args.seed,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_frozen_protocol": config["study"]["source_frozen_protocol"],
        "sealed_splits_used": result["sealed_splits_used"],
        "outputs": {
            "metrics.json": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            "checkpoint.pt": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        },
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run": run, "convergence": result["convergence"]}, indent=2))


if __name__ == "__main__":
    main()
