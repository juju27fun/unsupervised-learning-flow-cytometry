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

from p3_ssl.config import load_config, validate_mask_ablation_config, validate_study_config
from p3_ssl.study_training import train_study_cell


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one yeast patch-mask ablation candidate.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_mask_ablation_v1.yaml")
    )
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    ablation = load_config(args.config)
    validate_mask_ablation_config(ablation)
    candidates = set(ablation["training"]["candidate_policies"])
    if args.policy not in candidates:
        raise ValueError(
            f"Policy {args.policy} is not trainable; expected one of {sorted(candidates)}"
        )
    allowed_seeds = {int(seed) for seed in ablation["training"]["representation_seeds"]}
    if args.seed not in allowed_seeds:
        raise ValueError(f"Seed {args.seed} is not predeclared; expected {sorted(allowed_seeds)}")

    base_path = Path(ablation["study"]["base_config"])
    config = copy.deepcopy(load_config(base_path))
    config["study"]["protocol"] = ablation["study"]["protocol"]
    config["training"]["seed"] = args.seed
    config["masking"].update(ablation["policies"][args.policy])
    validate_study_config(config)
    result = train_study_cell(
        cell="A1",
        config=config,
        real_root=args.real_root,
        simulation_root=args.simulation_root,
        output_dir=args.output_dir,
        profile=args.profile,
        device=torch.device(args.device),
    )

    metrics_path = args.output_dir / "metrics.json"
    checkpoint_path = args.output_dir / "checkpoint.pt"
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
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
        "protocol": ablation["study"]["protocol"],
        "mask_policy": args.policy,
        "profile": args.profile,
        "seed": args.seed,
        "config_sha256": _sha256(args.config),
        "base_config_sha256": _sha256(base_path),
        "sealed_splits_used": result["sealed_splits_used"],
        "outputs": {
            "metrics.json": _sha256(metrics_path),
            "checkpoint.pt": _sha256(checkpoint_path),
        },
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run": run,
                "reconstruction": result["validation_reconstruction_controls"]["real"],
                "embedding_health": result["validation_embedding_health"]["real"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
