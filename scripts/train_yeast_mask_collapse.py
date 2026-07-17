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
from p3_ssl.mask_collapse_training import train_mask_collapse_cell


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered C0/C1 mask-collapse contrast.")
    parser.add_argument("--cell", choices=("C0", "C1"), required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_mask_ablation_v1.yaml")
    )
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    ablation = load_config(args.config)
    validate_mask_ablation_config(ablation)
    branch = ablation["conditional_next_stage"]["branch_pretext_pass_geometry_fail"]
    if args.seed != int(branch["first_stage_seed"]):
        raise ValueError("The first anti-collapse contrast is restricted to the frozen seed")
    report = json.loads(args.selection_report.read_text(encoding="utf-8"))
    if report.get("decision") != "run_preregistered_anti_collapse_contrast":
        raise ValueError("Selection report does not authorize anti-collapse training")
    if report.get("anti_collapse_policy") != args.policy:
        raise ValueError("Requested policy differs from the frozen report selection")

    base_path = Path(ablation["study"]["base_config"])
    config = copy.deepcopy(load_config(base_path))
    config["study"]["protocol"] = f"{ablation['study']['protocol']}-anti-collapse-v1"
    config["training"]["seed"] = args.seed
    config["masking"].update(ablation["policies"][args.policy])
    validate_study_config(config)
    vicreg_weight = float(branch["vicreg_global_weight"])
    result = train_mask_collapse_cell(
        cell=args.cell,
        config=config,
        real_root=args.real_root,
        output_dir=args.output_dir,
        profile=args.profile,
        device=torch.device(args.device),
        vicreg_weight=vicreg_weight,
    )

    metrics_path = args.output_dir / "metrics.json"
    checkpoint_path = args.output_dir / "checkpoint.pt"
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": config["study"]["real_dataset"],
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(workspace_root / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "protocol": config["study"]["protocol"],
        "mask_policy": args.policy,
        "cell": args.cell,
        "profile": args.profile,
        "seed": args.seed,
        "vicreg_weight": vicreg_weight if args.cell == "C1" else 0.0,
        "selection_report_sha256": _sha256(args.selection_report),
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
