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
from p3_ssl.followup_evaluation import evaluate_followup_checkpoints
from p3_ssl.followup_training import validate_followup_config


def _checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name, Path(raw_path)


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen yeast Week 2 checkpoints.")
    parser.add_argument("--checkpoint", action="append", type=_checkpoint, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_followup_week2_v2.yaml")
    )
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    names = [name for name, _ in args.checkpoint]
    if len(names) != len(set(names)):
        raise ValueError("Checkpoint names must be unique")
    config = load_config(args.config)
    validate_followup_config(config)
    payload = evaluate_followup_checkpoints(
        checkpoints=args.checkpoint,
        config=config,
        real_root=args.real_root,
        simulation_root=args.simulation_root,
        profile=args.profile,
        device=torch.device(args.device),
        output_dir=args.output_dir,
    )
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    output_paths = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path.name != "run.json"
    )
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": f"{config['study']['real_dataset']} + {config['study']['simulation_dataset']}",
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(repo_root.parent / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "profile": args.profile,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_runs": [str(path.parent / "run.json") for _, path in args.checkpoint],
        "sealed_splits_used": [],
        "outputs": {path.name: _sha256(path) for path in output_paths},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "checkpoints": names,
                "probe_rows": len(payload["probe_results"]),
                "sealed_splits_used": [],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
