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
from torch.utils.data import DataLoader

from p3_ssl.config import load_config
from p3_ssl.study_data import RealEventDataset, SEALED_REAL_SPLITS
from p3_ssl.study_model import YeastStudyModel, YeastStudyModelConfig
from p3_ssl.study_training import evaluate_reconstruction_controls


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
    parser = argparse.ArgumentParser(
        description="Audit frozen yeast reconstruction checkpoints against trivial controls."
    )
    parser.add_argument("--checkpoint", action="append", type=_checkpoint, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_rebuild_v2.yaml")
    )
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="development_validation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.split in SEALED_REAL_SPLITS:
        raise ValueError(f"Refusing sealed split for diagnostic audit: {args.split}")
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    config = load_config(args.config)
    dataset = RealEventDataset(args.real_root, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device(args.device)
    rows: dict[str, object] = {}

    for name, checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        seed = int(checkpoint["seed"])
        model = YeastStudyModel(YeastStudyModelConfig(**checkpoint["model_config"]))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        metrics = evaluate_reconstruction_controls(
            model,
            loader,
            config,
            seed=seed + 9_000_000,
            simulation=False,
            device=device,
        )
        zero_mse = float(metrics["zero_masked_mse"])
        model_mse = float(metrics["model_masked_mse"])
        target_rms = float(metrics["target_rms_on_mask"])
        output_rms = float(metrics["model_output_rms_on_mask"])
        rows[name] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "cell": checkpoint["cell"],
            "seed": seed,
            "metrics": metrics,
            "gates": {
                "beats_zero": model_mse < zero_mse,
                "output_rms_fraction_of_target": output_rms / target_rms,
                "nontrivial_amplitude_0p10": output_rms / target_rms >= 0.10,
            },
        }

    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "dataset": config["study"]["real_dataset"],
        "split": args.split,
        "sealed_splits_used": [],
        "n_samples": len(dataset),
        "checkpoints": rows,
    }
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
        "config_sha256": _sha256(args.config),
        "sealed_splits_used": [],
        "outputs": {"metrics.json": _sha256(metrics_path)},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
