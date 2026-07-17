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
from typing import Any

import torch

from p3_ssl.config import (
    load_config,
    validate_local_spectral_study_config,
    validate_mask_ablation_config,
    validate_study_config,
)
from p3_ssl.local_spectral_training import train_local_spectral_s1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _code_tree_is_clean(repo_root: Path) -> bool:
    paths = (
        "p3_ssl",
        "scripts/train_yeast_local_spectral.py",
        "configs/yeast_ssl_local_spectral_v1.yaml",
        "configs/yeast_ssl_rebuild_v1.yaml",
        "configs/yeast_ssl_mask_ablation_v1.yaml",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return not status.strip()


def _validate_overfit(path: Path, config: dict[str, Any]) -> None:
    run_path = path / "run.json"
    metrics_path = path / "metrics.json"
    if (
        _sha256(run_path) != config["study"]["source_overfit_run_json_sha256"]
        or _sha256(metrics_path) != config["study"]["source_overfit_metrics_sha256"]
    ):
        raise ValueError("S1 overfit artifact checksum mismatch")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        run.get("status") != "complete"
        or run.get("sealed_splits_used") != []
        or metrics.get("decision") != "run_s1_training_smoke"
        or not all(metrics.get("gates", {}).values())
        or metrics.get("sealed_splits_used") != []
    ):
        raise ValueError("S1 overfit artifact does not authorize training")


def _validate_c1(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if (
        run.get("run_id") != config["study"]["source_c1_run"]
        or run.get("status") != "complete"
        or run.get("cell") != "C1"
        or run.get("profile") != "full"
        or int(run.get("seed", -1)) != int(config["training"]["seed"])
        or run.get("mask_policy") != config["study"]["source_mask_policy"]
        or run.get("sealed_splits_used") != []
    ):
        raise ValueError("Source C1 run differs from the matched S1 control")
    for name, expected in run.get("outputs", {}).items():
        if _sha256(path / name) != expected:
            raise ValueError(f"Source C1 checksum mismatch: {name}")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the single S1 local spectral cell.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_local_spectral_v1.yaml")
    )
    parser.add_argument("--source-overfit", type=Path, required=True)
    parser.add_argument("--source-c1-run", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    code_tree_clean = _code_tree_is_clean(repo_root)
    if args.profile == "full" and not code_tree_clean:
        raise RuntimeError("Full S1 training requires committed code and configs")
    config = load_config(args.config)
    validate_local_spectral_study_config(config)
    if args.dataset_manifest_sha256 != config["study"]["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the frozen S1 config")
    _validate_overfit(args.source_overfit, config)
    source_c1 = _validate_c1(args.source_c1_run, config)

    base_path = repo_root / config["study"]["base_config"]
    ablation_path = repo_root / config["study"]["source_mask_ablation_config"]
    effective = copy.deepcopy(load_config(base_path))
    ablation = load_config(ablation_path)
    validate_mask_ablation_config(ablation)
    effective["study"]["protocol"] = config["study"]["protocol"]
    effective["training"]["seed"] = int(config["training"]["seed"])
    effective["masking"].update(ablation["policies"][config["study"]["source_mask_policy"]])
    validate_study_config(effective)
    result = train_local_spectral_s1(
        effective_config=effective,
        study_config=config,
        real_root=args.real_root,
        output_dir=args.output_dir,
        profile=args.profile,
        device=torch.device(args.device),
    )
    expected_initial = source_c1["training_contract"]["initial_model_sha256"]
    if result["training_contract"]["control_model_initial_sha256"] != expected_initial:
        raise RuntimeError("Reconstructed C1 initialization differs from the source run")

    metrics_path = args.output_dir / "metrics.json"
    checkpoint_path = args.output_dir / "checkpoint.pt"
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": config["study"]["dataset"],
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(repo_root.parent / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "code_tree_clean": code_tree_clean,
        "protocol": config["study"]["protocol"],
        "cell": "S1",
        "profile": args.profile,
        "seed": int(config["training"]["seed"]),
        "mask_policy": config["study"]["source_mask_policy"],
        "config_sha256": _sha256(args.config),
        "base_config_sha256": _sha256(base_path),
        "source_mask_ablation_config_sha256": _sha256(ablation_path),
        "source_overfit_run_json_sha256": _sha256(args.source_overfit / "run.json"),
        "source_c1_run_json_sha256": _sha256(args.source_c1_run / "run.json"),
        "training_contract": result["training_contract"],
        "dataset_contract": result["contract"],
        "decision": result["decision"],
        "sealed_splits_used": [],
        "outputs": {
            metrics_path.name: _sha256(metrics_path),
            checkpoint_path.name: _sha256(checkpoint_path),
        },
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "gates": result["gates"],
                "controls": result["validation_local_spectral_controls"],
                "embedding_health": result["validation_embedding_health"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
