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

from p3_ssl.config import (
    load_config,
    validate_mask_ablation_config,
    validate_mask_collapse_config,
    validate_study_config,
)
from p3_ssl.mask_collapse_training import train_mask_collapse_cell


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _code_tree_is_clean(repo_root: Path) -> bool:
    paths = (
        "p3_ssl",
        "scripts/train_yeast_mask_collapse.py",
        "configs/yeast_ssl_mask_collapse_v1.yaml",
        "configs/yeast_ssl_mask_ablation_v1.yaml",
        "configs/yeast_ssl_rebuild_v1.yaml",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return not status.strip()


def _validate_selection_report(
    path: Path,
    *,
    source_config: dict,
    source_config_path: Path,
    policy: str,
    seed: int,
) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    report_manifest_path = path.parent / "run.json"
    report_manifest = json.loads(report_manifest_path.read_text(encoding="utf-8"))
    if path.name != "metrics.json" or report_manifest.get("status") != "complete":
        raise ValueError("Selection report must be a completed manifested metrics.json")
    if report_manifest.get("outputs", {}).get("metrics.json") != _sha256(path):
        raise ValueError("Selection report checksum differs from its manifest")
    if report_manifest.get("config_sha256") != _sha256(source_config_path):
        raise ValueError("Selection report was not generated from the frozen source config")
    if report.get("protocol") != source_config["study"]["protocol"]:
        raise ValueError("Selection report protocol differs from the source mask study")
    if report.get("sealed_splits_used") or report_manifest.get("sealed_splits_used"):
        raise ValueError("Selection report must not use sealed splits")
    if report.get("decision") != "run_preregistered_anti_collapse_contrast":
        raise ValueError("Selection report does not authorize anti-collapse training")
    if report.get("anti_collapse_policy") != policy:
        raise ValueError("Requested policy differs from the frozen report selection")
    rows = report.get("rows", [])
    expected = set(source_config["training"]["candidate_policies"])
    if {row.get("policy") for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("Selection report does not contain one row per candidate policy")
    if any(int(row.get("seed", -1)) != seed or row.get("profile") != "full" for row in rows):
        raise ValueError("Selection report must derive from full seed-42 candidate runs")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered C0/C1 mask-collapse contrast."
    )
    parser.add_argument("--cell", choices=("C0", "C1"), required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_mask_collapse_v1.yaml")
    )
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    code_tree_clean = _code_tree_is_clean(repo_root)
    if args.profile == "full" and not code_tree_clean:
        raise RuntimeError("Full anti-collapse training requires committed code and configs")

    collapse = load_config(args.config)
    validate_mask_collapse_config(collapse)
    source_path = _repo_path(repo_root, collapse["study"]["source_mask_ablation_config"])
    ablation = load_config(source_path)
    validate_mask_ablation_config(ablation)
    if args.seed != int(collapse["training"]["first_stage_seed"]):
        raise ValueError("The first anti-collapse contrast is restricted to the frozen seed")
    _validate_selection_report(
        args.selection_report,
        source_config=ablation,
        source_config_path=source_path,
        policy=args.policy,
        seed=args.seed,
    )

    base_path = _repo_path(repo_root, collapse["study"]["base_config"])
    config = copy.deepcopy(load_config(base_path))
    config["study"]["protocol"] = collapse["study"]["protocol"]
    config["training"]["seed"] = args.seed
    config["masking"].update(ablation["policies"][args.policy])
    validate_study_config(config)
    vicreg_weight = float(collapse["training"]["vicreg_global_weight"])
    vicreg_config = dict(collapse["training"]["vicreg"])
    result = train_mask_collapse_cell(
        cell=args.cell,
        config=config,
        real_root=args.real_root,
        output_dir=args.output_dir,
        profile=args.profile,
        device=torch.device(args.device),
        vicreg_weight=vicreg_weight,
        vicreg_config=vicreg_config,
        drop_last_training_batch=bool(collapse["training"]["drop_last_training_batch"]),
    )

    metrics_path = args.output_dir / "metrics.json"
    checkpoint_path = args.output_dir / "checkpoint.pt"
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
        "code_tree_clean": code_tree_clean,
        "protocol": config["study"]["protocol"],
        "mask_policy": args.policy,
        "cell": args.cell,
        "profile": args.profile,
        "seed": args.seed,
        "vicreg_weight": vicreg_weight if args.cell == "C1" else 0.0,
        "vicreg_config": vicreg_config,
        "selection_report_sha256": _sha256(args.selection_report),
        "config_sha256": _sha256(args.config),
        "source_mask_ablation_config_sha256": _sha256(source_path),
        "base_config_sha256": _sha256(base_path),
        "training_contract": result["training_contract"],
        "dataset_contract": result["contract"],
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
