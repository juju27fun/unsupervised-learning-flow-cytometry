#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p3_ssl.config import load_config, validate_mask_ablation_config
from p3_ssl.mask_ablation_reporting import summarize_mask_ablation_run


def _run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be POLICY=PATH")
    policy, raw_path = value.split("=", 1)
    return policy, Path(raw_path)


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_clean_code(repo_root: Path) -> None:
    paths = (
        "p3_ssl/config.py",
        "p3_ssl/mask_ablation_reporting.py",
        "scripts/report_yeast_mask_ablation.py",
        "configs/yeast_ssl_mask_ablation_v1.yaml",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("Scientific mask reporting requires committed code and config")


def _plot(path: Path, rows: list[dict[str, object]], config: dict[str, object]) -> None:
    policies = [str(row["policy"]) for row in rows]
    positions = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(
        positions,
        [float(row["relative_improvement_vs_zero"]) for row in rows],
        color="#0077b6",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Reconstruction vs zero")
    axes[0].set_ylabel("relative MSE improvement")

    axes[1].bar(
        positions,
        [float(row["output_rms_fraction_of_target"]) for row in rows],
        color="#2a9d8f",
    )
    axes[1].axhline(
        float(config["promotion_gates"]["pretext"]["output_rms_fraction_of_target_min"]),
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    axes[1].set_title("Prediction amplitude")
    axes[1].set_ylabel("output RMS / target RMS")

    axes[2].bar(
        positions,
        [float(row["effective_rank"]) for row in rows],
        color="#d1495b",
    )
    axes[2].axhline(
        float(config["promotion_gates"]["geometry"]["effective_rank_min"]),
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    axes[2].set_title("Embedding geometry")
    axes[2].set_ylabel("effective rank / 96")
    for axis in axes:
        axis.set_xticks(positions, policies)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate yeast patch-mask ablation runs.")
    parser.add_argument("--run", action="append", type=_run, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_mask_ablation_v1.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")
    repo_root = Path(__file__).resolve().parents[1]
    _require_clean_code(repo_root)

    config = load_config(args.config)
    validate_mask_ablation_config(config)
    expected = set(config["training"]["candidate_policies"])
    supplied = {policy for policy, _ in args.run}
    if supplied != expected or len(args.run) != len(expected):
        raise ValueError(f"Expected candidate runs {sorted(expected)}, received {sorted(supplied)}")
    rows = []
    source_runs = {}
    source_contract = None
    for policy, run_dir in args.run:
        run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if (
            run_manifest.get("status") != "complete"
            or run_manifest.get("mask_policy") != policy
            or run_manifest.get("protocol") != config["study"]["protocol"]
            or run_manifest.get("profile") != "full"
            or int(run_manifest.get("seed", -1)) != int(config["training"]["local_smoke_seed"])
            or run_manifest.get("sealed_splits_used")
        ):
            raise ValueError(f"Invalid completed run for {policy}: {run_dir}")
        for output_name, expected_sha256 in run_manifest.get("outputs", {}).items():
            output_path = run_dir / output_name
            if not output_path.is_file() or _sha256(output_path) != expected_sha256:
                raise ValueError(f"Source output checksum mismatch for {policy}: {output_name}")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if (
            metrics.get("cell") != "A1"
            or metrics.get("profile") != "full"
            or int(metrics.get("seed", -1)) != int(config["training"]["local_smoke_seed"])
            or metrics.get("sealed_splits_used")
        ):
            raise ValueError(f"Source metrics mismatch for {policy}: {run_dir}")
        for key, value in config["policies"][policy].items():
            if metrics.get("masking", {}).get(key) != value:
                raise ValueError(f"Source mask policy mismatch for {policy}.{key}")
        candidate_contract = {
            key: run_manifest.get(key)
            for key in (
                "dataset",
                "repositories",
                "config_sha256",
                "base_config_sha256",
                "protocol",
                "profile",
                "seed",
            )
        }
        if source_contract is None:
            source_contract = candidate_contract
        elif candidate_contract != source_contract:
            raise ValueError("Mask candidates do not share one training/provenance contract")
        rows.append(summarize_mask_ablation_run(policy, metrics, config["promotion_gates"]))
        source_runs[policy] = str(run_dir)
    rows.sort(key=lambda row: list(config["training"]["candidate_policies"]).index(row["policy"]))
    eligible = [str(row["policy"]) for row in rows if row["eligible_for_utility_evaluation"]]
    pretext_candidates = [
        row
        for row in rows
        if row["gates"]["beats_zero"] and row["gates"]["nontrivial_amplitude"]
    ]
    anti_collapse_policy = (
        str(max(pretext_candidates, key=lambda row: row["relative_improvement_vs_zero"])["policy"])
        if pretext_candidates and not eligible
        else None
    )
    if eligible:
        decision = "run_development_utility_evaluation"
    elif anti_collapse_policy is not None:
        decision = "run_preregistered_anti_collapse_contrast"
    else:
        decision = "run_preregistered_phase_invariant_target_contrast"

    args.output_dir.mkdir(parents=True)
    csv_path = args.output_dir / "comparison.csv"
    csv_rows = [{key: value for key, value in row.items() if key != "gates"} for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    figure_path = args.output_dir / "mask_ablation_gates.png"
    _plot(figure_path, rows, config)
    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "protocol": config["study"]["protocol"],
        "sealed_splits_used": [],
        "source_runs": source_runs,
        "source_training_contract": source_contract,
        "decision_config_sha256": _sha256(args.config),
        "rows": rows,
        "eligible_for_utility_evaluation": eligible,
        "pretext_pass_policies": [str(row["policy"]) for row in pretext_candidates],
        "anti_collapse_policy": anti_collapse_policy,
        "decision": decision,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outputs = [csv_path, figure_path, metrics_path]
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": config["study"]["real_dataset"],
        "repositories": {"unsupervised-learning-flow-cytometry": _revision(repo_root)},
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "config_sha256": _sha256(args.config),
        "sealed_splits_used": [],
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
