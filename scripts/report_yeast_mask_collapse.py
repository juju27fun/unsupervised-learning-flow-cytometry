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
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p3_ssl.config import load_config, validate_mask_collapse_config
from p3_ssl.mask_collapse_reporting import (
    compare_mask_collapse_runs,
    summarize_mask_collapse_run,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_clean_code(repo_root: Path) -> None:
    paths = (
        "p3_ssl/config.py",
        "p3_ssl/mask_collapse_reporting.py",
        "scripts/report_yeast_mask_collapse.py",
        "configs/yeast_ssl_mask_collapse_v1.yaml",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("Scientific C0/C1 reporting requires committed code and config")


def _load_cell(cell: str, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("cell") != cell:
        raise ValueError(f"Invalid completed {cell} run: {run_dir}")
    if metrics.get("cell") != cell:
        raise ValueError(f"{cell} manifest/metrics mismatch")
    if manifest.get("sealed_splits_used") or metrics.get("sealed_splits_used"):
        raise ValueError(f"{cell} unexpectedly used a sealed split")
    for output_name, expected_sha256 in manifest.get("outputs", {}).items():
        output_path = run_dir / output_name
        if not output_path.is_file() or _sha256(output_path) != expected_sha256:
            raise ValueError(f"{cell} output checksum mismatch: {output_name}")
    return manifest, metrics


def _assert_comparable(c0: dict[str, Any], c1: dict[str, Any]) -> None:
    for key in (
        "protocol",
        "mask_policy",
        "profile",
        "seed",
        "selection_report_sha256",
        "config_sha256",
        "source_mask_ablation_config_sha256",
        "base_config_sha256",
        "dataset",
        "repositories",
        "training_contract",
        "dataset_contract",
    ):
        if c0.get(key) != c1.get(key):
            raise ValueError(f"C0/C1 are not comparable: {key} differs")
    if float(c0.get("vicreg_weight", -1.0)) != 0.0:
        raise ValueError("C0 must have zero VICReg weight")
    if float(c1.get("vicreg_weight", -1.0)) != 1.0:
        raise ValueError("C1 must use the frozen VICReg weight 1.0")
    if c0.get("profile") != "full" or int(c0.get("seed", -1)) != 42:
        raise ValueError("Scientific C0/C1 reporting requires the full frozen seed-42 runs")


def _plot(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    cells = [str(row["cell"]) for row in rows]
    positions = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    width = 0.24
    for offset, key, label, color in (
        (-width, "zero_masked_mse", "zero", "#767676"),
        (0.0, "interpolation_masked_mse", "interpolation", "#e09f3e"),
        (width, "model_masked_mse", "model", "#0077b6"),
    ):
        axes[0].bar(positions + offset, [row[key] for row in rows], width, label=label, color=color)
    axes[0].set_title("Held-out masked reconstruction")
    axes[0].set_ylabel("MSE (lower is better)")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(positions, [row["effective_rank"] for row in rows], color="#2a9d8f")
    axes[1].axhline(
        float(config["promotion_gates"]["geometry"]["effective_rank_min"]),
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    axes[1].set_title("Embedding effective rank")
    axes[1].set_ylabel("rank / 96 (higher is better)")

    axes[2].bar(positions, [row["mean_pairwise_cosine"] for row in rows], color="#d1495b")
    axes[2].axhline(
        float(config["promotion_gates"]["geometry"]["mean_pairwise_cosine_max"]),
        color="black",
        linestyle="--",
        linewidth=0.8,
    )
    axes[2].set_title("Mean pairwise cosine")
    axes[2].set_ylabel("similarity (lower is better)")
    for axis in axes:
        axis.set_xticks(positions, cells)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate the preregistered C0/C1 contrast.")
    parser.add_argument("--c0-run", type=Path, required=True)
    parser.add_argument("--c1-run", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_mask_collapse_v2.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")
    repo_root = Path(__file__).resolve().parents[1]
    _require_clean_code(repo_root)

    config = load_config(args.config)
    validate_mask_collapse_config(config)
    c0_manifest, c0_metrics = _load_cell("C0", args.c0_run)
    c1_manifest, c1_metrics = _load_cell("C1", args.c1_run)
    _assert_comparable(c0_manifest, c1_manifest)
    rows = [
        summarize_mask_collapse_run("C0", c0_metrics, config["promotion_gates"]),
        summarize_mask_collapse_run("C1", c1_metrics, config["promotion_gates"]),
    ]
    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "protocol": c0_manifest["protocol"],
        "mask_policy": c0_manifest["mask_policy"],
        "seed": c0_manifest["seed"],
        "profile": c0_manifest["profile"],
        "sealed_splits_used": [],
        "source_runs": {"C0": str(args.c0_run), "C1": str(args.c1_run)},
        **compare_mask_collapse_runs(rows[0], rows[1], config),
    }

    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = args.output_dir / "comparison.csv"
    flat_rows = [{key: value for key, value in row.items() if key != "gates"} for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    figure_path = args.output_dir / "anti_collapse_gates.png"
    _plot(figure_path, rows, config)

    outputs = [metrics_path, csv_path, figure_path]
    manifest = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": config["study"]["real_dataset"],
        "repositories": {"unsupervised-learning-flow-cytometry": _revision(repo_root)},
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "protocol": c0_manifest["protocol"],
        "mask_policy": c0_manifest["mask_policy"],
        "seed": c0_manifest["seed"],
        "config_sha256": _sha256(args.config),
        "sealed_splits_used": [],
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
