#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from p3_ssl.local_spectral_reporting import (
    audit_registered_dataset,
    sha256,
    summarize_local_spectral_result,
)


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_clean_code(repo_root: Path) -> None:
    paths = (
        "p3_ssl/local_spectral_reporting.py",
        "scripts/report_yeast_local_spectral.py",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("Scientific S1 reporting requires committed reporting code")


def _load_run(path: Path, cell: str) -> tuple[dict[str, Any], dict[str, Any]]:
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete" or run.get("cell") != cell:
        raise ValueError(f"Invalid completed {cell} run: {path}")
    if run.get("sealed_splits_used") or metrics.get("sealed_splits_used"):
        raise ValueError(f"{cell} unexpectedly used a sealed split")
    for output_name, expected in run.get("outputs", {}).items():
        if sha256(path / output_name) != expected:
            raise ValueError(f"{cell} output checksum mismatch: {output_name}")
    return run, metrics


def _plot(path: Path, s1_metrics: dict[str, Any], summary: dict[str, Any]) -> None:
    history = s1_metrics["history"]
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(epochs, [row["loss"] for row in history], label="total", color="#3b4cc0")
    axes[0, 0].plot(
        epochs,
        [row["local_spectral_prediction"] for row in history],
        label="local spectral",
        color="#b40426",
    )
    axes[0, 0].plot(epochs, [row["vicreg"] for row in history], label="VICReg", color="#2a9d8f")
    axes[0, 0].set_title("Training objectives")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend(frameon=False)

    controls = summary["controls"]
    control_names = ["zero", "train constant", "S1", "interpolation"]
    control_values = [
        controls["zero_mse"],
        controls["train_constant_mse"],
        controls["s1_mse"],
        controls["interpolation_mse"],
    ]
    axes[0, 1].bar(
        control_names,
        control_values,
        color=["#777777", "#e09f3e", "#0077b6", "#2a9d8f"],
    )
    axes[0, 1].set_title("Held-out masked target")
    axes[0, 1].set_ylabel("MSE (lower is better)")
    axes[0, 1].tick_params(axis="x", rotation=20)

    region_names = ["event", "boundary", "background"]
    region_ratios = [summary["regions"][name]["s1_to_interpolation_ratio"] for name in region_names]
    axes[1, 0].bar(region_names, region_ratios, color="#d1495b")
    axes[1, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("S1 relative to interpolation")
    axes[1, 0].set_ylabel("MSE ratio (>1 is worse)")

    geometry = summary["geometry"]
    positions = np.arange(2)
    cells = ["C1", "S1"]
    rank_axis = axes[1, 1]
    cosine_axis = rank_axis.twinx()
    width = 0.34
    rank_axis.bar(
        positions - width / 2,
        [geometry[cell]["effective_rank"] for cell in cells],
        width,
        color="#457b9d",
        label="effective rank",
    )
    cosine_axis.bar(
        positions + width / 2,
        [geometry[cell]["mean_pairwise_cosine"] for cell in cells],
        width,
        color="#e76f51",
        label="mean cosine",
    )
    rank_axis.axhline(8.0, color="#457b9d", linestyle="--", linewidth=1)
    cosine_axis.axhline(0.95, color="#e76f51", linestyle="--", linewidth=1)
    rank_axis.set_xticks(positions, cells)
    rank_axis.set_ylabel("effective rank", color="#457b9d")
    cosine_axis.set_ylabel("mean cosine", color="#e76f51")
    rank_axis.set_title("Embedding geometry")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("S1 local-spectral objective: optimization works, interpolation gate fails")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _report_text(summary: dict[str, Any], dataset: dict[str, Any]) -> str:
    controls = summary["controls"]
    geometry = summary["geometry"]
    return f"""# S1 local-spectral objective decision

**Decision:** `{summary['decision']}`

![S1 decision figure](s1_local_spectral_decision.png)

The optimization itself worked: local-spectral prediction loss fell by
`{summary['training']['relative_reduction']:.1%}` over
`{summary['training']['epochs']}` epochs. On development validation, S1 MSE was
`{controls['s1_mse']:.4f}`, compared with `{controls['zero_mse']:.4f}` for zero,
`{controls['train_constant_mse']:.4f}` for the train constant, and
`{controls['interpolation_mse']:.4f}` for features computed after linear
waveform interpolation. S1 is therefore `{controls['s1_to_interpolation_ratio']:.2f}x`
worse than the strongest deterministic control.

This failure is consistent across event, boundary, and background regions.
The representation is not collapsed: effective rank improves from
`{geometry['C1']['effective_rank']:.2f}` in C1 to
`{geometry['S1']['effective_rank']:.2f}` in S1, while mean pairwise cosine falls
from `{geometry['C1']['mean_pairwise_cosine']:.3f}` to
`{geometry['S1']['mean_pairwise_cosine']:.3f}`.

Under the frozen seed-42 PE25 development protocol, the objective/head package
learns a nontrivial target and healthy embedding geometry but does not recover
hidden local spectra better than interpolation. Development utility and more
representation seeds are not authorized. This is an exploratory stop decision,
not evidence that local-spectral SSL is universally ineffective.

## Provenance audit

- Registered manifest SHA256: `{dataset['manifest_sha256']}`
- Verified files: `{dataset['manifest_entries_verified']}`
- Events: `{dataset['event_count']}`
- Split counts: `{json.dumps(dataset['split_counts'], sort_keys=True)}`
- Record and capture-block overlap across splits: zero
- Sealed split used by S1: no
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the terminal S1 decision report.")
    parser.add_argument("--s1-run", type=Path, required=True)
    parser.add_argument("--c1-run", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    _require_clean_code(repo_root)
    s1_run, s1_metrics = _load_run(args.s1_run, "S1")
    _, c1_metrics = _load_run(args.c1_run, "C1")
    if s1_run.get("profile") != "full" or int(s1_run.get("seed", -1)) != 42:
        raise ValueError("S1 decision requires the frozen full seed-42 run")
    dataset_audit = audit_registered_dataset(args.dataset_root, args.dataset_manifest)
    if dataset_audit["manifest_sha256"] != s1_run["dataset_manifest_sha256"]:
        raise ValueError("S1 run and independently hashed dataset manifest differ")
    summary = summarize_local_spectral_result(s1_metrics, c1_metrics)
    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "source_s1_run": s1_run["run_id"],
        "sealed_splits_used": [],
        "dataset_audit": dataset_audit,
        **summary,
    }

    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure_path = args.output_dir / "s1_local_spectral_decision.png"
    _plot(figure_path, s1_metrics, result)
    report_path = args.output_dir / "report.md"
    report_path.write_text(_report_text(result, dataset_audit), encoding="utf-8")
    outputs = [metrics_path, figure_path, report_path]
    manifest = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": s1_run["dataset"],
        "dataset_manifest_sha256": dataset_audit["manifest_sha256"],
        "repositories": {"unsupervised-learning-flow-cytometry": _revision(repo_root)},
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "protocol": s1_run["protocol"],
        "decision": result["decision"],
        "sealed_splits_used": [],
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
