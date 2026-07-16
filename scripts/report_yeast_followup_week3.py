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

from p3_ssl.config import load_config
from p3_ssl.followup_week3 import plot_week3, write_week3_markdown


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the frozen Week 3 simulator decision.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evaluation-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")
    config = load_config(args.config)
    payload = json.loads(args.evaluation_metrics.read_text(encoding="utf-8"))
    if payload.get("sealed_splits_used") != [] or payload["protocol"] != config["study"]["protocol"]:
        raise PermissionError("Week 3 report refuses non-protocol or sealed-split evidence")
    args.output_dir.mkdir(parents=True)
    decision = args.output_dir / "week3_decision.json"
    decision.write_text(
        json.dumps(payload["decision"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = args.output_dir / "WEEK3_DECISION.md"
    write_week3_markdown(markdown, payload)
    figure_paths = plot_week3(payload, args.output_dir / "week3_simulator_comparison")
    table = args.output_dir / "week3_support_summary.csv"
    rows = []
    for source in ("baseline_v1", "corrected_v2"):
        report = payload["source_results"][source]["matching_by_caliper"]["1.50"]
        rows.append(
            {
                "simulation_source": source,
                "train_retained_fraction": report["train"]["real_retained_fraction"],
                "validation_retained_fraction": report["validation"]["real_retained_fraction"],
                "train_max_smd": report["train"]["post_match_smd_max"],
                "validation_max_smd": report["validation"]["post_match_smd_max"],
                "support_pass": report["support_pass"],
            }
        )
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs = [decision, markdown, table, *figure_paths]
    repo_root = Path(__file__).resolve().parents[1]
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": " + ".join(payload["datasets"].values()),
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(repo_root.parent / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "profile": "week3-publication-report",
        "config_sha256": _sha256(args.config),
        "source_evaluation": str(args.evaluation_metrics),
        "sealed_splits_used": [],
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
