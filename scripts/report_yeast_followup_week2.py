#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from p3_ssl.config import load_config
from p3_ssl.followup_reporting import (
    plot_week2,
    summarize_week2,
    write_decision_markdown,
    write_summary_csv,
)


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen yeast Week 2 promotion gate.")
    parser.add_argument("--evaluation-metrics", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_followup_week2_v1.yaml")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")
    payload = json.loads(args.evaluation_metrics.read_text(encoding="utf-8"))
    config = load_config(args.config)
    if payload.get("sealed_splits_used") != []:
        raise PermissionError("Week 2 report refuses evaluation that accessed a sealed split")
    summary = summarize_week2(payload, config)
    args.output_dir.mkdir(parents=True)
    decision_json = args.output_dir / "week2_decision.json"
    decision_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table_path = args.output_dir / "week2_seed_summary.csv"
    write_summary_csv(table_path, summary["rows"])
    decision_md = args.output_dir / "WEEK2_DECISION.md"
    write_decision_markdown(decision_md, summary)
    figure_paths = plot_week2(summary, args.output_dir / "week2_r0_r3_comparison")
    outputs = [decision_json, table_path, decision_md, *figure_paths]
    repo_root = Path(__file__).resolve().parents[1]
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
        "profile": "week2-full-decision",
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_evaluation": str(args.evaluation_metrics),
        "sealed_splits_used": [],
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"gate": summary["gate"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
