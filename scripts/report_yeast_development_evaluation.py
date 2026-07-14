#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from p3_ssl.study_reporting import build_development_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build yeast SSL development-smoke tables and figures.")
    parser.add_argument("--a0-metrics", type=Path, required=True)
    parser.add_argument("--checkpoint-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    summary = build_development_report(args.a0_metrics, args.checkpoint_metrics, args.output_dir)
    repo_root = Path(__file__).resolve().parents[1]

    def revision(path: Path) -> str:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
        ).stdout.strip()

    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": "yeast-events-representation@v2 + yeast-passage-simulations@v1",
        "repositories": {
            "unsupervised-learning-flow-cytometry": revision(repo_root),
            "particles2SNR-pipeline": revision(repo_root.parent / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source_sha256": {
            "a0_metrics": hashlib.sha256(args.a0_metrics.read_bytes()).hexdigest(),
            "checkpoint_metrics": hashlib.sha256(args.checkpoint_metrics.read_bytes()).hexdigest(),
        },
        "outputs": [*summary["outputs"], "summary.json"],
        "sealed_splits_used": [],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
