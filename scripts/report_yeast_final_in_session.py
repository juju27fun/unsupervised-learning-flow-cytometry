#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from p3_ssl.final_reporting import build_final_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the frozen yeast in-session test report.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    summary = build_final_report(args.metrics, args.output_dir)
    repo_root = Path(__file__).resolve().parents[1]

    def revision(path: Path) -> str:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
        ).stdout.strip()

    outputs = [*summary["outputs"], "summary.json"]
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": "yeast-events-representation@v3",
        "repositories": {
            "unsupervised-learning-flow-cytometry": revision(repo_root),
            "particles2SNR-pipeline": revision(repo_root.parent / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "source_sha256": hashlib.sha256(args.metrics.read_bytes()).hexdigest(),
        "outputs": outputs,
        "output_sha256": {
            name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
            for name in outputs
        },
        "sealed_splits_used": ["in_session_test"],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
