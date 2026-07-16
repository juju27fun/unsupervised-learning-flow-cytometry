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
from p3_ssl.followup_week3 import evaluate_week3


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen Week 3 simulator correction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--corrected-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")
    config = load_config(args.config)
    calibration = json.loads(
        (args.corrected_root / "support_calibration.json").read_text(encoding="utf-8")
    )
    if calibration.get("source_split") != "followup_train" or calibration.get(
        "sealed_splits_used"
    ) != []:
        raise PermissionError("Corrected simulation calibration violates the train-only contract")
    payload = evaluate_week3(
        real_root=args.real_root,
        baseline_root=args.baseline_root,
        corrected_root=args.corrected_root,
        config=config,
    )
    args.output_dir.mkdir(parents=True)
    metrics = args.output_dir / "metrics.json"
    metrics.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    probes = args.output_dir / "domain_probe_metrics.csv"
    with probes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload["probe_results"][0]))
        writer.writeheader()
        writer.writerows(payload["probe_results"])
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
        "profile": "week3-full-simulator-comparison",
        "config_sha256": _sha256(args.config),
        "sealed_splits_used": [],
        "outputs": {path.name: _sha256(path) for path in (metrics, probes)},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
