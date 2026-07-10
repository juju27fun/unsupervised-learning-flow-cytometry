#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from p3_ssl.run_assessment import assess_hybrid_run, write_run_assessment


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess a P3 hybrid-physics run against the physical-validation success criteria.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    assessment = assess_hybrid_run(args.run_dir)
    write_run_assessment(assessment, args.output_dir or args.run_dir)
    print(json.dumps({"run_dir": str(args.run_dir), "assessment_pass": assessment["assessment_pass"]}, indent=2))


if __name__ == "__main__":
    main()
