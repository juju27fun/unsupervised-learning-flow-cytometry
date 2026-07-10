#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p3_ssl.config import load_config
from p3_ssl.preflight import preflight_hybrid_run
from p3_ssl.serialization import json_safe


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight-check the P3 hybrid physical pipeline inputs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--real-manifest", required=True, type=Path)
    parser.add_argument("--simulation-source", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--profile", choices=["smoke", "full", "long"], default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--strict-exit-code",
        action="store_true",
        help="Exit with status 1 when required preflight checks fail.",
    )
    args = parser.parse_args()

    report = preflight_hybrid_run(
        config=load_config(args.config),
        real_manifest=args.real_manifest,
        profile=args.profile,
        device=args.device,
        simulation_source=args.simulation_source,
        output_root=args.output_root,
    )
    payload = json.dumps(json_safe(report), indent=2, allow_nan=False)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n")
    print(payload)
    if args.strict_exit_code and not report["run_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
