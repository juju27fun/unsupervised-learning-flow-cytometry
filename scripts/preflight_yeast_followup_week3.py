#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from p3_ssl.config import load_config
from p3_ssl.followup_domain import matched_pairs, signal_observables
from particles2snr.yeast_simulation import build_simulation_dataset, fit_support_calibration


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_real_train(root: Path, maximum: int, seed: int) -> np.ndarray:
    rows = _read_csv(root / "development_events.csv")
    forbidden = {row["development_split"] for row in rows} - {
        "followup_train",
        "followup_validation",
    }
    if forbidden:
        raise PermissionError(f"Development metadata contains forbidden splits: {sorted(forbidden)}")
    train = [row for row in rows if row["development_split"] == "followup_train"]
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(train), size=min(maximum, len(train)), replace=False))
    signals = np.load(root / "signals.npy", mmap_mode="r")
    return np.asarray([signals[int(train[int(index)]["signal_row"])] for index in selected])


def _load_generated_train(root: Path) -> np.ndarray:
    rows = [
        row
        for row in _read_csv(root / "simulation_metadata.csv")
        if row["split"] == "train" and int(row["view_index"]) == 0
    ]
    signals = np.load(root / "signals.npy", mmap_mode="r")
    return np.asarray([signals[int(row["signal_row"])] for row in rows], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only Week 3 envelope preflight.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")
    config = load_config(args.config)
    correction = config["correction"]
    calibration_config = correction["calibration"]
    maximum = int(calibration_config["pilot_examples_per_domain"])
    pilot_seed = int(calibration_config["pilot_seed"])
    calibration = fit_support_calibration(
        args.real_root,
        quantile_knots=int(calibration_config["quantile_knots"]),
        lower_quantile=float(calibration_config["robust_quantile_interval"][0]),
        upper_quantile=float(calibration_config["robust_quantile_interval"][1]),
    )
    real_signals = _load_real_train(args.real_root, maximum, seed=42)
    real_observables = signal_observables(real_signals)
    candidates = []
    with tempfile.TemporaryDirectory(prefix="yeast-week3-preflight-") as temporary:
        temporary_root = Path(temporary)
        for alpha in map(float, calibration_config["candidate_tukey_alpha"]):
            candidate_root = temporary_root / f"alpha-{alpha:.2f}"
            build_simulation_dataset(
                output_dir=candidate_root,
                n_train_latents=maximum,
                n_validation_latents=1,
                n_test_latents=1,
                views_per_latent=1,
                seed=pilot_seed,
                support_calibration=calibration,
                envelope_model="finite_support_tukey",
                tukey_alpha=alpha,
            )
            synthetic_observables = signal_observables(_load_generated_train(candidate_root))
            scaler = StandardScaler().fit(np.concatenate([real_observables, synthetic_observables]))
            _, _, report = matched_pairs(
                real_observables,
                synthetic_observables,
                scaler=scaler,
                caliper=float(config["evaluation"]["primary_caliper"]),
            )
            candidates.append({"tukey_alpha": alpha, **report})
    selected = min(
        candidates,
        key=lambda row: (row["post_match_smd_max"], -row["real_retained_fraction"]),
    )
    frozen_alpha = float(calibration_config["selected_tukey_alpha"])
    if selected["tukey_alpha"] != frozen_alpha:
        raise RuntimeError(
            f"Train-only candidate selection {selected['tukey_alpha']} differs from frozen {frozen_alpha}"
        )
    payload = {
        "schema_version": 1,
        "protocol": config["study"]["protocol"],
        "status": "pass",
        "selection_rule": calibration_config["selection_metric"],
        "selected_tukey_alpha": selected["tukey_alpha"],
        "candidates": candidates,
        "calibration": calibration,
        "real_splits_used": ["followup_train"],
        "validation_signals_accessed": 0,
        "sealed_splits_used": [],
    }
    args.output_dir.mkdir(parents=True)
    metrics = args.output_dir / "preflight_metrics.json"
    metrics.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": config["study"]["real_dataset"],
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(repo_root.parent / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "profile": "week3-train-only-preflight",
        "config_sha256": _sha256(args.config),
        "sealed_splits_used": [],
        "outputs": {metrics.name: _sha256(metrics)},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
