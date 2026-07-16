#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from p3_ssl.followup_domain import fit_domain_probe, matched_pairs, signal_observables
from p3_ssl.followup_features import load_followup_development
from p3_ssl.study_baselines import checkpoint_encoder_features, public_encoder_features


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, text=True, capture_output=True
    ).stdout.strip()


def _select(length: int, maximum: int, seed: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(length, maximum, replace=False))


def _synthetic(root: Path, split: str, maximum: int, seed: int) -> np.ndarray:
    rows = [row for row in _read_csv(root / "simulation_metadata.csv") if row["split"] == split]
    rows = [row for row in rows if int(row.get("view_index", 0)) == 0]
    selected = _select(len(rows), maximum, seed)
    signals = np.load(root / "signals.npy", mmap_mode="r")
    return np.asarray([signals[int(rows[int(index)]["signal_row"])] for index in selected], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding sensitivity for the Week 1 domain bridge.")
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--analytic-root", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--a3-checkpoint", type=Path, required=True)
    parser.add_argument("--a4-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train", type=int, default=400)
    parser.add_argument("--max-validation", type=int, default=200)
    parser.add_argument("--moment-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-batch-size", type=int, default=32)
    parser.add_argument("--caliper", type=float, default=1.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)

    real = load_followup_development(args.real_root)
    real_split = np.asarray([row["development_split"] for row in real.rows])
    real_arrays = {}
    for source_split, target_split, maximum, seed in (
        ("followup_train", "train", args.max_train, 61),
        ("followup_validation", "validation", args.max_validation, 62),
    ):
        available = np.flatnonzero(real_split == source_split)
        chosen = available[_select(len(available), maximum, seed)]
        real_arrays[target_split] = real.signals[chosen]
    source_arrays = {
        "analytic": {
            "train": _synthetic(args.analytic_root, "train", args.max_train, 71),
            "validation": _synthetic(args.analytic_root, "validation", args.max_validation, 72),
        },
        "template_diagnostic": {
            "train": _synthetic(args.template_root, "train", args.max_train, 73),
            "validation": _synthetic(args.template_root, "validation", args.max_validation, 74),
        },
    }
    ordered = [
        real_arrays["train"], real_arrays["validation"],
        source_arrays["analytic"]["train"], source_arrays["analytic"]["validation"],
        source_arrays["template_diagnostic"]["train"],
        source_arrays["template_diagnostic"]["validation"],
    ]
    lengths = [len(values) for values in ordered]
    all_signals = np.concatenate(ordered)
    representations = {}
    moment, moment_metadata = public_encoder_features(
        "moment",
        all_signals,
        batch_size=args.moment_batch_size,
        device=device,
        cache_dir=args.cache_dir,
    )
    representations["MOMENT_official"] = moment
    for name, checkpoint in (("A3_s42", args.a3_checkpoint), ("A4_s42", args.a4_checkpoint)):
        values, _ = checkpoint_encoder_features(
            all_signals,
            checkpoint,
            batch_size=args.checkpoint_batch_size,
            device=device,
        )
        representations[name] = values

    boundaries = np.cumsum([0, *lengths])
    metrics: list[dict[str, Any]] = []
    matching = {}
    for source_index, source_name in enumerate(("analytic", "template_diagnostic")):
        real_train = ordered[0]
        real_validation = ordered[1]
        synthetic_train = ordered[2 + 2 * source_index]
        synthetic_validation = ordered[3 + 2 * source_index]
        real_train_obs = signal_observables(real_train)
        real_validation_obs = signal_observables(real_validation)
        synthetic_train_obs = signal_observables(synthetic_train)
        synthetic_validation_obs = signal_observables(synthetic_validation)
        scaler = StandardScaler().fit(np.concatenate([real_train_obs, synthetic_train_obs]))
        train_r, train_s, train_report = matched_pairs(
            real_train_obs, synthetic_train_obs, scaler=scaler, caliper=args.caliper
        )
        validation_r, validation_s, validation_report = matched_pairs(
            real_validation_obs, synthetic_validation_obs, scaler=scaler, caliper=args.caliper
        )
        matching[source_name] = {"train": train_report, "validation": validation_report}
        real_train_slice = slice(boundaries[0], boundaries[1])
        real_validation_slice = slice(boundaries[1], boundaries[2])
        synthetic_train_slice = slice(boundaries[2 + 2 * source_index], boundaries[3 + 2 * source_index])
        synthetic_validation_slice = slice(
            boundaries[3 + 2 * source_index], boundaries[4 + 2 * source_index]
        )
        for representation, all_features in representations.items():
            feature_names = [f"dim_{index}" for index in range(all_features.shape[1])]
            base = (
                all_features[real_train_slice], all_features[synthetic_train_slice],
                all_features[real_validation_slice], all_features[synthetic_validation_slice],
            )
            for state in ("unmatched", "matched"):
                train_real, train_synthetic, validation_real, validation_synthetic = base
                if state == "matched":
                    train_real, train_synthetic = train_real[train_r], train_synthetic[train_s]
                    validation_real = validation_real[validation_r]
                    validation_synthetic = validation_synthetic[validation_s]
                for model in ("linear", "forest"):
                    result = fit_domain_probe(
                        train_real,
                        train_synthetic,
                        validation_real,
                        validation_synthetic,
                        feature_names=feature_names,
                        model=model,
                        seed=42,
                        compute_importance=False,
                    )
                    metrics.append(
                        {
                            "simulation_source": source_name,
                            "match_state": state,
                            "representation": representation,
                            "model": model,
                            "validation_roc_auc": result.roc_auc,
                            "converged": result.converged,
                            "n_train_per_domain": min(len(train_real), len(train_synthetic)),
                            "n_validation_per_domain": min(
                                len(validation_real), len(validation_synthetic)
                            ),
                        }
                    )
    with (args.output_dir / "embedding_domain_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "sealed_splits_used": [],
        "device": str(device),
        "moment": moment_metadata,
        "matching": matching,
        "interpretation": (
            "Analytic matched-subset AUCs remain exploratory because common support fails; "
            "template diagnostic is the aligned non-physical sensitivity control."
        ),
    }
    (args.output_dir / "embedding_domain_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run = {
        "schema_version": 1,
        "run_id": args.run_id,
        "project": "unsupervised-learning-flow-cytometry",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "audit_yeast_followup_embedding_domain.py",
        "dataset": "yeast-events-followup@v2 + yeast-passage-simulations@v1 + yeast-template-comparator@v2",
        "profile": "week1-bounded-cuda-embedding-sensitivity",
        "sealed_splits_used": [],
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(Path(__file__).resolve().parents[1]),
            "particles2SNR-pipeline": _revision(Path(__file__).resolve().parents[2] / "particles2SNR-pipeline"),
        },
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "run.json"
        },
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
