#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from p3_ssl.config import load_config
from p3_ssl.study_baselines import (
    BASELINE_METHODS,
    handcrafted_features,
    load_baseline_data,
    public_encoder_features,
    random_encoder_features,
    rms_features,
    supervised_conv1d,
)
from p3_ssl.study_evaluation import (
    cross_recording_retrieval,
    evaluate_linear_probe,
    label_efficiency_auc,
    real_variability_summary,
)
from p3_ssl.study_training import embedding_health_statistics, model_config_from_study


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(methods) - BASELINE_METHODS
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Run same-input A0 yeast proxy-label baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/yeast_ssl_rebuild_v1.yaml"))
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--methods", default=",".join(sorted(BASELINE_METHODS)))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cache-dir", type=Path, default=Path("../.cache/huggingface"))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")
    config = load_config(args.config)
    profile = config["baselines"]["profiles"][args.profile]
    evaluation = config["evaluation"]
    bootstrap_repeats = int(
        evaluation["profiles"][args.profile]["grouped_bootstrap_repeats"]
    )
    data = load_baseline_data(
        args.real_root,
        max_per_class=profile["max_events_per_class"],
        seed=int(config["training"]["seed"]),
    )
    methods = _parse_methods(args.methods)
    device = torch.device(args.device)
    selected = np.concatenate([data.train_indices, data.validation_indices])
    subset = np.asarray(data.signals[selected], dtype=np.float32)
    results = []
    method_metadata = {}
    persisted_embeddings = {}
    for method in methods:
        if method == "conv1d":
            for fraction in profile["label_fractions"]:
                for seed in profile["probe_seeds"]:
                    metrics = supervised_conv1d(
                        data,
                        fraction=float(fraction),
                        seed=int(seed),
                        epochs=int(profile["conv1d_epochs"]),
                        batch_size=int(profile["batch_size"]),
                        device=device,
                        bootstrap_repeats=bootstrap_repeats,
                        calibration_bins=int(evaluation["calibration_bins"]),
                        retrieval_neighbors=int(evaluation["retrieval_neighbors"]),
                    )
                    results.append({"method": method, "label_fraction": fraction, "seed": seed, **metrics})
            continue
        if method == "rms":
            subset_features = rms_features(subset)
            metadata = {"feature": "window RMS only", "shortcut_control": True}
        elif method == "raw":
            subset_features = subset
            metadata = {"feature": "all 4096 globally normalized samples"}
        elif method == "handcrafted":
            subset_features = handcrafted_features(subset)
            metadata = {"feature": "time-frequency-envelope", "n_features": subset_features.shape[1]}
        elif method == "random":
            subset_features = random_encoder_features(
                subset,
                config=model_config_from_study(config),
                batch_size=int(profile["batch_size"]),
                device=device,
                seed=int(config["training"]["seed"]),
            )
            metadata = {"feature": "untrained study encoder", "n_features": subset_features.shape[1]}
        else:
            subset_features, metadata = public_encoder_features(
                method,
                subset,
                batch_size=int(profile["public_batch_size"]),
                device=device,
                cache_dir=args.cache_dir,
            )
        if method in {"random", "moment", "patchtst"}:
            metadata["embedding_health"] = embedding_health_statistics(subset_features)
            persisted_embeddings[method] = subset_features
        features = np.empty((len(data.rows), subset_features.shape[1]), dtype=np.float32)
        features[selected] = subset_features
        method_metadata[method] = metadata
        validation_offset = len(data.train_indices)
        metadata["development_retrieval"] = cross_recording_retrieval(
            subset_features[validation_offset:],
            [data.rows[int(index)] for index in data.validation_indices],
            data.labels[data.validation_indices],
            neighbors=int(evaluation["retrieval_neighbors"]),
        )
        for fraction in profile["label_fractions"]:
            for seed in profile["probe_seeds"]:
                metrics, _ = evaluate_linear_probe(
                    features,
                    data,
                    fraction=float(fraction),
                    seed=int(seed),
                    bootstrap_repeats=bootstrap_repeats,
                    calibration_bins=int(evaluation["calibration_bins"]),
                )
                results.append({"method": method, "label_fraction": fraction, "seed": seed, **metrics})

    args.output_dir.mkdir(parents=True)
    payload = {
        "protocol": config["study"]["protocol"],
        "dataset": config["study"]["real_dataset"],
        "endpoint_scope": "source-group proxy diagnostic; not morphology or acquisition-OOD",
        "profile": args.profile,
        "methods": methods,
        "class_names": data.class_names,
        "n_train_events": int(data.train_indices.size),
        "n_validation_events": int(data.validation_indices.size),
        "sealed_splits_used": [],
        "method_metadata": method_metadata,
        "results": results,
        "label_efficiency_auc": label_efficiency_auc(results, "method"),
        "development_variability": real_variability_summary(data),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if persisted_embeddings:
        np.save(args.output_dir / "embedding_row_indices.npy", selected)
        for method, embeddings in persisted_embeddings.items():
            np.save(args.output_dir / f"embeddings_{method}.npy", embeddings)
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
        "profile": args.profile,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "sealed_splits_used": [],
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
