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

import numpy as np
import torch

from p3_ssl.config import load_config
from p3_ssl.study_baselines import (
    checkpoint_encoder_features,
    load_baseline_data,
    prediction_metrics,
    simulation_real_domain_probe,
)
from p3_ssl.study_evaluation import (
    cross_recording_retrieval,
    evaluate_linear_probe,
    label_efficiency_auc,
    perturb_signals,
    physical_embedding_diagnostics,
    real_variability_summary,
    robustness_metrics,
)
from p3_ssl.study_training import embedding_health_statistics


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name, Path(raw_path)


def _simulation_indices(root: Path, split: str) -> np.ndarray:
    with (root / "simulation_metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray(
        [int(row["signal_row"]) for row in rows if row["split"] == split], dtype=np.int64
    )


def _simulation_rows(root: Path) -> dict[int, dict[str, str]]:
    with (root / "simulation_metadata.csv").open(newline="", encoding="utf-8") as handle:
        return {int(row["signal_row"]): row for row in csv.DictReader(handle)}


def _bounded(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    return np.sort(np.random.default_rng(seed).choice(indices, size=maximum, replace=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen A1-A4 yeast study checkpoints.")
    parser.add_argument("--checkpoint", action="append", type=_checkpoint, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/yeast_ssl_rebuild_v1.yaml"))
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")

    config = load_config(args.config)
    profile = config["baselines"]["profiles"][args.profile]
    evaluation = config["evaluation"]
    bootstrap_repeats = int(
        evaluation["profiles"][args.profile]["grouped_bootstrap_repeats"]
    )
    seed = int(config["training"]["seed"])
    data = load_baseline_data(
        args.real_root, max_per_class=profile["max_events_per_class"], seed=seed
    )
    real_indices = np.concatenate([data.train_indices, data.validation_indices])
    real_signals = np.asarray(data.signals[real_indices], dtype=np.float32)
    simulation_signals = np.load(args.simulation_root / "signals.npy", mmap_mode="r")
    simulation_train = _bounded(
        _simulation_indices(args.simulation_root, "train"), len(data.train_indices), seed
    )
    simulation_validation = _bounded(
        _simulation_indices(args.simulation_root, "validation"), len(data.validation_indices), seed + 1
    )
    simulation_indices = np.concatenate([simulation_train, simulation_validation])
    simulation_subset = np.asarray(simulation_signals[simulation_indices], dtype=np.float32)
    simulation_by_index = _simulation_rows(args.simulation_root)
    simulation_train_rows = [simulation_by_index[int(index)] for index in simulation_train]
    simulation_validation_rows = [
        simulation_by_index[int(index)] for index in simulation_validation
    ]
    device = torch.device(args.device)
    results = []
    checkpoint_metadata = {}
    embedding_payloads = {}

    for name, checkpoint_path in args.checkpoint:
        real_embeddings, metadata = checkpoint_encoder_features(
            real_signals,
            checkpoint_path,
            batch_size=int(profile["batch_size"]),
            device=device,
        )
        features = np.empty((len(data.rows), real_embeddings.shape[1]), dtype=np.float32)
        features[real_indices] = real_embeddings
        probe_models = {}
        for fraction in profile["label_fractions"]:
            for probe_seed in profile["probe_seeds"]:
                metrics, probe_model = evaluate_linear_probe(
                    features,
                    data,
                    fraction=float(fraction),
                    seed=int(probe_seed),
                    bootstrap_repeats=bootstrap_repeats,
                    calibration_bins=int(evaluation["calibration_bins"]),
                )
                probe_models[(float(fraction), int(probe_seed))] = probe_model
                results.append(
                    {
                        "checkpoint": name,
                        "cell": metadata["cell"],
                        "label_fraction": fraction,
                        "seed": probe_seed,
                        **metrics,
                    }
                )
        simulation_embeddings, _ = checkpoint_encoder_features(
            simulation_subset,
            checkpoint_path,
            batch_size=int(profile["batch_size"]),
            device=device,
        )
        n_real_train = len(data.train_indices)
        n_simulation_train = len(simulation_train)
        domain = simulation_real_domain_probe(
            real_embeddings[:n_real_train],
            simulation_embeddings[:n_simulation_train],
            real_embeddings[n_real_train:],
            simulation_embeddings[n_simulation_train:],
            seed=seed,
        )
        validation_embeddings = real_embeddings[n_real_train:]
        validation_rows = [data.rows[int(index)] for index in data.validation_indices]
        retrieval = cross_recording_retrieval(
            validation_embeddings,
            validation_rows,
            data.labels[data.validation_indices],
            neighbors=int(evaluation["retrieval_neighbors"]),
        )
        robustness_fraction = float(evaluation["robustness_label_fraction"])
        robustness_seed = int(profile["probe_seeds"][0])
        probe_model = probe_models[(robustness_fraction, robustness_seed)]
        base_probability = probe_model.predict_proba(validation_embeddings)
        validation_signals = real_signals[n_real_train:]
        robustness = {}
        for perturbation_index, (perturbation_name, perturbation) in enumerate(
            evaluation["perturbations"].items()
        ):
            perturbed_signals = perturb_signals(
                validation_signals,
                perturbation,
                seed=seed + perturbation_index,
            )
            perturbed_embeddings, _ = checkpoint_encoder_features(
                perturbed_signals,
                checkpoint_path,
                batch_size=int(profile["batch_size"]),
                device=device,
            )
            perturbed_probability = probe_model.predict_proba(perturbed_embeddings)
            robustness[perturbation_name] = {
                **robustness_metrics(
                    validation_embeddings,
                    perturbed_embeddings,
                    base_probability,
                    perturbed_probability,
                ),
                "role": perturbation["role"],
                "perturbed_prediction_metrics": prediction_metrics(
                    data.labels[data.validation_indices],
                    perturbed_probability.argmax(axis=1),
                    data.class_names,
                ),
            }
        checkpoint_metadata[name] = {
            **metadata,
            "checkpoint": str(checkpoint_path),
            "real_embedding_health": embedding_health_statistics(real_embeddings),
            "simulation_embedding_health": embedding_health_statistics(simulation_embeddings),
            "simulation_real_domain_probe": domain,
            "development_retrieval": retrieval,
            "development_physical_fidelity": physical_embedding_diagnostics(
                simulation_embeddings[:n_simulation_train],
                simulation_embeddings[n_simulation_train:],
                simulation_train_rows,
                simulation_validation_rows,
                neighbors=int(evaluation["retrieval_neighbors"]),
                seed=seed,
            ),
            "development_robustness": {
                "probe_label_fraction": robustness_fraction,
                "probe_seed": robustness_seed,
                "perturbations": robustness,
                "interpretation": (
                    "gain sensitivity is diagnostic because absolute amplitude is unresolved, not a trained invariance"
                ),
            },
        }
        embedding_payloads[name] = (real_embeddings, simulation_embeddings)

    args.output_dir.mkdir(parents=True)
    np.save(args.output_dir / "real_embedding_row_indices.npy", real_indices)
    np.save(args.output_dir / "simulation_embedding_row_indices.npy", simulation_indices)
    for name, (real_embeddings, simulation_embeddings) in embedding_payloads.items():
        np.save(args.output_dir / f"real_embeddings_{name}.npy", real_embeddings)
        np.save(args.output_dir / f"simulation_embeddings_{name}.npy", simulation_embeddings)
    payload = {
        "protocol": config["study"]["protocol"],
        "real_dataset": config["study"]["real_dataset"],
        "simulation_dataset": config["study"]["simulation_dataset"],
        "endpoint_scope": "source-group proxy and simulation-real diagnostics; no morphology claim",
        "profile": args.profile,
        "checkpoint_metadata": checkpoint_metadata,
        "results": results,
        "label_efficiency_auc": label_efficiency_auc(results, "checkpoint"),
        "development_variability": real_variability_summary(data),
        "sealed_splits_used": [],
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
