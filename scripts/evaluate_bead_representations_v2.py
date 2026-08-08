#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.provenance import collect_git_state
from p3_ssl.bead_representation_benchmark import (
    FeatureSet,
    classical_descriptor_matrix,
    embedding_health,
    evaluate_real_features,
    evaluate_simulation_features,
    extract_embeddings,
    label_efficiency_auc,
    load_encoder,
    nested_real_subsets,
    nested_simulation_subsets,
    paired_grouped_classification_interval,
    paired_hierarchical_interval,
    verify_nested,
    write_csv,
)
from p3_ssl.bead_ssl_v2 import load_bead_ssl_v2_config
from p3_ssl.bead_ssl_v2_populations import load_v5_population, load_z8_v2_population


POLICIES = ("P25", "CYCLIC25")
SEEDS = (42, 43, 44, 45, 46)


def _record(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    row = next((item for item in records if f"{item['id']}@{item['version']}" == key), None)
    if row is None or row["status"] not in {"active", "reference"}:
        raise ValueError(f"registered dataset not found: {key}")
    return row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_map(runs_root: Path, run_ids: list[str]) -> tuple[dict[tuple[str, int], Path], int]:
    output: dict[tuple[str, int], Path] = {}
    epochs: set[int] = set()
    for run_id in run_ids:
        root = runs_root / run_id
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if run.get("status") != "complete" or run.get("profile") != "full":
            raise ValueError(f"ineligible source run: {run_id}")
        key = (str(run["training_mask_policy"]), int(run["seed"]))
        if key in output:
            raise ValueError(f"duplicate source: {key}")
        checkpoint = root / "checkpoints/latest.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output[key] = checkpoint
        epochs.add(int(run["epochs"]))
    if set(output) != {(policy, seed) for policy in POLICIES for seed in SEEDS}:
        raise ValueError("frozen benchmark requires the complete paired seeds 42-46 matrix")
    if len(epochs) != 1:
        raise ValueError("all source checkpoints must share one epoch budget")
    return output, epochs.pop()


def feature_sets(
    *,
    config: dict[str, Any],
    population,
    checkpoints: dict[tuple[str, int], Path],
    expected_epoch: int,
    device: torch.device,
) -> tuple[list[FeatureSet], list[dict[str, Any]]]:
    outputs = [
        FeatureSet("Raw PCA-64", None, population.signals, kind="raw_pca"),
        FeatureSet("Physical descriptors", None, classical_descriptor_matrix(population.signals)),
    ]
    health: list[dict[str, Any]] = []
    for seed in SEEDS:
        for method in ("Random frozen", *POLICIES):
            checkpoint = None if method == "Random frozen" else checkpoints[(method, seed)]
            model = load_encoder(
                config=config,
                seed=seed,
                checkpoint=checkpoint,
                device=device,
                expected_epoch=expected_epoch,
            )
            values = extract_embeddings(model, population.signals, device=device)
            outputs.append(FeatureSet(method, seed, values))
            health_values = values[: min(4096, len(values))]
            health.append(
                {
                    "method": method,
                    "representation_seed": seed,
                    "health_sample_count": len(health_values),
                    **embedding_health(health_values),
                }
            )
    return outputs, health


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen representation benchmark on Z8 synthetic v5 and real Z8 v2.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id", action="append", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/bead_ssl_z8_v5_v2.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subset-seeds", type=int, default=10)
    args = parser.parse_args()

    workspace = Workspace.load()
    config = load_bead_ssl_v2_config(args.config)
    records = [record.payload for record in load_records(workspace)]
    study = config["study"]
    simulation_record = _record(records, study["simulation_dataset"])
    event_record = _record(records, study["real_event_dataset"])
    signal_record = _record(records, study["real_signal_dataset"])
    roots = {
        "simulation": workspace.datasets_root / simulation_record["path"],
        "events": workspace.datasets_root / event_record["path"],
        "signals": workspace.datasets_root / signal_record["path"],
    }
    runs_root = workspace.artifacts_root / "unsupervised-learning-flow-cytometry/runs"
    checkpoints, expected_epoch = checkpoint_map(runs_root, args.source_run_id)
    output = runs_root / args.run_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = torch.device(args.device)

    populations = {
        "simulation_train": load_v5_population(roots["simulation"], split="train", normalization=config["data"]["normalization"]),
        "simulation_validation": load_v5_population(roots["simulation"], split="val", normalization=config["data"]["normalization"]),
        "real_train": load_z8_v2_population(roots["events"], roots["signals"], split="train", normalization=config["data"]["normalization"]),
        "real_validation": load_z8_v2_population(roots["events"], roots["signals"], split="val", normalization=config["data"]["normalization"]),
    }
    features: dict[str, list[FeatureSet]] = {}
    health: list[dict[str, Any]] = []
    for name, population in populations.items():
        features[name], rows = feature_sets(
            config=config,
            population=population,
            checkpoints=checkpoints,
            expected_epoch=expected_epoch,
            device=device,
        )
        health.extend({"population": name, **row} for row in rows)

    simulation_metrics: list[dict[str, Any]] = []
    simulation_predictions: list[dict[str, Any]] = []
    real_metrics: list[dict[str, Any]] = []
    real_predictions: list[dict[str, Any]] = []
    sim_validation = {(row.method, row.representation_seed): row for row in features["simulation_validation"]}
    real_validation = {(row.method, row.representation_seed): row for row in features["real_validation"]}
    for train_set in features["simulation_train"]:
        validation_set = sim_validation[(train_set.method, train_set.representation_seed)]
        for subset_seed in range(args.subset_seeds):
            subsets = nested_simulation_subsets(
                populations["simulation_train"].labels, seed=20260801 + subset_seed
            )
            if subset_seed:
                subsets.pop(1.0)
            verify_nested(subsets)
            metrics, predictions = evaluate_simulation_features(
                train_set,
                validation_set.values,
                populations["simulation_train"].labels,
                populations["simulation_validation"].labels,
                subsets,
                subset_seed=subset_seed,
            )
            simulation_metrics.extend(metrics)
            simulation_predictions.extend(predictions)
    for train_set in features["real_train"]:
        validation_set = real_validation[(train_set.method, train_set.representation_seed)]
        for subset_seed in range(args.subset_seeds):
            subsets = nested_real_subsets(
                populations["real_train"].labels,
                populations["real_train"].groups,
                seed=20260801 + subset_seed,
            )
            if subset_seed:
                subsets.pop(1.0)
            verify_nested(subsets)
            metrics, predictions = evaluate_real_features(
                train_set,
                validation_set.values,
                populations["real_train"],
                populations["real_validation"],
                subsets,
                subset_seed=subset_seed,
            )
            real_metrics.extend(metrics)
            real_predictions.extend(predictions)

    sim_auc = label_efficiency_auc(simulation_metrics, score_key="mean_r2")
    real_auc = label_efficiency_auc(real_metrics, score_key="macro_f1")
    write_csv(output / "simulation_metrics.csv", simulation_metrics)
    write_csv(output / "simulation_predictions.csv", simulation_predictions)
    write_csv(output / "real_metrics.csv", real_metrics)
    write_csv(output / "real_predictions.csv", real_predictions)
    write_csv(output / "label_efficiency_auc.csv", sim_auc + real_auc)
    write_csv(output / "embedding_health.csv", health)
    comparisons = {
        "simulation_10pct_mean_r2": paired_hierarchical_interval(
            [row for row in simulation_metrics if np.isclose(float(row["fraction"]), 0.10)],
            metric="mean_r2",
        ),
        "simulation_auc": paired_hierarchical_interval(
            [row for row in sim_auc if row["representation_seed"] is not None],
            metric="normalized_log_fraction_auc",
        ),
        "real_full_macro_f1": paired_hierarchical_interval(
            [row for row in real_metrics if np.isclose(float(row["fraction"]), 1.0)],
            metric="macro_f1",
        ),
        "real_full_macro_f1_grouped_by_source": paired_grouped_classification_interval(real_predictions),
    }
    (output / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    states = {
        "workspace": collect_git_state(workspace.root),
        "unsupervised-learning-flow-cytometry": collect_git_state(workspace.root / "unsupervised-learning-flow-cytometry"),
    }
    dataset_records = {"simulation": simulation_record, "real_events": event_record, "real_signals": signal_record}
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "kind": "bead-frozen-representation-benchmark-v2",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": ",".join(f"{row['id']}@{row['version']}" for row in dataset_records.values()),
        "datasets": {
            name: {"id": f"{row['id']}@{row['version']}", "manifest_sha256": row["manifest_sha256"]}
            for name, row in dataset_records.items()
        },
        "source_runs": args.source_run_id,
        "repositories": {name: state["revision"] for name, state in states.items()},
        "repository_dirty": {name: state["dirty"] for name, state in states.items()},
        "command": shlex.join(sys.argv),
        "protocol": "bead-frozen-representation-benchmark-development-v2",
        "encoder_seeds": list(SEEDS),
        "subset_seeds": args.subset_seeds,
        "checkpoint_epoch": expected_epoch,
        "simulation_target_semantics": {
            "duration_ms": "tau_ms (Gaussian sigma in the physical waveform model)",
            "doppler_khz": "frequency_khz",
        },
        "checkpoints": {
            policy: {str(seed): _sha256(checkpoints[(policy, seed)]) for seed in SEEDS}
            for policy in POLICIES
        },
        "source_sha256": {
            "config": _sha256(args.config),
            "entrypoint": _sha256(Path(__file__).resolve()),
            "population_module": _sha256(Path(__file__).resolve().parents[1] / "p3_ssl/bead_ssl_v2_populations.py"),
        },
        "nominal_ssl_budget": {
            "epochs": expected_epoch,
            "simulation_train": len(populations["simulation_train"].signals),
            "batch_size": int(config["training"]["profiles"]["full"]["batch_size"]),
            "optimizer_steps": expected_epoch
            * int(
                np.ceil(
                    len(populations["simulation_train"].signals)
                    / int(config["training"]["profiles"]["full"]["batch_size"])
                )
            ),
        },
        "sealed_splits_used": [],
        "outputs": [
            "simulation_metrics.csv",
            "simulation_predictions.csv",
            "real_metrics.csv",
            "real_predictions.csv",
            "label_efficiency_auc.csv",
            "embedding_health.csv",
            "comparisons.json",
        ],
    }
    (output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run": run, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
