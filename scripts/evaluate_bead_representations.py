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
import yaml

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.provenance import collect_git_state
from p3_ssl.bead_representation_benchmark import (
    FeatureSet,
    average_simulation_views,
    classical_descriptor_matrix,
    embedding_health,
    evaluate_real_features,
    evaluate_simulation_features,
    extract_embeddings,
    label_efficiency_auc,
    load_encoder,
    load_real_population,
    load_simulation_population,
    nested_real_subsets,
    nested_simulation_subsets,
    nominal_ssl_budget,
    paired_hierarchical_interval,
    paired_grouped_classification_interval,
    verify_nested,
    write_csv,
)
from p3_ssl.bead_ssl import load_bead_ssl_config


def _record(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    row = next(
        (item for item in records if f"{item['id']}@{item['version']}" == key),
        None,
    )
    if row is None or row["status"] not in {"active", "reference"}:
        raise ValueError(f"Registered dataset not found: {key}")
    return row


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_sets(
    *,
    config: dict[str, Any],
    population,
    checkpoint_root: Path,
    device: torch.device,
    seeds: tuple[int, ...],
) -> tuple[list[FeatureSet], list[dict[str, Any]]]:
    outputs = [
        FeatureSet("Raw PCA-64", None, population.signals, kind="raw_pca"),
        FeatureSet(
            "Physical descriptors",
            None,
            classical_descriptor_matrix(population.signals),
        ),
    ]
    health = []
    for seed in seeds:
        for method, pattern in (
            ("Random frozen", None),
            ("P25", f"bead-ssl-p25-b0-full-s{seed}-v1-cmp2/checkpoints/latest.pt"),
            (
                "CYCLIC25",
                f"bead-ssl-cyclic25-b0-full-s{seed}-v1-cmp2/checkpoints/latest.pt",
            ),
        ):
            checkpoint = None if pattern is None else checkpoint_root / pattern
            if checkpoint is not None and not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model = load_encoder(
                config=config,
                seed=seed,
                checkpoint=checkpoint,
                device=device,
            )
            values = extract_embeddings(model, population.signals, device=device)
            outputs.append(FeatureSet(method, seed, values))
            health.append(
                {
                    "method": method,
                    "representation_seed": seed,
                    **embedding_health(values),
                }
            )
    return outputs, health


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen P25/CYCLIC25 bead representations on development only."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bead_ssl_p25_v1.yaml"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subset-seeds", type=int, default=10)
    parser.add_argument("--encoder-seeds", default="42,43,44,45,46")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.encoder_seeds.split(","))
    if seeds != (42, 43, 44, 45, 46):
        raise ValueError("Final protocol requires encoder seeds 42-46")

    workspace = Workspace.load()
    config = load_bead_ssl_config(args.config)
    records = [record.payload for record in load_records(workspace)]
    simulation_record = _record(records, "yeast-passage-simulations@v1")
    real_key = (
        "particles2snr-f-dual-clean-c1-descriptor-events-"
        "saturation-reviewed-development@v1"
    )
    real_record = _record(records, real_key)
    output = (
        workspace.artifacts_root
        / "unsupervised-learning-flow-cytometry"
        / "runs"
        / args.run_id
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = torch.device(args.device)

    simulation_train = load_simulation_population(
        workspace.datasets_root / simulation_record["path"], split="train"
    )
    simulation_validation = load_simulation_population(
        workspace.datasets_root / simulation_record["path"], split="validation"
    )
    real_train = load_real_population(
        workspace.datasets_root / real_record["path"], split="train"
    )
    real_validation = load_real_population(
        workspace.datasets_root / real_record["path"], split="val"
    )
    checkpoint_root = (
        workspace.artifacts_root / "unsupervised-learning-flow-cytometry/runs"
    )

    sim_train_sets, sim_health = _feature_sets(
        config=config,
        population=simulation_train,
        checkpoint_root=checkpoint_root,
        device=device,
        seeds=seeds,
    )
    sim_validation_sets, _ = _feature_sets(
        config=config,
        population=simulation_validation,
        checkpoint_root=checkpoint_root,
        device=device,
        seeds=seeds,
    )
    real_train_sets, real_health = _feature_sets(
        config=config,
        population=real_train,
        checkpoint_root=checkpoint_root,
        device=device,
        seeds=seeds,
    )
    real_validation_sets, _ = _feature_sets(
        config=config,
        population=real_validation,
        checkpoint_root=checkpoint_root,
        device=device,
        seeds=seeds,
    )

    simulation_metrics = []
    simulation_predictions = []
    real_metrics = []
    real_predictions = []
    validation_sim_by_key = {
        (item.method, item.representation_seed): item for item in sim_validation_sets
    }
    validation_real_by_key = {
        (item.method, item.representation_seed): item for item in real_validation_sets
    }
    for train_set in sim_train_sets:
        averaged_train, train_targets, _ = average_simulation_views(
            simulation_train, train_set.values
        )
        validation_set = validation_sim_by_key[
            (train_set.method, train_set.representation_seed)
        ]
        averaged_validation, validation_targets, _ = average_simulation_views(
            simulation_validation, validation_set.values
        )
        train_set = FeatureSet(
            train_set.method,
            train_set.representation_seed,
            averaged_train,
            train_set.kind,
        )
        repeat_count = args.subset_seeds
        if train_set.representation_seed is None:
            repeat_count = args.subset_seeds
        for subset_seed in range(repeat_count):
            subsets = nested_simulation_subsets(
                train_targets, seed=20260720 + subset_seed
            )
            if subset_seed:
                subsets.pop(1.0)
            verify_nested(subsets)
            metrics, predictions = evaluate_simulation_features(
                train_set,
                averaged_validation,
                train_targets,
                validation_targets,
                subsets,
                subset_seed=subset_seed,
            )
            simulation_metrics.extend(metrics)
            simulation_predictions.extend(predictions)

    for train_set in real_train_sets:
        validation_set = validation_real_by_key[
            (train_set.method, train_set.representation_seed)
        ]
        for subset_seed in range(args.subset_seeds):
            subsets = nested_real_subsets(
                real_train.labels,
                real_train.groups,
                seed=20260720 + subset_seed,
            )
            if subset_seed:
                subsets.pop(1.0)
            verify_nested(subsets)
            metrics, predictions = evaluate_real_features(
                train_set,
                validation_set.values,
                real_train,
                real_validation,
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
    write_csv(output / "embedding_health.csv", sim_health + real_health)
    comparisons = {
        "simulation_10pct_mean_r2": paired_hierarchical_interval(
            [
                row
                for row in simulation_metrics
                if np.isclose(float(row["fraction"]), 0.10)
            ],
            metric="mean_r2",
        ),
        "simulation_auc": paired_hierarchical_interval(
            [row for row in sim_auc if row["representation_seed"] is not None],
            metric="normalized_log_fraction_auc",
        ),
        "real_full_macro_f1": paired_hierarchical_interval(
            [
                row
                for row in real_metrics
                if np.isclose(float(row["fraction"]), 1.0)
            ],
            metric="macro_f1",
        ),
        "real_full_macro_f1_grouped_by_source": (
            paired_grouped_classification_interval(real_predictions)
        ),
    }
    for method in ("P25", "CYCLIC25"):
        comparisons[f"simulation_10pct_{method}_vs_random"] = (
            paired_hierarchical_interval(
                [
                    row
                    for row in simulation_metrics
                    if np.isclose(float(row["fraction"]), 0.10)
                ],
                metric="mean_r2",
                left="Random frozen",
                right=method,
            )
        )
        comparisons[f"simulation_auc_{method}_vs_random"] = (
            paired_hierarchical_interval(
                [
                    row
                    for row in sim_auc
                    if row["representation_seed"] is not None
                ],
                metric="normalized_log_fraction_auc",
                left="Random frozen",
                right=method,
            )
        )
        comparisons[f"real_full_{method}_vs_random"] = (
            paired_hierarchical_interval(
                [
                    row
                    for row in real_metrics
                    if np.isclose(float(row["fraction"]), 1.0)
                ],
                metric="macro_f1",
                left="Random frozen",
                right=method,
            )
        )
        comparisons[f"real_full_{method}_vs_random_grouped_by_source"] = (
            paired_grouped_classification_interval(
                real_predictions,
                left="Random frozen",
                right=method,
            )
        )
    (output / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    repository_states = {
        "workspace": collect_git_state(workspace.root),
        "unsupervised-learning-flow-cytometry": collect_git_state(
            workspace.root / "unsupervised-learning-flow-cytometry"
        ),
    }
    checkpoint_hashes = {
        method: {
            str(seed): _sha256_file(
                checkpoint_root
                / (
                    f"bead-ssl-{policy}-b0-full-s{seed}-v1-cmp2/"
                    "checkpoints/latest.pt"
                )
            )
            for seed in seeds
        }
        for method, policy in (("P25", "p25"), ("CYCLIC25", "cyclic25"))
    }
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "kind": "bead-frozen-representation-benchmark",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": (
            "yeast-passage-simulations@v1 + "
            f"{real_key}"
        ),
        "repositories": {
            name: state["revision"] for name, state in repository_states.items()
        },
        "repository_dirty": {
            name: state["dirty"] for name, state in repository_states.items()
        },
        "command": shlex.join(sys.argv),
        "protocol": "bead-frozen-representation-benchmark-development-v1",
        "encoder_seeds": list(seeds),
        "subset_seeds": args.subset_seeds,
        "datasets": {
            "simulation": {
                "id": "yeast-passage-simulations@v1",
                "manifest_sha256": simulation_record["manifest_sha256"],
            },
            "real": {
                "id": real_key,
                "manifest_sha256": real_record["manifest_sha256"],
            },
        },
        "checkpoints": checkpoint_hashes,
        "source_sha256": {
            "config": _sha256_file(args.config),
            "benchmark_module": _sha256_file(
                Path(__file__).resolve().parents[1]
                / "p3_ssl"
                / "bead_representation_benchmark.py"
            ),
            "bead_ssl_module": _sha256_file(
                Path(__file__).resolve().parents[1] / "p3_ssl" / "bead_ssl.py"
            ),
            "model_module": _sha256_file(
                Path(__file__).resolve().parents[1] / "p3_ssl" / "models.py"
            ),
            "decimation_module": _sha256_file(
                Path(__file__).resolve().parents[1]
                / "p3_ssl"
                / "decimation.py"
            ),
            "particle_class_coverage_module": _sha256_file(
                workspace.root
                / "particles2SNR-pipeline"
                / "particles2snr"
                / "particle_class_coverage.py"
            ),
            "ssl_realism_audit_module": _sha256_file(
                workspace.root
                / "particles2SNR-pipeline"
                / "particles2snr"
                / "ssl_realism_audit.py"
            ),
            "entrypoint": _sha256_file(Path(__file__).resolve()),
        },
        "nominal_ssl_budget": nominal_ssl_budget(),
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
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"run": run, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
