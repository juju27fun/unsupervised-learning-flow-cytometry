from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

from .followup_features import (
    feature_matrix,
    extract_feature_families,
    fit_probe,
    load_followup_development,
    sample_record_groups,
)
from .study_baselines import checkpoint_encoder_features
from .study_evaluation import cross_recording_retrieval, physical_embedding_diagnostics
from .study_training import embedding_health_statistics


def _read_simulation_rows(root: Path, split: str) -> list[dict[str, str]]:
    with (root / "simulation_metadata.csv").open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def _bounded_simulation_rows(
    rows: list[dict[str, str]], maximum_latents: int | None
) -> list[dict[str, str]]:
    if maximum_latents is None:
        return rows
    latent_ids = sorted({row["latent_id"] for row in rows})[:maximum_latents]
    allowed = set(latent_ids)
    return [row for row in rows if row["latent_id"] in allowed]


def _grouped_macro_f1_bootstrap(
    labels: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    *,
    class_count: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    unique = np.unique(groups)
    if repeats <= 0 or len(unique) < 2:
        return {"status": "not_run", "n_groups": int(len(unique))}
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        values.append(
            f1_score(
                labels[indices],
                predictions[indices],
                labels=np.arange(class_count),
                average="macro",
                zero_division=0,
            )
        )
    low, high = np.quantile(values, (0.025, 0.975))
    return {
        "status": "ok",
        "group_unit": "capture_block_id",
        "n_groups": int(len(unique)),
        "n_repeats": repeats,
        "macro_f1_mean": float(np.mean(values)),
        "macro_f1_ci95": [float(low), float(high)],
    }


def _subgroup_macro_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    class_count: int,
) -> dict[str, Any]:
    width_edges = np.quantile([float(row["width_ms"]) for row in train_rows], (1 / 3, 2 / 3))
    snr_edges = np.quantile([float(row["snr_proxy"]) for row in train_rows], (1 / 3, 2 / 3))
    output: dict[str, Any] = {
        "threshold_source": "followup_train_only",
        "width_edges_ms": width_edges.tolist(),
        "snr_edges": snr_edges.tolist(),
        "quality": {},
        "width_tertile": {},
        "snr_tertile": {},
    }
    strata = {
        "quality": np.asarray([row["quality"] for row in rows]),
        "width_tertile": np.asarray(
            ["low middle high".split()[np.searchsorted(width_edges, float(row["width_ms"]), side="right")] for row in rows]
        ),
        "snr_tertile": np.asarray(
            ["low middle high".split()[np.searchsorted(snr_edges, float(row["snr_proxy"]), side="right")] for row in rows]
        ),
    }
    for axis, values in strata.items():
        for value in sorted(set(values.tolist())):
            mask = values == value
            output[axis][value] = {
                "n": int(mask.sum()),
                "macro_f1": float(
                    f1_score(
                        labels[mask],
                        predictions[mask],
                        labels=np.arange(class_count),
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
    return output


def _probe_matrix(
    *,
    learned: np.ndarray,
    handcrafted: np.ndarray,
    data: Any,
    fractions: list[float],
    probe_seeds: list[int],
    probes: tuple[str, ...],
    bootstrap_repeats: int,
) -> list[dict[str, Any]]:
    matrices = {
        "learned": learned,
        "handcrafted": handcrafted,
        "handcrafted_plus_learned": np.concatenate((handcrafted, learned), axis=1),
    }
    validation = data.validation_indices
    validation_rows = [data.rows[int(index)] for index in validation]
    train_rows = [data.rows[int(index)] for index in data.train_indices]
    groups = np.asarray([row["capture_block_id"] for row in validation_rows])
    rows = []
    for probe in probes:
        for fraction in fractions:
            for probe_seed in probe_seeds:
                selected = sample_record_groups(
                    data.rows, data.labels, data.train_indices, fraction, probe_seed
                )
                for method, matrix in matrices.items():
                    metrics, predictions = fit_probe(
                        matrix[selected],
                        data.labels[selected],
                        matrix[validation],
                        data.labels[validation],
                        probe=probe,
                        seed=probe_seed,
                        class_names=data.class_names,
                    )
                    row = {
                        "method": method,
                        "probe": probe,
                        "label_fraction": fraction,
                        "probe_seed": probe_seed,
                        "n_probe_events": int(len(selected)),
                        "n_probe_records": len(
                            {data.rows[int(index)]["record_id"] for index in selected}
                        ),
                        **metrics,
                    }
                    if probe == "linear" and fraction == 0.10:
                        row["grouped_bootstrap"] = _grouped_macro_f1_bootstrap(
                            data.labels[validation],
                            predictions,
                            groups,
                            class_count=len(data.class_names),
                            repeats=bootstrap_repeats,
                            seed=probe_seed,
                        )
                        row["subgroups"] = _subgroup_macro_f1(
                            data.labels[validation],
                            predictions,
                            validation_rows,
                            train_rows,
                            len(data.class_names),
                        )
                    rows.append(row)
    return rows


def _label_efficiency_auc(probe_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in probe_rows:
        grouped[(row["method"], row["probe"], int(row["probe_seed"]))].append(row)
    output = []
    for (method, probe, seed), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: float(row["label_fraction"]))
        fractions = np.asarray([float(row["label_fraction"]) for row in ordered])
        scores = np.asarray([float(row["macro_f1"]) for row in ordered])
        if len(np.unique(fractions)) < 2:
            continue
        area = float(np.trapezoid(scores, fractions))
        output.append(
            {
                "method": method,
                "probe": probe,
                "probe_seed": seed,
                "normalized_area": area / float(fractions[-1] - fractions[0]),
            }
        )
    return output


def evaluate_followup_checkpoints(
    *,
    checkpoints: list[tuple[str, Path]],
    config: dict[str, Any],
    real_root: Path,
    simulation_root: Path,
    profile: str,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    data = load_followup_development(real_root)
    families, feature_names = extract_feature_families(data.signals, data.rows)
    handcrafted = feature_matrix(families)
    if profile == "smoke":
        fractions = [0.10]
        probe_seeds = [42]
        probes = ("linear",)
        bootstrap_repeats = int(config["evaluation"]["smoke_grouped_bootstrap_repeats"])
        maximum_simulation_latents = 64
    else:
        fractions = [float(value) for value in config["evaluation"]["label_fractions"]]
        probe_seeds = [int(value) for value in config["evaluation"]["probe_seeds"]]
        probes = ("linear", "mlp")
        bootstrap_repeats = int(config["evaluation"]["grouped_bootstrap_repeats"])
        maximum_simulation_latents = None
    simulation_train_rows = _bounded_simulation_rows(
        _read_simulation_rows(simulation_root, config["data"]["simulation_train_split"]),
        maximum_simulation_latents,
    )
    simulation_validation_rows = _bounded_simulation_rows(
        _read_simulation_rows(simulation_root, config["data"]["simulation_validation_split"]),
        maximum_simulation_latents,
    )
    simulation_rows = simulation_train_rows + simulation_validation_rows
    simulation_array = np.load(simulation_root / "signals.npy", mmap_mode="r")
    simulation_signals = np.asarray(
        simulation_array[[int(row["signal_row"]) for row in simulation_rows]], dtype=np.float32
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_results = {}
    all_probe_rows = []
    batch_size = int(config["evaluation"]["batch_size"])
    for name, checkpoint_path in checkpoints:
        real_embeddings, metadata = checkpoint_encoder_features(
            data.signals, checkpoint_path, batch_size=batch_size, device=device
        )
        simulation_embeddings, simulation_metadata = checkpoint_encoder_features(
            simulation_signals, checkpoint_path, batch_size=batch_size, device=device
        )
        if metadata != simulation_metadata:
            raise ValueError("Real and simulation checkpoint metadata differ")
        if metadata["protocol"] != config["study"]["protocol"]:
            raise ValueError(f"Checkpoint {name} does not use the Week 2 protocol")
        training_payload = json.loads(
            (checkpoint_path.parent / "metrics.json").read_text(encoding="utf-8")
        )
        if training_payload["cell"] != metadata["cell"] or int(training_payload["seed"]) != int(
            metadata["seed"]
        ):
            raise ValueError(f"Checkpoint {name} and training metrics disagree")
        if training_payload.get("sealed_splits_used") != []:
            raise PermissionError(f"Checkpoint {name} training accessed a sealed split")
        probes_for_checkpoint = _probe_matrix(
            learned=real_embeddings,
            handcrafted=handcrafted,
            data=data,
            fractions=fractions,
            probe_seeds=probe_seeds,
            probes=probes,
            bootstrap_repeats=bootstrap_repeats,
        )
        for row in probes_for_checkpoint:
            row.update(
                {
                    "checkpoint": name,
                    "cell": metadata["cell"],
                    "representation_seed": int(metadata["seed"]),
                }
            )
        all_probe_rows.extend(probes_for_checkpoint)
        n_simulation_train = len(simulation_train_rows)
        validation_embeddings = real_embeddings[data.validation_indices]
        validation_rows = [data.rows[int(index)] for index in data.validation_indices]
        physical = physical_embedding_diagnostics(
            simulation_embeddings[:n_simulation_train],
            simulation_embeddings[n_simulation_train:],
            simulation_train_rows,
            simulation_validation_rows,
            neighbors=int(config["evaluation"]["retrieval_neighbors"]),
            seed=int(metadata["seed"]),
        )
        retained = physical["retained_factor_linear_probes"]
        checkpoint_results[name] = {
            **metadata,
            "checkpoint": str(checkpoint_path),
            "training_convergence": training_payload["convergence"],
            "training_runtime": training_payload["runtime"],
            "training_history": training_payload["history"],
            "real_validation_embedding_health": embedding_health_statistics(validation_embeddings),
            "simulation_validation_embedding_health": embedding_health_statistics(
                simulation_embeddings[n_simulation_train:]
            ),
            "physical_retention": physical,
            "mean_continuous_relative_mse_reduction": float(
                np.mean(
                    [value["relative_mse_reduction_vs_constant"] for value in retained.values()]
                )
            ),
            "component_count_balanced_accuracy": physical["component_count_probe"][
                "balanced_accuracy"
            ],
            "cross_recording_retrieval": cross_recording_retrieval(
                validation_embeddings,
                validation_rows,
                data.labels[data.validation_indices],
                neighbors=int(config["evaluation"]["retrieval_neighbors"]),
            ),
        }
        np.save(output_dir / f"real_embeddings_{name}.npy", real_embeddings)
        np.save(output_dir / f"simulation_embeddings_{name}.npy", simulation_embeddings)
    return {
        "protocol": config["study"]["protocol"],
        "source_frozen_protocol": config["study"]["source_frozen_protocol"],
        "profile": profile,
        "class_names": data.class_names,
        "known_limitation": "followup_validation has no shmoo capture block",
        "feature_names": feature_names,
        "checkpoint_results": checkpoint_results,
        "probe_results": all_probe_rows,
        "label_efficiency_auc": _label_efficiency_auc(all_probe_rows),
        "n_real_train": int(len(data.train_indices)),
        "n_real_validation": int(len(data.validation_indices)),
        "n_simulation_train_views": len(simulation_train_rows),
        "n_simulation_validation_views": len(simulation_validation_rows),
        "sealed_splits_used": [],
    }
