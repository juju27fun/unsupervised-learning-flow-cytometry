from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from p3_ssl.bead_finetuning import (
    REAL_CLASS_NAMES,
    FineTuningConfig,
    FineTuningDataset,
    aggregate_regression_by_group,
    group_safe_calibration_split,
    group_safe_subset,
    initialize_paired_models,
    sha256_file,
    validate_external_group_separation,
    validate_fraction,
    validate_split_access,
)
from p3_ssl.bead_ssl import configure_experiment, make_model


def tiny_config() -> dict:
    return {
        "study": {
            "protocol": "bead-ssl-comparison-v1",
            "simulation_dataset": "yeast-passage-simulations@v1",
        },
        "data": {"input_length": 32},
        "model": {
            "patch_size": 4,
            "patch_stride": 4,
            "d_model": 8,
            "n_heads": 2,
            "n_layers": 1,
            "dim_feedforward": 16,
            "dropout": 0.0,
            "activation": "gelu",
            "max_tokens": 8,
            "embedding_pool": "mean",
        },
        "masking": {"training_policy": "P25"},
        "loss": {"selected_cell": "B0", "cells": {"B0": {}}},
        "training": {"seed": 42},
    }


def checkpoint(
    path: Path,
    config: dict,
    policy: str,
    seed: int,
    *,
    epoch: int = 20,
) -> None:
    configured = configure_experiment(
        config, loss_cell="B0", mask_policy=policy, seed=seed
    )
    torch.manual_seed(seed)
    torch.save(
        {
            "model_state_dict": make_model(configured).state_dict(),
            "config": configured,
            "epoch": epoch,
        },
        path,
    )


def test_supported_fractions_are_task_specific() -> None:
    assert validate_fraction("simulation", 0.10) == 0.10
    assert validate_fraction("simulation", 1.0) == 1.0
    assert validate_fraction("real", 0.25) == 0.25
    with pytest.raises(ValueError, match="Unsupported simulation fraction"):
        validate_fraction("simulation", 0.25)
    with pytest.raises(ValueError, match="Unsupported real fraction"):
        validate_fraction("real", 0.10)


def test_group_safe_real_subset_and_calibration_have_all_classes() -> None:
    groups = np.repeat([f"g{index:02d}" for index in range(24)], 3)
    targets = np.tile(np.arange(3), 24)
    indices = np.arange(targets.size)
    subset = group_safe_subset(
        indices,
        targets=targets,
        groups=groups,
        fraction=0.25,
        task="real",
        seed=42,
    )
    fit, calibration = group_safe_calibration_split(
        subset,
        targets=targets,
        groups=groups,
        calibration_fraction=0.25,
        task="real",
        seed=42,
    )

    assert set(groups[fit]).isdisjoint(set(groups[calibration]))
    assert set(np.unique(targets[fit])) == {0, 1, 2}
    assert set(np.unique(targets[calibration])) == {0, 1, 2}
    assert set(groups[subset]) == set(groups[fit]) | set(groups[calibration])


def test_group_safe_simulation_subset_is_deterministic() -> None:
    groups = np.repeat([f"latent-{index:02d}" for index in range(40)], 2)
    targets = np.column_stack(
        (
            np.repeat(np.linspace(0.2, 1.2, 40), 2),
            np.repeat(np.linspace(5.0, 50.0, 40), 2),
        )
    )
    indices = np.arange(targets.shape[0])
    first = group_safe_subset(
        indices,
        targets=targets,
        groups=groups,
        fraction=0.10,
        task="simulation",
        seed=7,
    )
    second = group_safe_subset(
        indices,
        targets=targets,
        groups=groups,
        fraction=0.10,
        task="simulation",
        seed=7,
    )

    assert np.array_equal(first, second)
    selected_groups = set(groups[first])
    assert all(np.count_nonzero(groups[first] == group) == 2 for group in selected_groups)


def test_simulation_metrics_average_predictions_per_latent() -> None:
    targets = np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]])
    predictions = np.asarray(
        [[0.0, 1.0], [2.0, 3.0], [2.0, 3.0], [4.0, 5.0]]
    )
    grouped_targets, grouped_predictions = aggregate_regression_by_group(
        targets,
        predictions,
        np.asarray(["a", "a", "b", "b"]),
    )
    assert grouped_targets.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert grouped_predictions.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_paired_initialization_uses_same_head_and_frozen_checkpoints(
    tmp_path: Path,
) -> None:
    config = tiny_config()
    p25 = tmp_path / "p25.pt"
    cyclic = tmp_path / "cyclic.pt"
    checkpoint(p25, config, "P25", 42)
    checkpoint(cyclic, config, "CYCLIC25", 42)

    models, metadata = initialize_paired_models(
        config,
        seed=42,
        task="real",
        p25_checkpoint=p25,
        cyclic25_checkpoint=cyclic,
        device=torch.device("cpu"),
    )

    reference = models["from_scratch"].head.state_dict()
    for method in ("P25", "CYCLIC25"):
        for name, value in models[method].head.state_dict().items():
            assert torch.equal(value, reference[name])
    assert metadata["P25"]["epoch"] == 20
    assert metadata["CYCLIC25"]["epoch"] == 20
    assert models["P25"].head.out_features == len(REAL_CLASS_NAMES)


def test_paired_initialization_accepts_explicit_frozen_epoch(
    tmp_path: Path,
) -> None:
    config = tiny_config()
    p25 = tmp_path / "p25.pt"
    cyclic = tmp_path / "cyclic.pt"
    checkpoint(p25, config, "P25", 42, epoch=35)
    checkpoint(cyclic, config, "CYCLIC25", 42, epoch=35)

    _models, metadata = initialize_paired_models(
        config,
        seed=42,
        task="real",
        p25_checkpoint=p25,
        cyclic25_checkpoint=cyclic,
        device=torch.device("cpu"),
        expected_checkpoint_epoch=35,
    )

    assert metadata["P25"]["epoch"] == 35
    assert metadata["CYCLIC25"]["epoch"] == 35


def test_checkpoint_policy_mismatch_is_rejected(tmp_path: Path) -> None:
    config = tiny_config()
    p25 = tmp_path / "p25.pt"
    wrong_cyclic = tmp_path / "wrong-cyclic.pt"
    checkpoint(p25, config, "P25", 42)
    checkpoint(wrong_cyclic, config, "P25", 42)

    with pytest.raises(ValueError, match="masking config mismatch"):
        initialize_paired_models(
            config,
            seed=42,
            task="simulation",
            p25_checkpoint=p25,
            cyclic25_checkpoint=wrong_cyclic,
            device=torch.device("cpu"),
        )


def test_test_split_requires_hash_bound_confirmatory_manifest(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    p25 = tmp_path / "p25.pt"
    cyclic = tmp_path / "cyclic.pt"
    config.write_text("frozen: true\n", encoding="utf-8")
    p25.write_bytes(b"p25")
    cyclic.write_bytes(b"cyclic")
    checkpoints = {"P25": p25, "CYCLIC25": cyclic}
    sources = {
        "finetuning_entrypoint": tmp_path / "entrypoint.py",
        "finetuning_module": tmp_path / "module.py",
    }
    for name, path in sources.items():
        path.write_text(name, encoding="utf-8")
    settings = FineTuningConfig()

    with pytest.raises(PermissionError, match="without --confirmatory"):
        validate_split_access(
            task="simulation",
            fraction=1.0,
            seed=42,
            fit_splits=("train", "validation"),
            evaluation_split="test",
            settings=settings,
            confirmatory_manifest=None,
            config_path=config,
            checkpoint_paths=checkpoints,
            dataset_manifest_sha256="dataset-hash",
            source_paths=sources,
        )
    with pytest.raises(PermissionError, match="never be a fit split"):
        validate_split_access(
            task="simulation",
            fraction=1.0,
            seed=42,
            fit_splits=("train", "test"),
            evaluation_split="validation",
            settings=settings,
            confirmatory_manifest=None,
            config_path=config,
            checkpoint_paths=checkpoints,
            dataset_manifest_sha256="dataset-hash",
            source_paths=sources,
        )

    manifest = tmp_path / "confirmatory.json"
    manifest.write_text(
        json.dumps(
            {
                "confirmatory_test_authorized": True,
                "protocol_frozen": True,
                "config_sha256": sha256_file(config),
                "checkpoint_sha256": {
                    "P25": {"42": sha256_file(p25)},
                    "CYCLIC25": {"42": sha256_file(cyclic)},
                },
                "dataset_manifest_sha256": {
                    "simulation": "dataset-hash",
                },
                "source_sha256": {
                    name: sha256_file(path) for name, path in sources.items()
                },
                "test_open_count": 0,
                "sealed_split_accessed": False,
                "confirmatory_design": {
                    "encoder_seeds": [42],
                    "methods": ["from_scratch", "P25", "CYCLIC25"],
                    "tasks": {
                        "simulation": {
                            "fit_splits": ["train", "validation"],
                            "evaluation_split": "test",
                            "fraction": 1.0,
                            "settings": asdict(settings),
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    payload = validate_split_access(
        task="simulation",
        fraction=1.0,
        seed=42,
        fit_splits=("train", "validation"),
        evaluation_split="test",
        settings=settings,
        confirmatory_manifest=manifest,
        config_path=config,
        checkpoint_paths=checkpoints,
        dataset_manifest_sha256="dataset-hash",
        source_paths=sources,
    )
    assert payload is not None
    assert payload["confirmatory_test_authorized"]

    tampered = json.loads(manifest.read_text(encoding="utf-8"))
    tampered["source_sha256"]["finetuning_module"] = "wrong"
    manifest.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PermissionError, match="source hash mismatch"):
        validate_split_access(
            task="simulation",
            fraction=1.0,
            seed=42,
            fit_splits=("train", "validation"),
            evaluation_split="test",
            settings=settings,
            confirmatory_manifest=manifest,
            config_path=config,
            checkpoint_paths=checkpoints,
            dataset_manifest_sha256="dataset-hash",
            source_paths=sources,
        )


def test_confirmatory_manifest_accepts_seed_indexed_checkpoint_hashes(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    p25 = tmp_path / "p25.pt"
    cyclic = tmp_path / "cyclic.pt"
    config.write_text("frozen: true\n", encoding="utf-8")
    p25.write_bytes(b"p25")
    cyclic.write_bytes(b"cyclic")
    sources = {
        "finetuning_entrypoint": tmp_path / "entrypoint.py",
        "finetuning_module": tmp_path / "module.py",
    }
    for name, path in sources.items():
        path.write_text(name, encoding="utf-8")
    settings = FineTuningConfig()
    manifest = tmp_path / "confirmatory.json"
    manifest.write_text(
        json.dumps(
            {
                "confirmatory_test_authorized": True,
                "protocol_frozen": True,
                "config_sha256": sha256_file(config),
                "checkpoint_sha256": {
                    "P25": {"42": sha256_file(p25)},
                    "CYCLIC25": {"42": sha256_file(cyclic)},
                },
                "dataset_manifest_sha256": {
                    "simulation": "dataset-hash",
                    "real": "other-hash",
                },
                "source_sha256": {
                    name: sha256_file(path) for name, path in sources.items()
                },
                "test_open_count": 0,
                "sealed_split_accessed": False,
                "confirmatory_design": {
                    "encoder_seeds": [42],
                    "methods": ["from_scratch", "P25", "CYCLIC25"],
                    "tasks": {
                        "simulation": {
                            "fit_splits": ["train", "validation"],
                            "evaluation_split": "test",
                            "fraction": 1.0,
                            "settings": asdict(settings),
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    assert validate_split_access(
        task="simulation",
        fraction=1.0,
        seed=42,
        fit_splits=("train", "validation"),
        evaluation_split="test",
        settings=settings,
        confirmatory_manifest=manifest,
        config_path=config,
        checkpoint_paths={"P25": p25, "CYCLIC25": cyclic},
        dataset_manifest_sha256="dataset-hash",
        source_paths=sources,
    )


def test_dataset_rejects_cross_split_group_leakage_without_training() -> None:
    data = FineTuningDataset(
        signals=np.zeros((6, 32), dtype=np.float32),
        targets=np.column_stack(
            (np.arange(6, dtype=np.float32), np.arange(6, dtype=np.float32))
        ),
        groups=np.asarray(["a", "a", "b", "b", "c", "c"]),
        sample_ids=np.asarray([f"s{index}" for index in range(6)]),
        splits=np.asarray(["train", "train", "train", "validation", "validation", "validation"]),
        task="simulation",
        target_names=("duration_ms", "doppler_khz"),
    )
    data.validate(input_length=32)
    with pytest.raises(ValueError, match="Group leakage"):
        validate_external_group_separation(
            data,
            fit_splits=("train",),
            evaluation_split="validation",
        )


def test_external_group_separation_supports_confirmatory_train_plus_validation() -> None:
    data = FineTuningDataset(
        signals=np.zeros((9, 32), dtype=np.float32),
        targets=np.tile(np.arange(3), 3),
        groups=np.asarray([f"g{index}" for index in range(9)]),
        sample_ids=np.asarray([f"s{index}" for index in range(9)]),
        splits=np.asarray(["train"] * 3 + ["validation"] * 3 + ["test"] * 3),
        task="real",
        target_names=REAL_CLASS_NAMES,
    )
    fit, evaluation = validate_external_group_separation(
        data,
        fit_splits=("train", "validation"),
        evaluation_split="test",
    )
    assert fit.tolist() == list(range(6))
    assert evaluation.tolist() == list(range(6, 9))
