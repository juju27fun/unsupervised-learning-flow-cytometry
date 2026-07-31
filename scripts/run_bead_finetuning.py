#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import numpy as np

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.provenance import collect_git_state
from p3_ssl.bead_finetuning import (
    METHODS,
    FineTuningDataset,
    FineTuningConfig,
    load_finetuning_dataset,
    run_paired_finetuning,
    sha256_file,
    validate_fraction,
    validate_split_access,
)
from p3_ssl.bead_representation_benchmark import (
    load_real_population,
    load_simulation_population,
)
from p3_ssl.bead_ssl import load_bead_ssl_config


def _record(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    record = next(
        (
            row
            for row in records
            if f"{row['id']}@{row['version']}" == key
        ),
        None,
    )
    if record is None or record["status"] not in {"active", "reference"}:
        raise ValueError(f"Eligible registered dataset not found: {key}")
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune paired from-scratch, P25, and CYCLIC25 bead encoders "
            "with a shared seed, label subset, internal group-safe calibration "
            "split, and optimization budget."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Frozen bead SSL YAML config used by both source checkpoints.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Registered dataset identifier including version, for example id@v1.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help=(
            "Optional NPZ inside the registered dataset. By default, load the "
            "registered simulation or reviewed-event native format directly."
        ),
    )
    parser.add_argument(
        "--task",
        choices=("simulation", "real"),
        required=True,
        help="Simulation is two-output regression; real is 3-class bead classification.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Simulation: 0.10 or 1.0. Real: 0.25 or 1.0.",
    )
    parser.add_argument(
        "--fit-split",
        action="append",
        dest="fit_splits",
        help=(
            "Fit split; repeat for confirmatory train+validation. "
            "Defaults to train."
        ),
    )
    parser.add_argument(
        "--evaluation-split",
        required=True,
        help="Development split (validation/val) or sealed test split.",
    )
    parser.add_argument("--p25-checkpoint", type=Path, required=True)
    parser.add_argument("--cyclic25-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--confirmatory-manifest",
        type=Path,
        help=(
            "Required only for test. JSON must freeze and authorize the protocol "
            "and match config, P25/CYCLIC25 checkpoint, and dataset-manifest hashes."
        ),
    )
    return parser


def _native_dataset(
    root: Path,
    *,
    task: str,
    requested_splits: tuple[str, ...],
    normalization: str,
    allow_test: bool,
) -> FineTuningDataset:
    populations = []
    for split in requested_splits:
        if task == "simulation":
            population = load_simulation_population(
                root,
                split=split,
                normalization=normalization,
                allow_test=allow_test,
            )
        else:
            population = load_real_population(
                root,
                split=split,
                normalization=normalization,
                allow_test=allow_test,
            )
        populations.append((split, population))
    return FineTuningDataset(
        signals=np.concatenate([row.signals for _, row in populations]),
        targets=np.concatenate([row.labels for _, row in populations]),
        groups=np.concatenate([row.groups for _, row in populations]),
        sample_ids=np.concatenate([row.ids for _, row in populations]),
        splits=np.concatenate(
            [
                np.full(row.ids.size, split, dtype=object)
                for split, row in populations
            ]
        ),
        task=task,
        target_names=(
            ("duration_ms", "doppler_khz")
            if task == "simulation"
            else ("2um", "4um", "10um")
        ),
    )


def main() -> None:
    args = _parser().parse_args()
    fit_splits = tuple(args.fit_splits or ("train",))
    fraction = validate_fraction(args.task, args.fraction)
    settings = FineTuningConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        calibration_fraction=args.calibration_fraction,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    settings.validate()
    workspace = Workspace.load()
    config = load_bead_ssl_config(args.config)
    records = [record.payload for record in load_records(workspace)]
    dataset_record = _record(records, args.dataset_id)
    dataset_root = workspace.datasets_root / dataset_record["path"]
    dataset_path = None
    if args.dataset_file is not None:
        if args.dataset_file.is_absolute() or ".." in args.dataset_file.parts:
            raise ValueError("--dataset-file must stay inside the registered dataset")
        dataset_path = dataset_root / args.dataset_file
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
    checkpoint_paths = {
        "P25": args.p25_checkpoint,
        "CYCLIC25": args.cyclic25_checkpoint,
    }
    for path in checkpoint_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            path.resolve().relative_to(workspace.artifacts_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Checkpoint must be under the workspace artifacts root: {path}"
            ) from exc
    confirmatory = validate_split_access(
        task=args.task,
        fraction=fraction,
        seed=args.seed,
        fit_splits=fit_splits,
        evaluation_split=args.evaluation_split,
        settings=settings,
        confirmatory_manifest=args.confirmatory_manifest,
        config_path=args.config,
        checkpoint_paths=checkpoint_paths,
        dataset_manifest_sha256=dataset_record["manifest_sha256"],
        source_paths={
            "finetuning_entrypoint": Path(__file__).resolve(),
            "finetuning_module": (
                workspace.root
                / "unsupervised-learning-flow-cytometry"
                / "p3_ssl"
                / "bead_finetuning.py"
            ),
        },
    )
    if dataset_path is not None:
        data = load_finetuning_dataset(
            dataset_path,
            task=args.task,
            input_length=int(config["data"]["input_length"]),
            normalization=str(config["data"]["normalization"]),
        )
    else:
        requested_splits = tuple(dict.fromkeys((*fit_splits, args.evaluation_split)))
        data = _native_dataset(
            dataset_root,
            task=args.task,
            requested_splits=requested_splits,
            normalization=str(config["data"]["normalization"]),
            allow_test=confirmatory is not None,
        )
        data.validate(input_length=int(config["data"]["input_length"]))
    output_dir = (
        workspace.artifacts_root
        / "unsupervised-learning-flow-cytometry"
        / "runs"
        / args.run_id
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    output_dir.mkdir(parents=True)

    repository_states = {
        "workspace": collect_git_state(workspace.root),
        "unsupervised-learning-flow-cytometry": collect_git_state(
            workspace.root / "unsupervised-learning-flow-cytometry"
        ),
    }
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "kind": "bead-downstream-finetuning",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "fraction": fraction,
        "seed": args.seed,
        "fit_splits": list(fit_splits),
        "evaluation_split": args.evaluation_split,
        "sealed_splits_used": (
            ["test"]
            if "test" in {*fit_splits, args.evaluation_split}
            else []
        ),
        "confirmatory_manifest": (
            {
                "path": str(args.confirmatory_manifest),
                "sha256": sha256_file(args.confirmatory_manifest),
            }
            if confirmatory is not None
            else None
        ),
        "dataset": args.dataset_id,
        "datasets": {
            "downstream": {
                "id": args.dataset_id,
                "manifest": dataset_record["manifest"],
                "manifest_sha256": dataset_record["manifest_sha256"],
                "file": str(args.dataset_file) if args.dataset_file else None,
                "file_sha256": (
                    sha256_file(dataset_path) if dataset_path is not None else None
                ),
            }
        },
        "checkpoints": {
            policy: {"path": str(path), "sha256": sha256_file(path)}
            for policy, path in checkpoint_paths.items()
        },
        "source_sha256": {
            "config": sha256_file(args.config),
            "finetuning_module": sha256_file(
                Path(__file__).resolve().parents[1]
                / "p3_ssl"
                / "bead_finetuning.py"
            ),
            "benchmark_module": sha256_file(
                Path(__file__).resolve().parents[1]
                / "p3_ssl"
                / "bead_representation_benchmark.py"
            ),
            "bead_ssl_module": sha256_file(
                Path(__file__).resolve().parents[1] / "p3_ssl" / "bead_ssl.py"
            ),
            "model_module": sha256_file(
                Path(__file__).resolve().parents[1] / "p3_ssl" / "models.py"
            ),
            "decimation_module": sha256_file(
                Path(__file__).resolve().parents[1]
                / "p3_ssl"
                / "decimation.py"
            ),
            "particle_class_coverage_module": sha256_file(
                workspace.root
                / "particles2SNR-pipeline"
                / "particles2snr"
                / "particle_class_coverage.py"
            ),
            "ssl_realism_audit_module": sha256_file(
                workspace.root
                / "particles2SNR-pipeline"
                / "particles2snr"
                / "ssl_realism_audit.py"
            ),
            "entrypoint": sha256_file(Path(__file__).resolve()),
        },
        "repositories": {
            name: state["revision"] for name, state in repository_states.items()
        },
        "repository_dirty": {
            name: state["dirty"] for name, state in repository_states.items()
        },
        "command": shlex.join(sys.argv),
        "settings": settings.__dict__,
        "outputs": [
            "metrics.json",
            "metrics.csv",
            *[
                f"{method}_{suffix}"
                for method in METHODS
                for suffix in (
                    "metrics.json",
                    "predictions.csv",
                    "checkpoint.pt",
                )
            ],
        ],
    }
    (output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        result = run_paired_finetuning(
            config,
            data,
            fraction=fraction,
            fit_splits=fit_splits,
            evaluation_split=args.evaluation_split,
            seed=args.seed,
            p25_checkpoint=args.p25_checkpoint,
            cyclic25_checkpoint=args.cyclic25_checkpoint,
            settings=settings,
            device=torch.device(args.device),
            output_dir=output_dir,
        )
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        (output_dir / "run.json").write_text(
            json.dumps(run, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    run["status"] = "complete"
    run["completed_at"] = datetime.now(timezone.utc).isoformat()
    run["counts"] = result["counts"]
    (output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
