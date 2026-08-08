#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from p3_ssl.bead_finetuning import (
    METHODS,
    FineTuningConfig,
    FineTuningDataset,
    run_paired_finetuning,
    sha256_file,
    validate_fraction,
)
from p3_ssl.bead_ssl_v2 import load_bead_ssl_v2_config
from p3_ssl.bead_ssl_v2_populations import load_v5_population, load_z8_v2_population


def _record(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    row = next((item for item in records if f"{item['id']}@{item['version']}" == key), None)
    if row is None or row["status"] not in {"active", "reference"}:
        raise ValueError(f"eligible registered dataset not found: {key}")
    return row


def _source_run(runs_root: Path, run_id: str, policy: str, seed: int) -> tuple[Path, dict[str, Any]]:
    root = runs_root / run_id
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if (
        run.get("status") != "complete"
        or run.get("profile") != "full"
        or run.get("training_mask_policy") != policy
        or int(run.get("seed", -1)) != seed
    ):
        raise ValueError(f"source run mismatch: {run_id}")
    checkpoint = root / "checkpoints/latest.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint, run


def _dataset(
    *,
    task: str,
    simulation_root: Path,
    event_root: Path,
    signal_root: Path,
    normalization: str,
) -> FineTuningDataset:
    if task == "simulation":
        populations = [
            ("train", load_v5_population(simulation_root, split="train", normalization=normalization)),
            ("val", load_v5_population(simulation_root, split="val", normalization=normalization)),
        ]
        target_names = ("duration_ms", "doppler_khz")
    else:
        populations = [
            ("train", load_z8_v2_population(event_root, signal_root, split="train", normalization=normalization)),
            ("val", load_z8_v2_population(event_root, signal_root, split="val", normalization=normalization)),
        ]
        target_names = ("2um", "4um", "10um")
    return FineTuningDataset(
        signals=np.concatenate([population.signals for _, population in populations]),
        targets=np.concatenate([population.labels for _, population in populations]),
        groups=np.concatenate([population.groups for _, population in populations]),
        sample_ids=np.concatenate([population.ids for _, population in populations]),
        splits=np.concatenate(
            [np.full(population.ids.size, split, dtype=object) for split, population in populations]
        ),
        task=task,
        target_names=target_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired end-to-end fine-tuning on Z8 v5/v2 development data.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", choices=("simulation", "real"), required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--p25-source-run-id", required=True)
    parser.add_argument("--cyclic25-source-run-id", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/bead_ssl_z8_v5_v2.yaml",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.seed not in range(42, 47):
        raise ValueError("v2 protocol requires seeds 42-46")
    fraction = validate_fraction(args.task, args.fraction)
    settings = FineTuningConfig(epochs=args.epochs, batch_size=args.batch_size)
    settings.validate()

    workspace = Workspace.load()
    config = load_bead_ssl_v2_config(args.config)
    records = [record.payload for record in load_records(workspace)]
    study = config["study"]
    simulation = _record(records, study["simulation_dataset"])
    events = _record(records, study["real_event_dataset"])
    signals = _record(records, study["real_signal_dataset"])
    roots = {
        "simulation": workspace.datasets_root / simulation["path"],
        "events": workspace.datasets_root / events["path"],
        "signals": workspace.datasets_root / signals["path"],
    }
    runs_root = workspace.artifacts_root / "unsupervised-learning-flow-cytometry/runs"
    p25_checkpoint, p25_run = _source_run(runs_root, args.p25_source_run_id, "P25", args.seed)
    cyclic_checkpoint, cyclic_run = _source_run(
        runs_root, args.cyclic25_source_run_id, "CYCLIC25", args.seed
    )
    if int(p25_run["epochs"]) != int(cyclic_run["epochs"]):
        raise ValueError("paired source runs do not share the frozen epoch budget")
    expected_epoch = int(p25_run["epochs"])
    config["training"]["profiles"]["full"]["epochs"] = expected_epoch
    data = _dataset(
        task=args.task,
        simulation_root=roots["simulation"],
        event_root=roots["events"],
        signal_root=roots["signals"],
        normalization=str(config["data"]["normalization"]),
    )
    data.validate(input_length=int(config["data"]["input_length"]))
    output = runs_root / args.run_id
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    states = {
        "workspace": collect_git_state(workspace.root),
        "unsupervised-learning-flow-cytometry": collect_git_state(workspace.root / "unsupervised-learning-flow-cytometry"),
    }
    dataset_records = {"simulation": simulation, "real_events": events, "real_signals": signals}
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "kind": "bead-downstream-finetuning-v2",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "fraction": fraction,
        "seed": args.seed,
        "fit_splits": ["train"],
        "evaluation_split": "val",
        "sealed_splits_used": [],
        "dataset": ",".join(f"{row['id']}@{row['version']}" for row in dataset_records.values()),
        "datasets": {
            name: {
                "id": f"{row['id']}@{row['version']}",
                "manifest": row["manifest"],
                "manifest_sha256": row["manifest_sha256"],
            }
            for name, row in dataset_records.items()
        },
        "source_runs": [args.p25_source_run_id, args.cyclic25_source_run_id],
        "checkpoints": {
            "P25": {"path": str(p25_checkpoint), "sha256": sha256_file(p25_checkpoint)},
            "CYCLIC25": {"path": str(cyclic_checkpoint), "sha256": sha256_file(cyclic_checkpoint)},
        },
        "checkpoint_epoch": expected_epoch,
        "simulation_target_semantics": {
            "duration_ms": "tau_ms (Gaussian sigma in the physical waveform model)",
            "doppler_khz": "frequency_khz",
        },
        "source_sha256": {
            "config": sha256_file(args.config),
            "entrypoint": sha256_file(Path(__file__).resolve()),
            "finetuning_module": sha256_file(Path(__file__).resolve().parents[1] / "p3_ssl/bead_finetuning.py"),
            "population_module": sha256_file(Path(__file__).resolve().parents[1] / "p3_ssl/bead_ssl_v2_populations.py"),
        },
        "repositories": {name: state["revision"] for name, state in states.items()},
        "repository_dirty": {name: state["dirty"] for name, state in states.items()},
        "command": shlex.join(sys.argv),
        "settings": settings.__dict__,
        "outputs": [
            "metrics.json",
            "metrics.csv",
            *[
                f"{method}_{suffix}"
                for method in METHODS
                for suffix in ("metrics.json", "predictions.csv", "checkpoint.pt")
            ],
        ],
    }
    (output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        result = run_paired_finetuning(
            config,
            data,
            fraction=fraction,
            fit_splits=("train",),
            evaluation_split="val",
            seed=args.seed,
            p25_checkpoint=p25_checkpoint,
            cyclic25_checkpoint=cyclic_checkpoint,
            settings=settings,
            device=torch.device(args.device),
            output_dir=output,
            expected_checkpoint_epoch=expected_epoch,
            protocol="bead-downstream-finetuning-v2",
        )
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        (output / "run.json").write_text(
            json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    run["status"] = "complete"
    run["completed_at"] = datetime.now(timezone.utc).isoformat()
    run["counts"] = result["counts"]
    (output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
