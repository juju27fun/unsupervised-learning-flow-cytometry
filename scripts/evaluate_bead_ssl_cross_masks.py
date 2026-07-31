#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.provenance import collect_git_state
from p3_ssl.bead_ssl import (
    RealBeadValidationDataset,
    SingleBeadSimulationDataset,
    configure_experiment,
    evaluate_reconstruction,
    load_bead_ssl_config,
    make_model,
)


POLICIES = ("P25", "CYCLIC25")
SEEDS = (42, 43, 44)
CHECKPOINTS = {
    "p25_selected": "best.pt",
    "epoch20": "latest.pt",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result = next(
        (
            row
            for row in records
            if f"{row['id']}@{row['version']}" == key
        ),
        None,
    )
    if result is None or result["status"] not in {"active", "reference"}:
        raise ValueError(f"Eligible registered dataset not found: {key}")
    return result


def source_run_id(policy: str, seed: int) -> str:
    revision = "cmp2" if policy == "CYCLIC25" else "cmp1"
    return f"bead-ssl-{policy.lower()}-b0-full-s{seed}-v1-{revision}"


def _load_source(
    runs_root: Path,
    policy: str,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    source = runs_root / source_run_id(policy, seed)
    run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise ValueError(f"Incomplete source run: {source.name}")
    if run.get("training_mask_policy") != policy or run.get("seed") != seed:
        raise ValueError(f"Source run metadata mismatch: {source.name}")
    for filename in CHECKPOINTS.values():
        if not (source / "checkpoints" / filename).is_file():
            raise FileNotFoundError(source / "checkpoints" / filename)
    return source, run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained bead SSL B0 models under P25 and CYCLIC25 masks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bead_ssl_p25_v1.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    workspace = Workspace.load()
    base_config = load_bead_ssl_config(args.config)
    records = [record.payload for record in load_records(workspace)]
    simulation_record = _record(
        records, base_config["study"]["simulation_dataset"]
    )
    real_record = _record(records, base_config["study"]["real_dataset"])
    simulation_root = workspace.datasets_root / simulation_record["path"]
    real_root = workspace.datasets_root / real_record["path"]
    runs_root = (
        workspace.artifacts_root
        / "unsupervised-learning-flow-cytometry"
        / "runs"
    )
    output = (
        workspace.artifacts_root
        / "unsupervised-learning-flow-cytometry"
        / "evaluations"
        / args.run_id
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output}")
    output.mkdir(parents=True)

    repository_states = {
        "workspace": collect_git_state(workspace.root),
        "unsupervised-learning-flow-cytometry": collect_git_state(
            workspace.root / "unsupervised-learning-flow-cytometry"
        ),
        "particles2SNR-pipeline": collect_git_state(
            workspace.root / "particles2SNR-pipeline"
        ),
    }
    run: dict[str, Any] = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "kind": "bead-ssl-cross-mask-evaluation",
        "dataset": (
            f"{base_config['study']['simulation_dataset']} + "
            f"{base_config['study']['real_dataset']}"
        ),
        "datasets": {
            "simulation": {
                "id": base_config["study"]["simulation_dataset"],
                "manifest": simulation_record["manifest"],
                "manifest_sha256": simulation_record["manifest_sha256"],
            },
            "real": {
                "id": base_config["study"]["real_dataset"],
                "manifest": real_record["manifest"],
                "manifest_sha256": real_record["manifest_sha256"],
            },
        },
        "source_runs": [
            source_run_id(policy, seed)
            for policy in POLICIES
            for seed in SEEDS
        ],
        "repositories": {
            name: state["revision"] for name, state in repository_states.items()
        },
        "repository_dirty": {
            name: state["dirty"] for name, state in repository_states.items()
        },
        "source_sha256": {
            "config": _sha256(args.config),
            "evaluation_module": _sha256(
                Path(__file__).resolve().parents[1] / "p3_ssl/bead_ssl.py"
            ),
            "masking_module": _sha256(
                Path(__file__).resolve().parents[1] / "p3_ssl/masking.py"
            ),
            "entrypoint": _sha256(Path(__file__).resolve()),
        },
        "command": (
            "unsupervised-learning-flow-cytometry/scripts/"
            f"evaluate_bead_ssl_cross_masks.py --config {args.config} "
            f"--run-id {args.run_id} --device {args.device}"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "evaluation_policies": list(POLICIES),
        "checkpoint_policies": list(CHECKPOINTS),
        "cyclic_evaluation_schedule": "all_unique_passes_per_sample",
        "real_cyclic_support": "reviewed_annotation_bounds",
        "sealed_splits_used": [],
        "outputs": ["metrics.json"],
    }
    (output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    profile = base_config["training"]["profiles"]["full"]
    simulation_dataset = SingleBeadSimulationDataset(
        simulation_root,
        split=base_config["data"]["simulation_validation_split"],
        normalization=base_config["data"]["normalization"],
        sampling_frequency_hz=float(
            base_config["data"]["sampling_frequency_hz"]
        ),
    )
    real_dataset = RealBeadValidationDataset(
        real_root,
        split=base_config["data"]["real_validation_split"],
        normalization=base_config["data"]["normalization"],
    )
    simulation_loader = DataLoader(
        simulation_dataset,
        batch_size=int(profile["batch_size"]),
        shuffle=False,
        num_workers=int(profile["num_workers"]),
    )
    real_loader = DataLoader(
        real_dataset,
        batch_size=int(profile["batch_size"]),
        shuffle=False,
        num_workers=int(profile["num_workers"]),
    )
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    try:
        for training_policy in POLICIES:
            for seed in SEEDS:
                source, _source_run = _load_source(
                    runs_root, training_policy, seed
                )
                config = configure_experiment(
                    base_config,
                    loss_cell="B0",
                    mask_policy=training_policy,
                    seed=seed,
                )
                for checkpoint_policy, filename in CHECKPOINTS.items():
                    checkpoint = torch.load(
                        source / "checkpoints" / filename,
                        map_location=device,
                        weights_only=False,
                    )
                    model = make_model(config).to(device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    for evaluation_policy in POLICIES:
                        simulation_metrics, _ = evaluate_reconstruction(
                            model,
                            simulation_loader,
                            config,
                            device,
                            mask_seed=seed,
                            evaluation_policy=evaluation_policy,
                            max_examples=0,
                        )
                        real_metrics, _ = evaluate_reconstruction(
                            model,
                            real_loader,
                            config,
                            device,
                            mask_seed=seed,
                            evaluation_policy=evaluation_policy,
                            max_examples=0,
                        )
                        row = {
                            "source_run_id": source.name,
                            "training_mask_policy": training_policy,
                            "evaluation_mask_policy": evaluation_policy,
                            "checkpoint_policy": checkpoint_policy,
                            "checkpoint_epoch": int(checkpoint["epoch"]),
                            "seed": seed,
                            "simulation_validation": simulation_metrics,
                            "real_validation": real_metrics,
                        }
                        rows.append(row)
                        print(
                            json.dumps(
                                {
                                    "training": training_policy,
                                    "evaluation": evaluation_policy,
                                    "checkpoint": checkpoint_policy,
                                    "seed": seed,
                                    "simulation_mse": simulation_metrics[
                                        "model"
                                    ]["masked_mse"],
                                    "real_mse": real_metrics["model"][
                                        "masked_mse"
                                    ],
                                },
                                sort_keys=True,
                            )
                        )
        result = {
            "protocol": "bead-ssl-cross-mask-evaluation-v1",
            "counts": {
                "simulation_validation": len(simulation_dataset),
                "real_validation": len(real_dataset),
                "rows": len(rows),
            },
            "cyclic_evaluation_schedule": "all_unique_passes_per_sample",
            "real_cyclic_support": "reviewed_annotation_bounds",
            "rows": rows,
        }
        (output / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        (output / "run.json").write_text(
            json.dumps(run, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    run["status"] = "complete"
    (output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
