#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.provenance import collect_git_state
from p3_ssl.bead_ssl import (
    configure_experiment,
    load_bead_ssl_config,
    train_bead_ssl,
)
from p3_ssl.config import validate_bead_simulation_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(records: list[dict], key: str) -> dict:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one registered bead SSL comparison cell."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bead_ssl_p25_v1.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--loss-cell",
        choices=("B0", "B1", "B2", "B3"),
        default="B0",
    )
    parser.add_argument(
        "--mask-policy",
        choices=("P25", "CYCLIC25"),
        default="P25",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    workspace = Workspace.load()
    config = configure_experiment(
        load_bead_ssl_config(args.config),
        loss_cell=args.loss_cell,
        mask_policy=args.mask_policy,
        seed=args.seed,
    )
    validate_bead_simulation_dataset(config)
    records = [record.payload for record in load_records(workspace)]
    simulation = _record(records, config["study"]["simulation_dataset"])
    real = _record(records, config["study"]["real_dataset"])
    output_dir = (
        workspace.artifacts_root
        / "unsupervised-learning-flow-cytometry"
        / "runs"
        / args.run_id
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output_dir}")
    repository_states = {
        "workspace": collect_git_state(workspace.root),
        "unsupervised-learning-flow-cytometry": collect_git_state(
            workspace.root / "unsupervised-learning-flow-cytometry"
        ),
        "particles2SNR-pipeline": collect_git_state(
            workspace.root / "particles2SNR-pipeline"
        ),
    }
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": (
            f"{config['study']['simulation_dataset']} + "
            f"{config['study']['real_dataset']}"
        ),
        "datasets": {
            "simulation": {
                "id": config["study"]["simulation_dataset"],
                "manifest": simulation["manifest"],
                "manifest_sha256": simulation["manifest_sha256"],
            },
            "real": {
                "id": config["study"]["real_dataset"],
                "manifest": real["manifest"],
                "manifest_sha256": real["manifest_sha256"],
            },
        },
        "repositories": {
            name: state["revision"] for name, state in repository_states.items()
        },
        "repository_dirty": {
            name: state["dirty"] for name, state in repository_states.items()
        },
        "repository_provenance_errors": {
            name: state["error"]
            for name, state in repository_states.items()
            if state["error"] is not None
        },
        "source_sha256": {
            "config": _sha256(args.config),
            "training_module": _sha256(
                Path(__file__).resolve().parents[1] / "p3_ssl/bead_ssl.py"
            ),
            "masking_module": _sha256(
                Path(__file__).resolve().parents[1] / "p3_ssl/masking.py"
            ),
            "loss_module": _sha256(
                Path(__file__).resolve().parents[1] / "p3_ssl/losses.py"
            ),
            "model_module": _sha256(
                Path(__file__).resolve().parents[1] / "p3_ssl/models.py"
            ),
            "entrypoint": _sha256(Path(__file__).resolve()),
        },
        "command": (
            "unsupervised-learning-flow-cytometry/scripts/train_bead_ssl_b0.py "
            f"--config {args.config} --run-id {args.run_id} "
            f"--profile {args.profile} --loss-cell {args.loss_cell} "
            f"--mask-policy {args.mask_policy} --seed {args.seed} "
            f"--device {args.device}"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "profile": args.profile,
        "seed": int(config["training"]["seed"]),
        "training_stage": "synthetic_only",
        "loss_cell": args.loss_cell,
        "training_mask_policy": args.mask_policy,
        "evaluation_mask_policy": config["masking"]["evaluation_policy"],
        "sealed_splits_used": [],
        "outputs": [
            "checkpoints/best.pt",
            "checkpoints/latest.pt",
            "history.json",
            "metrics.json",
            "simulation_reconstruction_examples.npz",
            "real_reconstruction_examples.npz",
        ],
    }
    try:
        result = train_bead_ssl(
            config,
            simulation_root=workspace.datasets_root / simulation["path"],
            real_root=workspace.datasets_root / real["path"],
            output_dir=output_dir,
            profile_name=args.profile,
            device_name=args.device,
        )
    except Exception as exc:
        if output_dir.exists():
            run["status"] = "failed"
            run["error"] = f"{type(exc).__name__}: {exc}"
            (output_dir / "run.json").write_text(
                json.dumps(run, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        raise
    run["status"] = "complete"
    run["seed"] = result["seed"]
    (output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
