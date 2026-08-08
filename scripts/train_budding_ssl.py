#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.provenance import collect_git_state
from p3_ssl.budding_ssl import (
    load_budding_ssl_config,
    train_budding_ssl,
)


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
        raise ValueError(f"eligible registered dataset not found: {key}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the controlled data-oriented budding masked learner."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/yeast_budding_ssl_b0_v1.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--visual-approval-receipt",
        type=Path,
        help="Receipt-backed approval required for the full profile.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    workspace = Workspace.load()
    config = load_budding_ssl_config(args.config)
    config["training"]["seed"] = int(args.seed)
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
        raise FileExistsError(f"refusing to overwrite run: {output_dir}")

    repository_states = {
        "workspace": collect_git_state(workspace.root),
        "unsupervised-learning-flow-cytometry": collect_git_state(
            workspace.root / "unsupervised-learning-flow-cytometry"
        ),
        "particles2SNR-pipeline": collect_git_state(
            workspace.root / "particles2SNR-pipeline"
        ),
    }
    project_root = Path(__file__).resolve().parents[1]
    receipt_argument = (
        ""
        if args.visual_approval_receipt is None
        else f" --visual-approval-receipt {args.visual_approval_receipt}"
    )
    command = (
        "unsupervised-learning-flow-cytometry/scripts/train_budding_ssl.py "
        f"--config {args.config} --run-id {args.run_id} "
        f"--profile {args.profile} --seed {args.seed} --device {args.device}"
        f"{receipt_argument}"
    )
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
                "source_group": config["data"]["real_source_group"],
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
            "training_module": _sha256(project_root / "p3_ssl/budding_ssl.py"),
            "synthetic_loader": _sha256(
                project_root / "p3_ssl/budding_synthetic_data.py"
            ),
            "real_loader": _sha256(project_root / "p3_ssl/study_data.py"),
            "masking_module": _sha256(project_root / "p3_ssl/masking.py"),
            "model_module": _sha256(project_root / "p3_ssl/models.py"),
            "entrypoint": _sha256(Path(__file__).resolve()),
        },
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "profile": args.profile,
        "seed": int(args.seed),
        "training_stage": "synthetic_only",
        "real_training_usage": "forbidden",
        "sealed_splits_used": [],
        "visual_checkpoint": {
            "required_before_full_gpu": True,
            "approved": False,
            "next_stage_blocked": args.profile == "full",
        },
        "outputs": [
            "checkpoints/best.pt",
            "checkpoints/latest.pt",
            "history.json",
            "metrics.json",
            "simulation_reconstruction_examples.npz",
            "real_reconstruction_examples.npz",
        ],
    }
    if args.profile == "full":
        if args.visual_approval_receipt is None:
            raise PermissionError(
                "full GPU execution requires --visual-approval-receipt"
            )
        receipt_path = args.visual_approval_receipt.resolve()
        if receipt_path.name != "receipt.json" or receipt_path.parent.name != "review":
            raise PermissionError("visual approval must point to review/receipt.json")
        try:
            receipt_reference = receipt_path.relative_to(workspace.root.resolve())
        except ValueError as exc:
            raise PermissionError(
                "visual approval receipt must be stored inside the workspace"
            ) from exc
        checkpoint_dir = receipt_path.parent.parent
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        contract_path = checkpoint_dir / str(receipt["contract_file"])
        decisions_path = checkpoint_dir / str(receipt["decisions_file"])
        if (
            _sha256(contract_path) != receipt["contract_sha256"]
            or _sha256(decisions_path) != receipt["decisions_sha256"]
        ):
            raise PermissionError("visual approval receipt hash verification failed")
        checkpoint_run = json.loads(
            (checkpoint_dir / "run.json").read_text(encoding="utf-8")
        )
        if (
            checkpoint_run.get("status") != "visual_review_complete"
            or not checkpoint_run.get("visual_checkpoint", {}).get("approved")
        ):
            raise PermissionError("visual checkpoint receipt is not approved")
        run["visual_checkpoint"] = {
            "required_before_full_gpu": True,
            "approved": True,
            "next_stage_blocked": False,
            "checkpoint_run_id": receipt["run_id"],
            "receipt": receipt_reference.as_posix(),
        }
    try:
        result = train_budding_ssl(
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
    run["optimizer_updates"] = result["optimizer_updates"]
    run["counts"] = result["counts"]
    run["best_coverage_cycle"] = result["best_coverage_cycle"]
    run["completed_coverage_cycles"] = result["completed_coverage_cycles"]
    run["training_control"] = result["training_control"]
    (output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
