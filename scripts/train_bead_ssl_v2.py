#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from internship_workspace.provenance import collect_git_state
from internship_workspace.visual_review_store import ReviewStore
from p3_ssl.bead_ssl import configure_experiment
from p3_ssl.bead_ssl_v2 import load_bead_ssl_v2_config, train_bead_ssl_v2


METHOD_RUN_ID = "bead-ssl-v2-matched-loss-method-r1"
METHOD_EVIDENCE_ID = "bead-ssl-v2-matched-loss-method"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(records: list[dict], key: str) -> dict:
    result = next(
        (row for row in records if f"{row['id']}@{row['version']}" == key),
        None,
    )
    if result is None or result["status"] not in {"active", "reference"}:
        raise ValueError(f"eligible registered dataset not found: {key}")
    return result


def _verify_method(workspace: Workspace) -> dict[str, str]:
    store = ReviewStore(
        workspace.artifacts_root / "cross-project/reviews" / METHOD_RUN_ID
    )
    receipt = store.verify_receipt()
    decision = store.current()["decisions"][METHOD_EVIDENCE_ID]["decision"]
    if decision != "approved":
        raise PermissionError("bead SSL v2 method is not approved")
    return {
        "run_id": METHOD_RUN_ID,
        "evidence_id": METHOD_EVIDENCE_ID,
        "decision": decision,
        "receipt_sha256": _sha256(store.receipt_path),
        "contract_sha256": str(receipt["contract_sha256"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one paired Z8-v5 bead SSL v2 cell."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "configs/bead_ssl_z8_v5_v2.yaml"
        ),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--profile", choices=("smoke", "pilot", "full"), required=True
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--mask-policy", choices=("P25", "CYCLIC25"), required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    workspace = Workspace.load()
    method = _verify_method(workspace)
    config = configure_experiment(
        load_bead_ssl_v2_config(args.config),
        loss_cell="B0",
        mask_policy=args.mask_policy,
        seed=args.seed,
    )
    if args.epochs is not None:
        if args.epochs < 1 or args.epochs > 60:
            raise ValueError("epochs must be between 1 and 60")
        config["training"]["profiles"][args.profile]["epochs"] = args.epochs
    records = [record.payload for record in load_records(workspace)]
    study = config["study"]
    simulation = _record(records, study["simulation_dataset"])
    real_events = _record(records, study["real_event_dataset"])
    real_signals = _record(records, study["real_signal_dataset"])
    output_dir = (
        workspace.artifacts_root
        / "unsupervised-learning-flow-cytometry/runs"
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
    dataset_records = {
        "simulation": simulation,
        "real_events": real_events,
        "real_signals": real_signals,
    }
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "command": shlex.join([sys.executable, *sys.argv]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "dataset": ",".join(
            str(record["id"]) + "@" + str(record["version"])
            for record in dataset_records.values()
        ),
        "datasets": {
            name: {
                "id": f"{record['id']}@{record['version']}",
                "manifest": record["manifest"],
                "manifest_sha256": record["manifest_sha256"],
            }
            for name, record in dataset_records.items()
        },
        "method_evidence": method,
        "repositories": {
            name: state["revision"] for name, state in repository_states.items()
        },
        "repository_dirty": {
            name: state["dirty"] for name, state in repository_states.items()
        },
        "source_sha256": {
            "config": _sha256(args.config),
            "training_module": _sha256(
                Path(__file__).resolve().parents[1] / "p3_ssl/bead_ssl.py"
            ),
            "v2_module": _sha256(
                Path(__file__).resolve().parents[1] / "p3_ssl/bead_ssl_v2.py"
            ),
            "entrypoint": _sha256(Path(__file__).resolve()),
        },
        "profile": args.profile,
        "epochs": int(config["training"]["profiles"][args.profile]["epochs"]),
        "seed": args.seed,
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
        result = train_bead_ssl_v2(
            config,
            simulation_root=workspace.datasets_root / simulation["path"],
            real_event_root=workspace.datasets_root / real_events["path"],
            real_signal_root=workspace.datasets_root / real_signals["path"],
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
    run["selected_epoch"] = result["selected_epoch"]
    run["monitoring"] = result.get("monitoring")
    (output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
