#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from internship_workspace.config import Workspace
from internship_workspace.datasets import load_records
from p3_ssl.bead_ssl import configure_experiment, evaluate_reconstruction, make_model
from p3_ssl.bead_ssl_v2 import Z8RealValidationDataset, load_bead_ssl_v2_config


POLICIES = ("P25", "CYCLIC25")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result = next(
        (row for row in records if f"{row['id']}@{row['version']}" == key), None
    )
    if result is None or result["status"] not in {"active", "reference"}:
        raise ValueError(f"eligible registered dataset not found: {key}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export deterministic P25/CYCLIC25 reconstruction examples."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "configs/bead_ssl_z8_v5_v2.yaml"
        ),
    )
    parser.add_argument("--source-run-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if len(args.source_run_id) != 2:
        raise ValueError("exactly one P25 and one CYCLIC25 source run are required")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")

    workspace = Workspace.load()
    config = load_bead_ssl_v2_config(args.config)
    records = [record.payload for record in load_records(workspace)]
    study = config["study"]
    event_record = _record(records, study["real_event_dataset"])
    signal_record = _record(records, study["real_signal_dataset"])
    dataset = Z8RealValidationDataset(
        workspace.datasets_root / event_record["path"],
        workspace.datasets_root / signal_record["path"],
        split=config["data"]["real_validation_split"],
        normalization=config["data"]["normalization"],
    )
    if len(dataset) < 3:
        raise ValueError("real validation dataset has fewer than three examples")
    loader = DataLoader(
        Subset(dataset, range(3)), batch_size=3, shuffle=False, num_workers=0
    )
    device = torch.device(args.device)
    runs_root = (
        workspace.artifacts_root / "unsupervised-learning-flow-cytometry/runs"
    )
    args.output_dir.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "protocol": "bead-ssl-v2-visual-example-export-v1",
        "dataset": f"{event_record['id']}@{event_record['version']}",
        "selection_rule": "first three deterministic real validation examples",
        "sealed_splits_used": [],
        "sources": {},
        "outputs": [],
    }
    observed_policies: set[str] = set()
    for run_id in args.source_run_id:
        source = runs_root / run_id
        run = json.loads((source / "run.json").read_text(encoding="utf-8"))
        policy = str(run["training_mask_policy"])
        if policy not in POLICIES or policy in observed_policies:
            raise ValueError(f"invalid or duplicate source policy: {policy}")
        observed_policies.add(policy)
        checkpoint_path = source / "checkpoints/latest.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        policy_config = configure_experiment(
            config, loss_cell="B0", mask_policy=policy, seed=int(run["seed"])
        )
        model = make_model(policy_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        manifest["sources"][policy] = {
            "run_id": run_id,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_sha256": _sha256(checkpoint_path),
        }
        for evaluation_policy in POLICIES:
            _, examples = evaluate_reconstruction(
                model,
                loader,
                policy_config,
                device,
                mask_seed=int(run["seed"]),
                evaluation_policy=evaluation_policy,
                max_examples=3,
            )
            output_name = f"train_{policy.lower()}_evaluate_{evaluation_policy.lower()}.npz"
            np.savez_compressed(args.output_dir / output_name, **examples)
            manifest["outputs"].append(output_name)
            manifest.setdefault("sample_ids", {})[evaluation_policy] = [
                str(value) for value in examples["sample_id"]
            ]
    if observed_policies != set(POLICIES):
        raise ValueError("one source per policy is required")
    (args.output_dir / "visual_examples_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
