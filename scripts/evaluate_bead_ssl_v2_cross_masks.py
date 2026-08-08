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
from internship_workspace.scientific_visual import computation_fingerprint
from p3_ssl.bead_ssl import configure_experiment, evaluate_reconstruction, make_model
from p3_ssl.bead_ssl_v2 import (
    Z8AsymmetricSyntheticDataset,
    Z8RealValidationDataset,
    load_bead_ssl_v2_config,
)


POLICIES = ("P25", "CYCLIC25")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_metrics_manifest(
    output: Path, run: dict[str, Any], result: dict[str, Any]
) -> None:
    provenance = {
        "datasets": sorted(
            str(record["id"]) for record in run.get("datasets", {}).values()
        ),
        "inputs": {
            "source_runs": list(run["source_runs"]),
            "counts": result["counts"],
            "sealed_splits_used": list(run.get("sealed_splits_used", [])),
        },
        "parameters": {
            "evaluation_policies": list(run["evaluation_policies"]),
            "checkpoint_policy": run["checkpoint_policy"],
            "cyclic_evaluation_schedule": run["cyclic_evaluation_schedule"],
        },
        "metric_definitions": {
            "masked_mse": "Mean squared error over samples hidden by the declared evaluation mask.",
            "masked_derivative_mse": "Mean squared first-difference error over adjacent hidden samples.",
            "event_support_masked_mse": "Masked MSE restricted to predeclared particle support.",
            "background_masked_mse": "Masked MSE outside predeclared particle support.",
        },
        "code": dict(run["source_sha256"]),
        "git_revision": dict(run["repositories"]),
    }
    manifest = {
        "schema_version": 1,
        "analysis_run_id": run["run_id"],
        "computation_provenance": provenance,
        "computation_fingerprint": computation_fingerprint(provenance),
        "metrics": [
            {"path": "metrics.json", "sha256": _sha256(output / "metrics.json")}
        ],
    }
    (output / "metrics_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _record(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    result = next(
        (row for row in records if f"{row['id']}@{row['version']}" == key), None
    )
    if result is None or result["status"] not in {"active", "reference"}:
        raise ValueError(f"eligible registered dataset not found: {key}")
    return result


def load_sources(runs_root: Path, run_ids: list[str]) -> list[tuple[Path, dict[str, Any]]]:
    if not run_ids:
        raise ValueError("at least one source run is required")
    sources: list[tuple[Path, dict[str, Any]]] = []
    seen: set[tuple[str, int]] = set()
    for run_id in run_ids:
        root = runs_root / run_id
        run = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if run.get("status") != "complete":
            raise ValueError(f"incomplete source run: {run_id}")
        policy = str(run.get("training_mask_policy"))
        seed = int(run.get("seed"))
        if policy not in POLICIES:
            raise ValueError(f"unexpected policy in {run_id}: {policy}")
        if run.get("profile") != "full":
            raise ValueError(f"cross-mask evaluation requires full runs: {run_id}")
        if (policy, seed) in seen:
            raise ValueError(f"duplicate policy/seed source: {policy}/{seed}")
        checkpoint = root / "checkpoints/latest.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        seen.add((policy, seed))
        sources.append((root, run))
    seeds = {seed for policy, seed in seen if policy == "P25"}
    cyclic_seeds = {seed for policy, seed in seen if policy == "CYCLIC25"}
    if seeds != cyclic_seeds:
        raise ValueError("P25 and CYCLIC25 source seeds must be paired")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-evaluate final bead SSL v2 models under P25 and CYCLIC25."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(Path(__file__).resolve().parents[1] / "configs/bead_ssl_z8_v5_v2.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id", action="append", default=[])
    parser.add_argument("--repair-metrics-manifest", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    workspace = Workspace.load()
    output = (
        workspace.artifacts_root
        / "unsupervised-learning-flow-cytometry/evaluations"
        / args.run_id
    )
    if args.repair_metrics_manifest:
        run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        if run.get("status") != "complete":
            raise ValueError("cannot repair manifest for an incomplete evaluation")
        _write_metrics_manifest(output, run, result)
        outputs = list(run.get("outputs", []))
        if "metrics_manifest.json" not in outputs:
            outputs.append("metrics_manifest.json")
        run["outputs"] = outputs
        (output / "run.json").write_text(
            json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    base_config = load_bead_ssl_v2_config(args.config)
    records = [record.payload for record in load_records(workspace)]
    study = base_config["study"]
    simulation_record = _record(records, study["simulation_dataset"])
    real_event_record = _record(records, study["real_event_dataset"])
    real_signal_record = _record(records, study["real_signal_dataset"])
    roots = {
        "simulation": workspace.datasets_root / simulation_record["path"],
        "real_events": workspace.datasets_root / real_event_record["path"],
        "real_signals": workspace.datasets_root / real_signal_record["path"],
    }
    runs_root = workspace.artifacts_root / "unsupervised-learning-flow-cytometry/runs"
    sources = load_sources(runs_root, args.source_run_id)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    output.mkdir(parents=True)

    dataset_records = {
        "simulation": simulation_record,
        "real_events": real_event_record,
        "real_signals": real_signal_record,
    }
    states = {
        "workspace": collect_git_state(workspace.root),
        "unsupervised-learning-flow-cytometry": collect_git_state(
            workspace.root / "unsupervised-learning-flow-cytometry"
        ),
    }
    run: dict[str, Any] = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "kind": "bead-ssl-v2-cross-mask-evaluation",
        "dataset": ",".join(
            f"{record['id']}@{record['version']}" for record in dataset_records.values()
        ),
        "datasets": {
            name: {
                "id": f"{record['id']}@{record['version']}",
                "manifest": record["manifest"],
                "manifest_sha256": record["manifest_sha256"],
            }
            for name, record in dataset_records.items()
        },
        "source_runs": args.source_run_id,
        "repositories": {name: state["revision"] for name, state in states.items()},
        "repository_dirty": {name: state["dirty"] for name, state in states.items()},
        "source_sha256": {
            "config": _sha256(args.config),
            "entrypoint": _sha256(Path(__file__).resolve()),
            "v2_module": _sha256(Path(__file__).resolve().parents[1] / "p3_ssl/bead_ssl_v2.py"),
        },
        "command": "evaluate_bead_ssl_v2_cross_masks.py "
        + " ".join(f"--source-run-id {value}" for value in args.source_run_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "evaluation_policies": list(POLICIES),
        "checkpoint_policy": "fixed_final",
        "cyclic_evaluation_schedule": "all_unique_passes_per_sample",
        "sealed_splits_used": [],
        "outputs": ["metrics.json", "metrics_manifest.json"],
    }
    (output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    profile = base_config["training"]["profiles"]["full"]
    simulation_dataset = Z8AsymmetricSyntheticDataset(
        roots["simulation"],
        split=base_config["data"]["simulation_validation_split"],
        normalization=base_config["data"]["normalization"],
        sampling_frequency_hz=float(base_config["data"]["sampling_frequency_hz"]),
    )
    real_dataset = Z8RealValidationDataset(
        roots["real_events"],
        roots["real_signals"],
        split=base_config["data"]["real_validation_split"],
        normalization=base_config["data"]["normalization"],
    )
    loaders = {
        "simulation_validation": DataLoader(
            simulation_dataset,
            batch_size=int(profile["batch_size"]),
            shuffle=False,
            num_workers=int(profile["num_workers"]),
        ),
        "real_validation": DataLoader(
            real_dataset,
            batch_size=int(profile["batch_size"]),
            shuffle=False,
            num_workers=int(profile["num_workers"]),
        ),
    }
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    try:
        for source, source_run in sources:
            training_policy = str(source_run["training_mask_policy"])
            seed = int(source_run["seed"])
            config = configure_experiment(
                base_config, loss_cell="B0", mask_policy=training_policy, seed=seed
            )
            checkpoint = torch.load(
                source / "checkpoints/latest.pt",
                map_location=device,
                weights_only=False,
            )
            model = make_model(config).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            for evaluation_policy in POLICIES:
                domain_metrics: dict[str, Any] = {}
                for domain, loader in loaders.items():
                    domain_metrics[domain], _ = evaluate_reconstruction(
                        model,
                        loader,
                        config,
                        device,
                        mask_seed=seed,
                        evaluation_policy=evaluation_policy,
                        max_examples=0,
                        include_regions=True,
                    )
                rows.append(
                    {
                        "source_run_id": source.name,
                        "training_mask_policy": training_policy,
                        "evaluation_mask_policy": evaluation_policy,
                        "checkpoint_policy": "fixed_final",
                        "checkpoint_epoch": int(checkpoint["epoch"]),
                        "seed": seed,
                        **domain_metrics,
                    }
                )
        result = {
            "protocol": "bead-ssl-cross-mask-evaluation-v2",
            "counts": {
                "simulation_validation": len(simulation_dataset),
                "real_validation": len(real_dataset),
                "rows": len(rows),
            },
            "checkpoint_policy": "fixed_final",
            "cyclic_evaluation_schedule": "all_unique_passes_per_sample",
            "real_cyclic_support": "Z8-v2 physical event bounds",
            "rows": rows,
        }
        (output / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_metrics_manifest(output, run, result)
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = f"{type(exc).__name__}: {exc}"
        (output / "run.json").write_text(
            json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    run["status"] = "complete"
    (output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
