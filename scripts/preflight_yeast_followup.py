#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from p3_ssl.followup_features import load_followup_development
from p3_ssl.followup_objectives import FollowupObjectiveConfig, followup_ssl_objective
from p3_ssl.study_model import YeastStudyModel, YeastStudyModelConfig


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, text=True, capture_output=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU preflight for frozen yeast R0-R3 cells.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if set(config["data"]["forbidden_training_splits"]) != {
        "in_session_test", "sealed_acquisition_test", "followup_test", "test"
    }:
        raise ValueError("Frozen final-split guard is incomplete")
    data = load_followup_development(args.dataset_root)
    train = data.train_indices[:4]
    target = torch.from_numpy(np.asarray(data.signals[train], dtype=np.float32)).unsqueeze(1)
    device = torch.device(args.device)
    target = target.to(device)
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[..., 1024:2048] = True
    masked_input = target.masked_fill(mask, 0.0)
    augmented = torch.roll(target, shifts=8, dims=-1) + 0.01 * torch.randn_like(target)

    model_cfg = YeastStudyModelConfig(
        input_length=config["data"]["input_length"],
        patch_size=config["model"]["patch_size"],
        patch_stride=config["model"]["patch_stride"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        activation=config["model"]["activation"],
        max_tokens=config["model"]["max_tokens"],
    )
    spectral = config["objectives"]["spectral"]
    vicreg = config["objectives"]["vicreg"]
    objective_cfg = FollowupObjectiveConfig(
        spectral_windows=tuple(spectral["windows_samples"]),
        spectral_hop_divisor=4,
        spectral_center=bool(spectral.get("center", False)),
        spectral_epsilon=float(spectral["epsilon"]),
        time_weight=float(config["objectives"]["time_weight"]),
        spectral_weight=float(spectral["weight"]),
        vicreg_weight=float(vicreg["global_weight"]),
        invariance_weight=float(vicreg["invariance_weight"]),
        variance_weight=float(vicreg["variance_weight"]),
        covariance_weight=float(vicreg["covariance_weight"]),
        variance_floor=float(vicreg["variance_floor"]),
        variance_epsilon=float(vicreg["epsilon"]),
    )
    torch.manual_seed(42)
    results = {}
    for cell in ("R0", "R1", "R2", "R3"):
        model = YeastStudyModel(model_cfg).to(device).train()
        output = model(masked_input)
        paired = None
        if cell in {"R1", "R3"}:
            second = model(augmented)
            paired = torch.stack([output["embedding"], second["embedding"]], dim=1)
        loss, terms = followup_ssl_objective(
            cell,
            output["reconstruction"],
            target,
            mask,
            paired_embeddings=paired,
            config=objective_cfg,
        )
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        finite = bool(torch.isfinite(loss) and all(torch.isfinite(value).all() for value in gradients))
        results[cell] = {
            "loss": float(loss.detach().cpu()),
            "terms": {name: float(value.detach().cpu()) for name, value in terms.items()},
            "finite_gradients": finite,
            "n_parameters_with_gradients": len(gradients),
        }
    status = "pass" if all(value["finite_gradients"] for value in results.values()) else "fail"
    payload = {
        "schema_version": 1,
        "status": status,
        "device": str(device),
        "dataset": "yeast-events-followup@v2",
        "splits_loaded": ["followup_train", "followup_validation"],
        "sealed_splits_used": [],
        "config_sha256": _sha256(args.config),
        "cells": results,
    }
    metrics = args.output_dir / "preflight_metrics.json"
    metrics.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol = {
        "protocol": config["study"]["protocol"],
        "config_sha256": _sha256(args.config),
        "cells": config["objectives"]["cells"],
        "datasets": {
            "real": config["study"]["real_dataset"],
            "simulation": config["study"]["simulation_dataset"],
        },
        "forbidden_training_splits": config["data"]["forbidden_training_splits"],
        "training": config["training"],
        "evaluation": config["evaluation"],
    }
    protocol_path = args.output_dir / "frozen_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {
        "schema_version": 1,
        "run_id": args.run_id,
        "project": "unsupervised-learning-flow-cytometry",
        "status": "complete" if status == "pass" else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "preflight_yeast_followup.py",
        "dataset": "yeast-events-followup@v2 + yeast-passage-simulations@v2",
        "profile": "week1-cpu-preflight",
        "sealed_splits_used": [],
        "config_sha256": _sha256(args.config),
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(Path(__file__).resolve().parents[1]),
            "particles2SNR-pipeline": _revision(Path(__file__).resolve().parents[2] / "particles2SNR-pipeline"),
        },
        "outputs": {
            metrics.name: _sha256(metrics),
            protocol_path.name: _sha256(protocol_path),
        },
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
