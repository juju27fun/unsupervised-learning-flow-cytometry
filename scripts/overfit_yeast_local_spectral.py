#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from p3_ssl.config import (
    load_config,
    validate_local_spectral_study_config,
    validate_mask_ablation_config,
    validate_study_config,
)
from p3_ssl.local_spectral_target import LocalSpectralTargetConfig, local_spectral_target
from p3_ssl.local_spectral_training import run_local_spectral_fixed_overfit
from p3_ssl.study_data import RealEventDataset, validate_real_event_dataset_contract
from p3_ssl.study_model import YeastStudyModel
from p3_ssl.study_training import model_config_from_study, seed_everything


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(str(tuple(values.shape)).encode("ascii"))
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def _code_tree_is_clean(repo_root: Path) -> bool:
    paths = (
        "p3_ssl",
        "scripts/overfit_yeast_local_spectral.py",
        "configs/yeast_ssl_local_spectral_v1.yaml",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return not status.strip()


def _target_config(config: dict[str, Any]) -> LocalSpectralTargetConfig:
    target = config["target"]
    return LocalSpectralTargetConfig(
        input_length=int(target["input_length"]),
        sampling_frequency_hz=float(target["sampling_frequency_hz"]),
        patch_size=int(target["patch_size"]),
        window_samples=int(target["window_samples"]),
        first_frequency_bin=int(target["first_frequency_bin"]),
        stop_frequency_bin=int(target["stop_frequency_bin"]),
        first_valid_token=int(target["first_valid_token"]),
        stop_valid_token=int(target["stop_valid_token"]),
    )


def _validate_predictability(path: Path, config: dict[str, Any]) -> None:
    run_path = path / "run.json"
    metrics_path = path / "metrics.json"
    if (
        _sha256(run_path) != config["study"]["source_predictability_run_json_sha256"]
        or _sha256(metrics_path) != config["study"]["source_predictability_metrics_sha256"]
    ):
        raise ValueError("S1 predictability artifact checksum mismatch")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        run.get("status") != "complete"
        or run.get("sealed_splits_used") != []
        or metrics.get("decision") != "run_fixed_batch_overfit"
        or not all(metrics.get("gates", {}).values())
        or metrics.get("sealed_splits_used") != []
    ):
        raise ValueError("S1 predictability artifact does not authorize overfit")


def _batch(dataset: RealEventDataset, indices: range) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [dataset[index] for index in indices]
    return (
        torch.stack([row["signal"] for row in rows]),
        torch.stack([row["event_mask"] for row in rows]),
    )


def _plot(path: Path, results: dict[str, dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, result in results.items():
        ax.plot(
            [row["step"] for row in result["history"]],
            [row["masked_feature_mse"] for row in result["history"]],
            marker="o",
            label=name,
        )
        ax.axhline(
            result["strongest_baseline_masked_feature_mse"],
            linestyle="--",
            linewidth=0.8,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Fixed-mask local spectral MSE")
    ax.set_title("S1 implementation and capacity preflight")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit the S1 target on fixed real batches.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_local_spectral_v1.yaml")
    )
    parser.add_argument("--source-predictability", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    if not _code_tree_is_clean(repo_root):
        raise RuntimeError("S1 overfit requires committed code and config")
    config = load_config(args.config)
    validate_local_spectral_study_config(config)
    if args.dataset_manifest_sha256 != config["study"]["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the frozen S1 config")
    _validate_predictability(args.source_predictability, config)
    contract = validate_real_event_dataset_contract(args.real_root)
    if not contract["valid"]:
        raise ValueError(f"Real dataset contract failed: {contract['errors']}")

    base = load_config(repo_root / config["study"]["base_config"])
    ablation = load_config(repo_root / config["study"]["source_mask_ablation_config"])
    validate_mask_ablation_config(ablation)
    effective = copy.deepcopy(base)
    effective["study"]["protocol"] = config["study"]["protocol"]
    effective["training"]["seed"] = int(config["training"]["seed"])
    effective["masking"].update(ablation["policies"][config["study"]["source_mask_policy"]])
    validate_study_config(effective)
    target_config = _target_config(config)
    data = RealEventDataset(args.real_root, effective["data"]["real_train_split"], max_events=72)
    constant_signals, _ = _batch(data, range(64))
    train_constant = local_spectral_target(constant_signals, target_config).mean(dim=0)
    overfit_signals, overfit_events = _batch(data, range(64, 72))
    profile = config["training"]["preflight_profiles"]["overfit"]
    device = torch.device(args.device)
    seed = int(config["training"]["seed"])
    base_model_config = model_config_from_study(effective)
    treatment_model_config = replace(
        base_model_config, local_spectral_features=target_config.feature_count
    )
    seed_everything(seed)
    control = YeastStudyModel(base_model_config)
    control_encoder_sha = _state_sha256(control.reconstructor.encoder_state_dict())
    results = {}
    treatment_encoder_hashes = {}
    for count in (1, 8):
        seed_everything(seed)
        model = YeastStudyModel(treatment_model_config)
        treatment_encoder_hashes[str(count)] = _state_sha256(
            model.reconstructor.encoder_state_dict()
        )
        results[f"n{count}"] = run_local_spectral_fixed_overfit(
            model,
            overfit_signals[:count],
            overfit_events[:count],
            train_constant,
            effective,
            seed=seed,
            target_config=target_config,
            steps=int(profile["steps"]),
            learning_rate=float(profile["learning_rate"]),
            grad_clip_norm=float(effective["training"]["grad_clip_norm"]),
            log_every=int(profile["log_every"]),
            device=device,
        )
    encoder_initialization_matched = all(
        value == control_encoder_sha for value in treatment_encoder_hashes.values()
    )
    required_improvement = float(
        config["gates"]["preflight"]["fixed_one_and_eight_improvement_min"]
    )
    gates = {
        "encoder_initialization_matches_c1": encoder_initialization_matched,
        "one_example_improvement": results["n1"][
            "relative_improvement_vs_strongest_baseline"
        ]
        >= required_improvement,
        "eight_example_improvement": results["n8"][
            "relative_improvement_vs_strongest_baseline"
        ]
        >= required_improvement,
        "finite_nonzero_encoder_and_head_gradients": all(
            result["gates"]["finite_gradients"]
            and result["gates"]["nonzero_encoder_gradient"]
            and result["gates"]["nonzero_head_gradient"]
            for result in results.values()
        ),
        "nontrivial_amplitude": all(
            result["gates"]["nontrivial_amplitude_0p10"] for result in results.values()
        ),
    }
    decision = "run_s1_training_smoke" if all(gates.values()) else "reject_s1_implementation"
    args.output_dir.mkdir(parents=True)
    figure_path = args.output_dir / "local_spectral_fixed_overfit.png"
    _plot(figure_path, results)
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "protocol": config["study"]["protocol"],
        "dataset": config["study"]["dataset"],
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "device": str(device),
        "constant_pool_events": 64,
        "overfit_event_offset": 64,
        "control_encoder_initial_sha256": control_encoder_sha,
        "treatment_encoder_initial_sha256": treatment_encoder_hashes,
        "results": results,
        "gates": gates,
        "decision": decision,
        "sealed_splits_used": [],
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": config["study"]["dataset"],
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(repo_root.parent / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "profile": "fixed-batch-overfit",
        "config_sha256": _sha256(args.config),
        "source_predictability_run_json_sha256": _sha256(
            args.source_predictability / "run.json"
        ),
        "sealed_splits_used": [],
        "outputs": {
            metrics_path.name: _sha256(metrics_path),
            figure_path.name: _sha256(figure_path),
        },
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "gates": gates, "results": results}, indent=2))


if __name__ == "__main__":
    main()
