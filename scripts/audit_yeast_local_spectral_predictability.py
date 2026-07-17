#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from p3_ssl.config import (
    load_config,
    validate_local_spectral_study_config,
    validate_mask_ablation_config,
    validate_study_config,
)
from p3_ssl.local_spectral_target import (
    LocalSpectralTargetConfig,
    analytic_signal,
    local_spectral_frame_regions,
    local_spectral_frequencies,
    local_spectral_target,
)
from p3_ssl.study_data import RealEventDataset, validate_real_event_dataset_contract
from p3_ssl.study_training import build_mask_batch, interpolation_baseline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _code_tree_is_clean(repo_root: Path) -> bool:
    paths = (
        "p3_ssl",
        "scripts/audit_yeast_local_spectral_predictability.py",
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


def _validate_source_supplement(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    run_path = path / "run.json"
    metrics_path = path / "metrics.json"
    if (
        _sha256(run_path) != config["study"]["source_supplement_run_json_sha256"]
        or _sha256(metrics_path) != config["study"]["source_supplement_metrics_sha256"]
    ):
        raise ValueError("Source handcrafted supplement checksum mismatch")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        run.get("status") != "complete"
        or run.get("sealed_splits_used") != []
        or metrics.get("decision_audit", {}).get("decision")
        != "mask_only_rejection_confirmed_no_complementarity"
        or metrics.get("sealed_splits_used") != []
    ):
        raise ValueError("Source supplement does not authorize the terminal target amendment")
    return metrics


def _masked_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[float, int]:
    count = int(selected.sum()) * target.shape[-1]
    if count == 0:
        return 0.0, 0
    values = (prediction - target).square() * selected.to(target.dtype).unsqueeze(-1)
    return float(values.sum()), count


def _plot_example(
    path: Path,
    signal: torch.Tensor,
    target: torch.Tensor,
    interpolation_target: torch.Tensor,
    event_mask: torch.Tensor,
    frequencies: torch.Tensor,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    time_ms = np.arange(signal.numel()) / 1000.0
    axes[0].plot(time_ms, signal.numpy(), color="#1f2937", linewidth=0.8)
    active = np.flatnonzero(event_mask.numpy())
    if active.size:
        axes[0].axvspan(active[0] / 1000.0, active[-1] / 1000.0, color="#f59e0b", alpha=0.2)
    axes[0].set_ylabel("Normalized signal")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_title("Complete target source; orange is the reviewed event")
    extent = [0.128, 3.968, float(frequencies[0]) / 1000.0, float(frequencies[-1]) / 1000.0]
    axes[1].imshow(target.T.numpy(), aspect="auto", origin="lower", extent=extent, cmap="viridis")
    axes[1].set_ylabel("Frequency (kHz)")
    axes[1].set_title("Analytic local log-power target")
    difference = torch.abs(target - interpolation_target)
    axes[2].imshow(
        difference.T.numpy(), aspect="auto", origin="lower", extent=extent, cmap="magma"
    )
    axes[2].set_xlabel("Token-center time (ms)")
    axes[2].set_ylabel("Frequency (kHz)")
    axes[2].set_title("Absolute feature difference after waveform interpolation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit predictability of the single S1 target.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_local_spectral_v1.yaml")
    )
    parser.add_argument("--source-supplement", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    if not _code_tree_is_clean(repo_root):
        raise RuntimeError("S1 predictability audit requires committed code and config")
    config = load_config(args.config)
    validate_local_spectral_study_config(config)
    if args.dataset_manifest_sha256 != config["study"]["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the frozen S1 config")
    _validate_source_supplement(args.source_supplement, config)
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
    profile = config["training"]["preflight_profiles"]["predictability"]
    train_data = RealEventDataset(
        args.real_root,
        effective["data"]["real_train_split"],
        max_events=int(profile["max_train_events"]),
    )
    validation_data = RealEventDataset(
        args.real_root,
        effective["data"]["real_validation_split"],
        max_events=int(profile["max_validation_events"]),
    )
    batch_size = int(profile["batch_size"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    target_config = _target_config(config)

    train_sum = torch.zeros(target_config.valid_token_count, target_config.feature_count)
    train_count = 0
    for batch in train_loader:
        target = local_spectral_target(batch["signal"], target_config)
        train_sum += target.sum(dim=0)
        train_count += len(target)
    train_constant = train_sum / train_count

    sums: dict[str, float] = {name: 0.0 for name in ("zero", "train_constant", "interpolation")}
    counts: dict[str, int] = {name: 0 for name in sums}
    region_sums: dict[str, dict[str, float]] = {
        baseline: {region: 0.0 for region in ("event", "background", "boundary")}
        for baseline in sums
    }
    region_counts: dict[str, dict[str, int]] = {
        baseline: {region: 0 for region in ("event", "background", "boundary")}
        for baseline in sums
    }
    target_square_sum = 0.0
    selected_feature_count = 0
    interpolation_square_sum = 0.0
    mask_independent = True
    phase_numerator = 0.0
    phase_denominator = 0.0
    example: tuple[torch.Tensor, ...] | None = None
    seed = int(config["training"]["seed"])
    for batch_index, batch in enumerate(validation_loader):
        signals = batch["signal"]
        events = batch["event_mask"]
        target = local_spectral_target(signals, target_config)
        _, token_mask, hidden_mask = build_mask_batch(
            signals, events, effective, seed + batch_index * 100_003
        )
        mask_independent = mask_independent and torch.equal(
            target, local_spectral_target(signals, target_config)
        )
        valid_mask = token_mask[
            :, target_config.first_valid_token : target_config.stop_valid_token
        ]
        interpolation_signal = interpolation_baseline(signals, hidden_mask)
        interpolation_target = local_spectral_target(interpolation_signal, target_config)
        predictions = {
            "zero": torch.zeros_like(target),
            "train_constant": train_constant.unsqueeze(0).expand_as(target),
            "interpolation": interpolation_target,
        }
        regions = local_spectral_frame_regions(events, target_config)
        for name, prediction in predictions.items():
            value, count = _masked_sums(prediction, target, valid_mask)
            sums[name] += value
            counts[name] += count
            for region, region_mask in regions.items():
                value, count = _masked_sums(
                    prediction, target, valid_mask & region_mask
                )
                region_sums[name][region] += value
                region_counts[name][region] += count
        selected = valid_mask.unsqueeze(-1).expand_as(target)
        target_square_sum += float((target.square() * selected).sum())
        interpolation_square_sum += float((interpolation_target.square() * selected).sum())
        selected_feature_count += int(selected.sum())
        if batch_index == 0:
            phase_count = min(32, len(signals))
            analytic = analytic_signal(signals[:phase_count, 0])
            phase = torch.tensor(0.731, dtype=signals.dtype)
            rotated = (analytic * torch.exp(1j * phase)).real.unsqueeze(1)
            rotated_target = local_spectral_target(rotated, target_config)
            phase_numerator += float((target[:phase_count] - rotated_target).square().sum())
            phase_denominator += float(target[:phase_count].square().sum())
            example = (
                signals[0, 0].clone(),
                target[0].clone(),
                interpolation_target[0].clone(),
                events[0].clone(),
            )

    mse = {name: sums[name] / counts[name] for name in sums}
    region_mse = {
        name: {
            region: (
                region_sums[name][region] / region_counts[name][region]
                if region_counts[name][region]
                else None
            )
            for region in region_sums[name]
        }
        for name in region_sums
    }
    phase_error = phase_numerator / phase_denominator
    target_rms = float(np.sqrt(target_square_sum / selected_feature_count))
    interpolation_rms = float(np.sqrt(interpolation_square_sum / selected_feature_count))
    gates = {
        "finite_targets_and_baselines": all(np.isfinite(list(mse.values()))),
        "target_nonconstant": mse["train_constant"] > 0.0,
        "phase_rotation_relative_error": phase_error
        <= float(config["gates"]["preflight"]["phase_rotation_relative_error_max"]),
        "mask_target_independence": mask_independent,
    }
    decision = "run_fixed_batch_overfit" if all(gates.values()) else "reject_s1_target_preflight"
    args.output_dir.mkdir(parents=True)
    figure_path = args.output_dir / "local_spectral_target_example.png"
    if example is None:
        raise RuntimeError("No validation example was available")
    _plot_example(
        figure_path,
        *example,
        local_spectral_frequencies(target_config),
    )
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "protocol": config["study"]["protocol"],
        "dataset": config["study"]["dataset"],
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "n_train_events": len(train_data),
        "n_validation_events": len(validation_data),
        "mask_policy": config["study"]["source_mask_policy"],
        "target": config["target"],
        "frequencies_hz": local_spectral_frequencies(target_config).tolist(),
        "phase_rotation_relative_error": phase_error,
        "mask_target_independent": mask_independent,
        "baseline_masked_mse": mse,
        "baseline_region_masked_mse": region_mse,
        "target_rms_on_mask": target_rms,
        "interpolation_target_rms_on_mask": interpolation_rms,
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
        "profile": "target-predictability-preflight",
        "config_sha256": _sha256(args.config),
        "source_supplement_run_json_sha256": _sha256(args.source_supplement / "run.json"),
        "sealed_splits_used": [],
        "outputs": {
            metrics_path.name: _sha256(metrics_path),
            figure_path.name: _sha256(figure_path),
        },
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "gates": gates, "mse": mse}, indent=2))


if __name__ == "__main__":
    main()
