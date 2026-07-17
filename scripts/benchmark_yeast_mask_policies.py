#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p3_ssl.config import load_config, validate_mask_ablation_config, validate_study_config
from p3_ssl.masking import (
    PatchSpec,
    build_patch_aligned_isolated_masks,
    build_ssl_masks,
    mask_coherence_summary,
)
from p3_ssl.predictability import (
    autoregressive_prediction,
    harmonic_regression_prediction,
    interpolation_prediction,
    masked_mse_numpy,
    nearest_prediction,
    visible_mean_prediction,
)
from p3_ssl.study_data import RealEventDataset


BASELINES = ("visible_mean", "interpolation", "nearest", "autoregressive", "harmonic")


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, object]], policies: list[str]) -> list[dict[str, object]]:
    output = []
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        zero = float(np.mean([float(row["zero_mse"]) for row in selected]))
        summary: dict[str, object] = {
            "policy": policy,
            "n_signals": len(selected),
            "zero_mse": zero,
            "mean_target_ratio": float(np.mean([float(row["target_ratio"]) for row in selected])),
            "mean_hidden_ratio": float(np.mean([float(row["hidden_ratio"]) for row in selected])),
            "mean_target_event_fraction": float(
                np.mean([float(row["target_event_fraction"]) for row in selected])
            ),
            "mask_acceptance_rate": float(
                np.mean([bool(row["mask_accepted"]) for row in selected])
            ),
        }
        for baseline in BASELINES:
            mse = float(np.mean([float(row[f"{baseline}_mse"]) for row in selected]))
            summary[f"{baseline}_mse"] = mse
            summary[f"{baseline}_relative_improvement_vs_zero"] = (zero - mse) / zero
            summary[f"{baseline}_win_fraction_vs_zero"] = float(
                np.mean(
                    [
                        float(row[f"{baseline}_mse"]) < float(row["zero_mse"])
                        for row in selected
                    ]
                )
            )
        output.append(summary)
    return output


def _plot(path: Path, summaries: list[dict[str, object]]) -> None:
    policies = [str(row["policy"]) for row in summaries]
    positions = np.arange(len(policies))
    width = 0.16
    colors = ("#8c8c8c", "#0077b6", "#d1495b", "#e9c46a", "#2a9d8f")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, (baseline, color) in enumerate(zip(BASELINES, colors)):
        axes[0].bar(
            positions + (index - (len(BASELINES) - 1) / 2.0) * width,
            [float(row[f"{baseline}_relative_improvement_vs_zero"]) for row in summaries],
            width,
            color=color,
            label=baseline.replace("_", " "),
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(positions, policies)
    axes[0].set_ylabel("relative MSE improvement vs zero")
    axes[0].set_title("Predictability under complete mask policy")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(
        positions - width / 2,
        [float(row["mean_target_ratio"]) for row in summaries],
        width,
        label="loss target",
        color="#0077b6",
    )
    axes[1].bar(
        positions + width / 2,
        [float(row["mean_hidden_ratio"]) for row in summaries],
        width,
        label="patch-aligned hidden context",
        color="#d1495b",
    )
    axes[1].set_xticks(positions, policies)
    axes[1].set_ylabel("fraction of trace")
    axes[1].set_title("Requested target versus context actually hidden")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark complete yeast masking policies.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_mask_ablation_v1.yaml")
    )
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-signals", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")
    if args.max_signals <= 0:
        raise ValueError("--max-signals must be positive")

    ablation = load_config(args.config)
    validate_mask_ablation_config(ablation)
    base_path = Path(ablation["study"]["base_config"])
    base = load_config(base_path)
    dataset = RealEventDataset(
        args.real_root, base["data"]["real_validation_split"], max_events=args.max_signals
    )
    samples_per_ms = float(base["data"]["sampling_frequency_hz"]) / 1000.0
    spec = PatchSpec(
        input_length=int(base["data"]["input_length"]),
        patch_size=int(base["model"]["patch_size"]),
        patch_stride=int(base["model"]["patch_stride"]),
    )
    rows: list[dict[str, object]] = []
    policy_names = list(ablation["policies"])
    for policy_index, policy_name in enumerate(policy_names):
        policy = ablation["policies"][policy_name]
        policy_config = copy.deepcopy(base)
        policy_config["masking"].update(policy)
        validate_study_config(policy_config)
        for signal_index in range(len(dataset)):
            item = dataset[signal_index]
            signal = item["signal"][0].numpy()
            event = item["event_mask"].numpy()
            common = {
                "signal": signal,
                "spec": spec,
                "rng": np.random.default_rng(
                    args.seed + policy_index * 1_000_003 + signal_index
                ),
                "mask_ratio": float(policy["mask_ratio"]),
                "high_derivative_probability": float(policy["high_derivative_probability"]),
                "event_mask": event,
                "event_biased_probability": float(policy["event_biased_probability"]),
                "avoid_fully_hidden_events": bool(base["masking"]["avoid_fully_hidden_events"]),
                "max_event_hidden_fraction": float(base["masking"]["max_event_hidden_fraction"]),
                "max_mask_attempts": int(base["masking"]["max_mask_attempts"]),
            }
            if policy["strategy"] == "patch_aligned_isolated":
                result = build_patch_aligned_isolated_masks(
                    **common,
                    minimum_visible_tokens_between_masks=int(
                        policy["minimum_visible_tokens_between_masks"]
                    ),
                )
            else:
                result = build_ssl_masks(
                    **common,
                    min_block_length=int(
                        round(float(policy["min_block_ms"]) * samples_per_ms)
                    ),
                    max_block_length=int(
                        round(float(policy["max_block_ms"]) * samples_per_ms)
                    ),
                    guard_points=int(round(float(policy["guard_ms"]) * samples_per_ms)),
                )
            target = result["target_time_mask"]
            hidden = result["token_time_mask"]
            coherence = mask_coherence_summary(target, hidden, event)
            predictions = {
                "visible_mean": visible_mean_prediction(signal, hidden),
                "interpolation": interpolation_prediction(signal, hidden),
                "nearest": nearest_prediction(signal, hidden),
                "autoregressive": autoregressive_prediction(signal, hidden),
                "harmonic": harmonic_regression_prediction(
                    signal, hidden, sampling_frequency_hz=1_000_000.0
                ),
            }
            row: dict[str, object] = {
                "policy": policy_name,
                "signal_index": signal_index,
                "snr_proxy": float(dataset.rows[signal_index]["snr_proxy"]),
                "target_ratio": float(target.mean()),
                "hidden_ratio": float(hidden.mean()),
                "target_event_fraction": float(coherence["target_event_fraction"]),
                "mask_accepted": bool(result["mask_accepted"]),
                "zero_mse": masked_mse_numpy(np.zeros_like(signal), signal, target),
            }
            for baseline, prediction in predictions.items():
                row[f"{baseline}_mse"] = masked_mse_numpy(prediction, signal, target)
            rows.append(row)

    summaries = _summarize(rows, policy_names)
    args.output_dir.mkdir(parents=True)
    observations_path = args.output_dir / "observations.csv"
    summary_path = args.output_dir / "summary.csv"
    figure_path = args.output_dir / "mask_policy_predictability.png"
    metrics_path = args.output_dir / "metrics.json"
    _write_csv(observations_path, rows)
    _write_csv(summary_path, summaries)
    _plot(figure_path, summaries)
    metrics = {
        "schema_version": 1,
        "run_id": args.run_id,
        "dataset": ablation["study"]["real_dataset"],
        "split": base["data"]["real_validation_split"],
        "sealed_splits_used": [],
        "n_signals": len(dataset),
        "policies": ablation["policies"],
        "summary": summaries,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outputs = [observations_path, summary_path, figure_path, metrics_path]
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": ablation["study"]["real_dataset"],
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(workspace_root / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "config_sha256": _sha256(args.config),
        "sealed_splits_used": [],
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
