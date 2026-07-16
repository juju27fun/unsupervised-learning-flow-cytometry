#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p3_ssl.predictability import (
    harmonic_regression_prediction,
    interpolation_prediction,
    masked_mse_numpy,
    nearest_prediction,
    sample_region_block_mask,
    visible_mean_prediction,
)
from p3_ssl.study_data import RealEventDataset, SimulatedLatentDataset


BASELINES = ("visible_mean", "interpolation", "nearest", "harmonic")


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


def _parse_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("Expected a comma-separated list of positive integers")
    return values


def _load_real(root: Path, maximum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = RealEventDataset(root, "development_validation", max_events=maximum)
    signals = []
    events = []
    snr = []
    for index in range(len(dataset)):
        item = dataset[index]
        signals.append(item["signal"][0].numpy())
        events.append(item["event_mask"].numpy())
        snr.append(float(dataset.rows[index]["snr_proxy"]))
    return np.stack(signals), np.stack(events), np.asarray(snr)


def _load_simulation(root: Path, maximum: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = SimulatedLatentDataset(root, "validation", max_latents=(maximum + 1) // 2)
    signals = []
    events = []
    snr = []
    for latent_index in range(len(dataset)):
        item = dataset[latent_index]
        for view_index, row in enumerate(dataset.latent_rows[latent_index]):
            signals.append(item["signals"][view_index, 0].numpy())
            events.append(item["event_masks"][view_index].numpy())
            snr.append(float(row["snr_db"]))
            if len(signals) == maximum:
                return np.stack(signals), np.stack(events), np.asarray(snr)
    return np.stack(signals), np.stack(events), np.asarray(snr)


def _snr_labels(values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    low, high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    labels = np.where(values <= low, "low", np.where(values <= high, "medium", "high"))
    return labels, [float(low), float(high)]


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        for stratum in ("all", str(row["snr_stratum"])):
            groups[(str(row["domain"]), str(row["mode"]), int(row["block_length"]), stratum)].append(row)
    summaries = []
    for (domain, mode, block_length, stratum), group in sorted(groups.items()):
        zero = float(np.mean([float(row["zero_mse"]) for row in group]))
        summary: dict[str, object] = {
            "domain": domain,
            "mode": mode,
            "block_length": block_length,
            "duration_us": block_length,
            "snr_stratum": stratum,
            "n_signals": len(group),
            "mean_target_event_fraction": float(
                np.mean([float(row["target_event_fraction"]) for row in group])
            ),
            "zero_mse": zero,
        }
        for baseline in BASELINES:
            mse = float(np.mean([float(row[f"{baseline}_mse"]) for row in group]))
            summary[f"{baseline}_mse"] = mse
            summary[f"{baseline}_relative_improvement_vs_zero"] = (
                (zero - mse) / zero if zero > 0.0 else None
            )
            summary[f"{baseline}_win_fraction_vs_zero"] = float(
                np.mean(
                    [
                        float(row[f"{baseline}_mse"]) < float(row["zero_mse"])
                        for row in group
                    ]
                )
            )
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, summaries: list[dict[str, object]]) -> None:
    domains = ("real", "simulation")
    modes = ("random", "event", "background")
    colors = {
        "visible_mean": "#8c8c8c",
        "interpolation": "#0077b6",
        "nearest": "#d1495b",
        "harmonic": "#2a9d8f",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for row_index, domain in enumerate(domains):
        for column_index, mode in enumerate(modes):
            axis = axes[row_index, column_index]
            selected = [
                row
                for row in summaries
                if row["domain"] == domain
                and row["mode"] == mode
                and row["snr_stratum"] == "all"
            ]
            for baseline in BASELINES:
                axis.plot(
                    [int(row["block_length"]) for row in selected],
                    [float(row[f"{baseline}_relative_improvement_vs_zero"]) for row in selected],
                    marker="o",
                    linewidth=1.5,
                    color=colors[baseline],
                    label=baseline.replace("_", " "),
                )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xscale("log", base=2)
            axis.grid(alpha=0.2)
            axis.set_title(f"{domain} / {mode}")
            if row_index == 1:
                axis.set_xlabel("contiguous mask duration (us)")
            if column_index == 0:
                axis.set_ylabel("relative MSE improvement vs zero")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(BASELINES), frameon=False)
    fig.suptitle("Yeast mask predictability map", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark predictable yeast mask regimes.")
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--simulation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-signals", type=int, default=64)
    parser.add_argument("--block-lengths", default="8,16,32,64,128,256,512")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")
    if args.max_signals <= 0:
        raise ValueError("--max-signals must be positive")
    block_lengths = _parse_ints(args.block_lengths)

    domains = {
        "real": _load_real(args.real_root, args.max_signals),
        "simulation": _load_simulation(args.simulation_root, args.max_signals),
    }
    rows: list[dict[str, object]] = []
    snr_thresholds = {}
    for domain_index, (domain, (signals, events, snr)) in enumerate(domains.items()):
        strata, thresholds = _snr_labels(snr)
        snr_thresholds[domain] = thresholds
        for mode_index, mode in enumerate(("random", "event", "background")):
            for length_index, block_length in enumerate(block_lengths):
                for signal_index, (signal, event) in enumerate(zip(signals, events)):
                    rng = np.random.default_rng(
                        args.seed
                        + domain_index * 10_000_019
                        + mode_index * 1_000_003
                        + length_index * 100_003
                        + signal_index
                    )
                    target = sample_region_block_mask(event, block_length, mode, rng)
                    predictions = {
                        "visible_mean": visible_mean_prediction(signal, target),
                        "interpolation": interpolation_prediction(signal, target),
                        "nearest": nearest_prediction(signal, target),
                        "harmonic": harmonic_regression_prediction(
                            signal, target, sampling_frequency_hz=1_000_000.0
                        ),
                    }
                    row: dict[str, object] = {
                        "domain": domain,
                        "signal_index": signal_index,
                        "mode": mode,
                        "block_length": block_length,
                        "snr": float(snr[signal_index]),
                        "snr_stratum": str(strata[signal_index]),
                        "target_event_fraction": float(np.mean(event[target])),
                        "zero_mse": masked_mse_numpy(np.zeros_like(signal), signal, target),
                    }
                    for baseline, prediction in predictions.items():
                        row[f"{baseline}_mse"] = masked_mse_numpy(prediction, signal, target)
                    rows.append(row)

    summaries = _summarize(rows)
    args.output_dir.mkdir(parents=True)
    observations_path = args.output_dir / "observations.csv"
    summary_path = args.output_dir / "summary.csv"
    figure_path = args.output_dir / "predictability_map.png"
    _write_csv(observations_path, rows)
    _write_csv(summary_path, summaries)
    _plot(figure_path, summaries)
    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "datasets": {
            "real": "yeast-events-representation@v3",
            "simulation": "yeast-passage-simulations@v1",
        },
        "splits": {"real": "development_validation", "simulation": "validation"},
        "sealed_splits_used": [],
        "n_signals_per_domain": {name: int(values[0].shape[0]) for name, values in domains.items()},
        "block_lengths": block_lengths,
        "snr_tertile_thresholds": snr_thresholds,
        "summary": summaries,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    outputs = [observations_path, summary_path, figure_path, metrics_path]
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": "yeast-events-representation@v3 + yeast-passage-simulations@v1",
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(workspace_root / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "sealed_splits_used": [],
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
