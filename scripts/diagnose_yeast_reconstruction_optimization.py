#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
import torch
from torch.utils.data import DataLoader

from p3_ssl.config import load_config
from p3_ssl.masking import mask_spans
from p3_ssl.reconstruction_diagnostics import run_fixed_mask_overfit
from p3_ssl.study_data import RealEventDataset, SEALED_REAL_SPLITS
from p3_ssl.study_model import YeastStudyModel
from p3_ssl.study_training import model_config_from_study


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


def _plot_prediction(
    path: Path,
    signal: torch.Tensor,
    prediction: torch.Tensor,
    target_mask: torch.Tensor,
    event_mask: torch.Tensor,
    title: str,
) -> None:
    x = signal[0, 0].numpy()
    pred = prediction[0, 0].numpy()
    target = target_mask[0].numpy()
    event = event_mask[0].numpy()
    index = np.arange(x.size)
    masked_prediction = np.where(target, pred, np.nan)
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(index, x, color="#4b5563", linewidth=0.8, label="target signal")
    axis.plot(
        index,
        masked_prediction,
        color="#d1495b",
        linewidth=1.1,
        label="prediction on loss mask",
    )
    for start, end in mask_spans(event):
        axis.axvspan(start, end, color="#4c956c", alpha=0.08)
    for start, end in mask_spans(target):
        axis.axvspan(start, end, color="#d1495b", alpha=0.10)
    axis.set(title=title, xlabel="sample", ylabel="normalized amplitude")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prove whether the yeast reconstructor can memorize fixed real masks."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_rebuild_v1.yaml")
    )
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="development_train")
    parser.add_argument("--sample-count", action="append", type=int)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()

    if args.split in SEALED_REAL_SPLITS:
        raise ValueError(f"Refusing sealed split for optimization diagnostic: {args.split}")
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")
    sample_counts = args.sample_count or [1, 8]
    if any(count <= 0 for count in sample_counts):
        raise ValueError("--sample-count must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)

    config = load_config(args.config)
    device = torch.device(args.device)
    maximum = max(sample_counts)
    RealEventDataset(args.real_root, args.split, max_events=maximum)
    cases: dict[str, dict[str, object]] = {}
    plot_payloads = []

    for case_index, count in enumerate(sample_counts):
        subset = RealEventDataset(args.real_root, args.split, max_events=count)
        batch = next(iter(DataLoader(subset, batch_size=count, shuffle=False)))
        case_seed = args.seed + case_index * 10_003
        torch.manual_seed(case_seed)
        model = YeastStudyModel(model_config_from_study(config))
        result, prediction, target_mask = run_fixed_mask_overfit(
            model,
            batch["signal"],
            batch["event_mask"],
            config,
            seed=case_seed,
            steps=args.steps,
            learning_rate=args.learning_rate,
            device=device,
        )
        case_name = f"fixed_{count}"
        cases[case_name] = result
        plot_payloads.append(
            (case_name, batch["signal"], prediction, target_mask, batch["event_mask"])
        )

    args.output_dir.mkdir(parents=True)
    output_paths = []
    for case_name, signal, prediction, target_mask, event_mask in plot_payloads:
        plot_path = args.output_dir / f"{case_name}_prediction.png"
        _plot_prediction(
            plot_path,
            signal,
            prediction,
            target_mask,
            event_mask,
            f"Fixed-mask optimization diagnostic: {case_name}",
        )
        output_paths.append(plot_path)

    result = {
        "schema_version": 1,
        "run_id": args.run_id,
        "dataset": config["study"]["real_dataset"],
        "split": args.split,
        "sealed_splits_used": [],
        "cases": cases,
        "all_cases_pass": all(
            all(bool(value) for value in dict(case["gates"]).values())
            for case in cases.values()
        ),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_paths.append(metrics_path)

    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": config["study"]["real_dataset"],
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(workspace_root / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "config_sha256": _sha256(args.config),
        "sealed_splits_used": [],
        "outputs": {path.name: _sha256(path) for path in output_paths},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
