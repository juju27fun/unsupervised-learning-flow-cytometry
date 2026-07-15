#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from p3_ssl.config import load_config
from p3_ssl.final_evaluation import (
    FINAL_SPLIT,
    LABEL_FRACTION,
    MINIMUM_EFFECT,
    paired_comparison,
    prior_final_open,
)
from p3_ssl.study_baselines import (
    checkpoint_encoder_features,
    fit_linear_probe,
    handcrafted_features,
    load_baseline_data,
    prediction_metrics,
)
from p3_ssl.study_evaluation import calibration_metrics, grouped_bootstrap_metrics


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name, Path(raw_path)


def _evaluate(
    *,
    method: str,
    representation_seed: int | None,
    features: np.ndarray,
    data: Any,
    test_indices: np.ndarray,
    probe_seeds: list[int],
    bootstrap_repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows = []
    prediction_rows = []
    test_labels = data.labels[test_indices]
    test_groups = np.asarray(
        [data.rows[int(index)]["capture_block_id"] for index in test_indices], dtype=str
    )
    for probe_seed in probe_seeds:
        model, train = fit_linear_probe(
            features, data, fraction=LABEL_FRACTION, seed=probe_seed
        )
        probabilities = model.predict_proba(features[test_indices])
        predictions = probabilities.argmax(axis=1)
        metrics = {
            **prediction_metrics(test_labels, predictions, data.class_names),
            "calibration": calibration_metrics(test_labels, probabilities),
            "grouped_bootstrap": grouped_bootstrap_metrics(
                test_labels,
                predictions,
                probabilities,
                test_groups,
                class_count=len(data.class_names),
                repeats=bootstrap_repeats,
                seed=probe_seed,
            ),
            "probe_optimization": model.probe_optimization_,
            "n_probe_events": int(train.size),
            "n_probe_records": len(
                {data.rows[int(index)]["record_id"] for index in train}
            ),
        }
        metric_rows.append(
            {
                "method": method,
                "representation_seed": representation_seed,
                "probe_seed": probe_seed,
                "label_fraction": LABEL_FRACTION,
                **metrics,
            }
        )
        for local_index, row_index in enumerate(test_indices):
            source = data.rows[int(row_index)]
            prediction_rows.append(
                {
                    "method": method,
                    "representation_seed": "" if representation_seed is None else representation_seed,
                    "probe_seed": probe_seed,
                    "signal_row": int(row_index),
                    "event_id": source["event_id"],
                    "record_id": source["record_id"],
                    "capture_block_id": source["capture_block_id"],
                    "true_proxy": data.class_names[int(test_labels[local_index])],
                    "predicted_proxy": data.class_names[int(predictions[local_index])],
                    **{
                        f"probability_{name}": float(probabilities[local_index, class_id])
                        for class_id, name in enumerate(data.class_names)
                    },
                }
            )
    return metric_rows, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the frozen yeast in-session test once.")
    parser.add_argument("--checkpoint", action="append", type=_checkpoint, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/yeast_ssl_rebuild_v1.yaml"))
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--development-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")
    prior = prior_final_open(args.output_dir.parent)
    if prior is not None:
        raise SystemExit(f"The frozen {FINAL_SPLIT} was already opened by {prior}")

    config = load_config(args.config)
    profile = config["baselines"]["profiles"]["full"]
    probe_seeds = [int(seed) for seed in profile["probe_seeds"]]
    bootstrap_repeats = int(config["evaluation"]["profiles"]["full"]["grouped_bootstrap_repeats"])
    data = load_baseline_data(args.real_root, seed=int(config["training"]["seed"]))
    split = np.asarray([row["development_split"] for row in data.rows])
    test_indices = np.flatnonzero(split == FINAL_SPLIT)
    if not test_indices.size or set(data.labels[test_indices]) != set(range(len(data.class_names))):
        raise SystemExit("The final split is missing or does not contain every proxy class")
    train_blocks = {data.rows[int(index)]["capture_block_id"] for index in data.train_indices}
    validation_blocks = {
        data.rows[int(index)]["capture_block_id"] for index in data.validation_indices
    }
    test_blocks = {data.rows[int(index)]["capture_block_id"] for index in test_indices}
    if test_blocks & (train_blocks | validation_blocks):
        raise SystemExit("A capture block crosses the final-test boundary")

    development_decision = json.loads(args.development_decision.read_text(encoding="utf-8"))
    if development_decision["promotion_decision"] != "do_not_promote_a4":
        raise SystemExit("Unexpected development decision; final protocol is not frozen")
    checkpoints = args.checkpoint
    expected_checkpoints = {
        f"{cell}_s{seed}" for cell in ("a3", "a4") for seed in (42, 43, 44)
    }
    if {name for name, _ in checkpoints} != expected_checkpoints or len(checkpoints) != 6:
        raise SystemExit("Final evaluation requires exactly three A3 and three A4 checkpoints")

    selected_indices = np.concatenate([data.train_indices, test_indices])
    selected_signals = np.asarray(data.signals[selected_indices], dtype=np.float32)
    selected_handcrafted = handcrafted_features(selected_signals)
    features = np.empty(
        (len(data.rows), selected_handcrafted.shape[1]), dtype=np.float32
    )
    features[selected_indices] = selected_handcrafted
    all_metrics, all_predictions = _evaluate(
        method="handcrafted",
        representation_seed=None,
        features=features,
        data=data,
        test_indices=test_indices,
        probe_seeds=probe_seeds,
        bootstrap_repeats=bootstrap_repeats,
    )

    device = torch.device(args.device)
    checkpoint_hashes = {}
    for name, checkpoint_path in checkpoints:
        embeddings, metadata = checkpoint_encoder_features(
            selected_signals,
            checkpoint_path,
            batch_size=int(profile["batch_size"]),
            device=device,
        )
        expected_cell = name.split("_", 1)[0].upper()
        if metadata["cell"] != expected_cell:
            raise SystemExit(f"Checkpoint {name} declares {metadata['cell']}, expected {expected_cell}")
        checkpoint_features = np.empty((len(data.rows), embeddings.shape[1]), dtype=np.float32)
        checkpoint_features[selected_indices] = embeddings
        metric_rows, prediction_rows = _evaluate(
            method=metadata["cell"],
            representation_seed=int(metadata["seed"]),
            features=checkpoint_features,
            data=data,
            test_indices=test_indices,
            probe_seeds=probe_seeds,
            bootstrap_repeats=bootstrap_repeats,
        )
        all_metrics.extend(metric_rows)
        all_predictions.extend(prediction_rows)
        checkpoint_hashes[name] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    comparisons = [
        paired_comparison(all_metrics, "A4", "handcrafted"),
        paired_comparison(all_metrics, "A4", "A3"),
    ]
    primary = comparisons[0]
    decision = {
        "minimum_effect_macro_f1": MINIMUM_EFFECT,
        "primary_effect_at_least_minimum": primary["mean_difference"] >= MINIMUM_EFFECT,
        "primary_interval_excludes_zero": primary["ci_95_low"] > 0.0,
        "promotion_decision": "do_not_promote_a4",
        "decision_reason": (
            "The development protocol already rejected promotion; the final in-session test "
            "is a one-time confirmatory estimate and cannot reopen model selection."
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "protocol": config["study"]["protocol"],
        "scope": "one-time in-session test of source-condition proxies; not morphology or OOD",
        "dataset": config["study"]["real_dataset"],
        "test_split": FINAL_SPLIT,
        "n_test_events": int(test_indices.size),
        "n_test_capture_blocks": len(test_blocks),
        "class_names": data.class_names,
        "label_fraction": LABEL_FRACTION,
        "probe_seeds": probe_seeds,
        "methods": ["handcrafted", "A3", "A4"],
        "results": all_metrics,
        "paired_comparisons": comparisons,
        "decision": decision,
        "development_decision_sha256": hashlib.sha256(
            args.development_decision.read_bytes()
        ).hexdigest(),
        "checkpoint_sha256": checkpoint_hashes,
        "sealed_splits_used": [FINAL_SPLIT],
    }
    args.output_dir.mkdir(parents=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path = args.output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_predictions[0]))
        writer.writeheader()
        writer.writerows(all_predictions)

    repo_root = Path(__file__).resolve().parents[1]
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": config["study"]["real_dataset"],
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(repo_root.parent / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "outputs": ["metrics.json", "predictions.csv"],
        "output_sha256": {
            "metrics.json": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            "predictions.csv": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        },
        "sealed_splits_used": [FINAL_SPLIT],
        "final_test_opened_once": True,
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"comparisons": comparisons, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
