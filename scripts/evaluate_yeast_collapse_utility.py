#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from p3_ssl.collapse_utility import (
    apply_utility_gate,
    fit_train_only_pca,
    paired_group_macro_f1_difference,
)
from p3_ssl.config import load_config, validate_collapse_utility_config
from p3_ssl.study_baselines import (
    checkpoint_encoder_features,
    load_baseline_data,
    random_encoder_features,
)
from p3_ssl.study_evaluation import (
    cross_recording_retrieval,
    evaluate_linear_probe,
    label_efficiency_auc,
    real_variability_summary,
)
from p3_ssl.study_training import embedding_health_statistics, model_config_from_study


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _code_tree_is_clean(repo_root: Path) -> bool:
    paths = (
        "p3_ssl",
        "scripts/evaluate_yeast_collapse_utility.py",
        "configs/yeast_ssl_collapse_utility_v1.yaml",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return not status.strip()


def _validate_promotion(
    report_path: Path,
    c0_run: Path,
    c1_run: Path,
    protocol: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_manifest = json.loads((report_path.parent / "run.json").read_text(encoding="utf-8"))
    if report_manifest.get("outputs", {}).get("metrics.json") != _sha256(report_path):
        raise ValueError("Promotion report checksum mismatch")
    if report.get("decision") != "run_development_utility_evaluation" or not report.get(
        "eligible_for_utility_evaluation"
    ):
        raise ValueError("Promotion report does not authorize utility evaluation")
    if report.get("protocol") != protocol or report.get("sealed_splits_used"):
        raise ValueError("Promotion report protocol or split contract mismatch")

    manifests = {}
    for cell, run_dir in (("C0", c0_run), ("C1", c1_run)):
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if (
            manifest.get("cell") != cell
            or manifest.get("status") != "complete"
            or manifest.get("profile") != "full"
            or int(manifest.get("seed", -1)) != 42
            or manifest.get("sealed_splits_used")
            or not manifest.get("code_tree_clean")
        ):
            raise ValueError(f"Invalid promoted {cell} run")
        for name, expected in manifest.get("outputs", {}).items():
            if _sha256(run_dir / name) != expected:
                raise ValueError(f"{cell} output checksum mismatch: {name}")
        manifests[cell] = manifest
    comparable = (
        "protocol",
        "mask_policy",
        "seed",
        "profile",
        "selection_report_sha256",
        "config_sha256",
        "source_mask_ablation_config_sha256",
        "base_config_sha256",
        "training_contract",
        "dataset_contract",
        "repositories",
    )
    if any(manifests["C0"].get(key) != manifests["C1"].get(key) for key in comparable):
        raise ValueError("Promoted C0/C1 runs are not comparable")
    return report, manifests


def _plot(
    path: Path,
    results: list[dict[str, Any]],
    gate: dict[str, Any],
    method_order: list[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in method_order:
        rows = [row for row in results if row["method"] == method]
        fractions = sorted({float(row["label_fraction"]) for row in rows})
        means = [
            np.mean([row["macro_f1"] for row in rows if row["label_fraction"] == fraction])
            for fraction in fractions
        ]
        axes[0].plot(fractions, means, marker="o", label=method)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Labeled training fraction")
    axes[0].set_ylabel("Development macro-F1")
    axes[0].set_title("Label efficiency")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    comparisons = gate["comparisons"]
    names = list(comparisons)
    gains = [comparisons[name]["gain"] for name in names]
    lower = [comparisons[name]["paired_interval"][0] for name in names]
    upper = [comparisons[name]["paired_interval"][1] for name in names]
    positions = np.arange(len(names))
    axes[1].bar(positions, gains, color="#0077b6")
    axes[1].errorbar(
        positions,
        gains,
        yerr=[np.asarray(gains) - np.asarray(lower), np.asarray(upper) - np.asarray(gains)],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(0.03, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(positions, names)
    axes[1].set_ylabel("C1 macro-F1 gain")
    axes[1].set_title("Primary 10% paired comparisons")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate promoted yeast C1 development utility.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/yeast_ssl_collapse_utility_v1.yaml")
    )
    parser.add_argument("--promotion-report", type=Path, required=True)
    parser.add_argument("--c0-run", type=Path, required=True)
    parser.add_argument("--c1-run", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    code_tree_clean = _code_tree_is_clean(repo_root)
    if args.profile == "full" and not code_tree_clean:
        raise RuntimeError("Full utility evaluation requires committed code and config")
    config = load_config(args.config)
    validate_collapse_utility_config(config)
    if args.dataset_manifest_sha256 != config["study"]["real_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the frozen utility config")
    promotion, source_manifests = _validate_promotion(
        args.promotion_report,
        args.c0_run,
        args.c1_run,
        config["study"]["promotion_protocol"],
    )
    summary = json.loads((args.real_root / "dataset_summary.json").read_text(encoding="utf-8"))
    if (
        summary.get("dataset_id") != config["study"]["real_dataset"]
        or summary.get("source_dataset_id") != config["study"]["parent_real_dataset"]
        or summary.get("sealed_splits_used") != []
        or set(summary.get("split_counts", {}))
        != {"development_train", "development_validation"}
    ):
        raise ValueError("Development-only dataset provenance contract failed")

    utility_profile = config["profiles"][args.profile]
    base_config = load_config(_repo_path(repo_root, config["study"]["base_config"]))
    data = load_baseline_data(
        args.real_root,
        max_per_class=utility_profile["max_events_per_class"],
        seed=42,
    )
    selected = np.concatenate([data.train_indices, data.validation_indices])
    signals = np.asarray(data.signals[selected], dtype=np.float32)
    n_train = len(data.train_indices)
    device = torch.device(args.device)
    batch_size = int(utility_profile["batch_size"])
    method_order = list(config["methods"])
    subset_features: dict[str, np.ndarray] = {}
    method_metadata: dict[str, Any] = {}

    for cell, run_dir in (("C1", args.c1_run), ("C0", args.c0_run)):
        features, metadata = checkpoint_encoder_features(
            signals, run_dir / "checkpoint.pt", batch_size=batch_size, device=device
        )
        subset_features[cell] = features
        method_metadata[cell] = {
            **metadata,
            "checkpoint_sha256": _sha256(run_dir / "checkpoint.pt"),
            "embedding_health": embedding_health_statistics(features),
        }
    random_features = random_encoder_features(
        signals,
        config=model_config_from_study(base_config),
        batch_size=batch_size,
        device=device,
        seed=int(config["methods"]["random"]["seed"]),
    )
    subset_features["random"] = random_features
    method_metadata["random"] = {
        "seed": int(config["methods"]["random"]["seed"]),
        "embedding_health": embedding_health_statistics(random_features),
    }
    subset_features["raw"] = signals
    method_metadata["raw"] = {"dimensions": int(signals.shape[1])}
    pca_features, pca_metadata = fit_train_only_pca(
        signals[:n_train], signals, config["methods"]["pca96"]
    )
    subset_features["pca96"] = pca_features
    method_metadata["pca96"] = {
        **pca_metadata,
        "embedding_health": embedding_health_statistics(pca_features),
    }

    validation_rows = [data.rows[int(index)] for index in data.validation_indices]
    validation_labels = data.labels[data.validation_indices]
    groups = np.asarray(
        [row.get("capture_block_id") or row["record_id"] for row in validation_rows],
        dtype=str,
    )
    results = []
    predictions: dict[tuple[str, float, int], np.ndarray] = {}
    for method in method_order:
        values = subset_features[method]
        full = np.empty((len(data.rows), values.shape[1]), dtype=np.float32)
        full[selected] = values
        validation_values = values[n_train:]
        method_metadata[method]["development_retrieval"] = cross_recording_retrieval(
            validation_values,
            validation_rows,
            validation_labels,
            neighbors=int(config["evaluation"]["retrieval_neighbors"]),
        )
        for fraction in utility_profile["label_fractions"]:
            for seed in utility_profile["probe_seeds"]:
                metrics, model = evaluate_linear_probe(
                    full,
                    data,
                    fraction=float(fraction),
                    seed=int(seed),
                    bootstrap_repeats=int(utility_profile["grouped_bootstrap_repeats"]),
                    calibration_bins=int(config["evaluation"]["calibration_bins"]),
                )
                prediction = model.predict(full[data.validation_indices]).astype(np.int16)
                predictions[(method, float(fraction), int(seed))] = prediction
                results.append(
                    {
                        "method": method,
                        "label_fraction": float(fraction),
                        "seed": int(seed),
                        **metrics,
                    }
                )

    primary_fraction = float(config["primary_gate"]["label_fraction"])
    comparisons = {}
    for baseline in method_order[1:]:
        candidate_predictions = [
            predictions[("C1", primary_fraction, int(seed))]
            for seed in utility_profile["probe_seeds"]
        ]
        baseline_predictions = [
            predictions[(baseline, primary_fraction, int(seed))]
            for seed in utility_profile["probe_seeds"]
        ]
        comparisons[baseline] = paired_group_macro_f1_difference(
            validation_labels,
            groups,
            candidate_predictions,
            baseline_predictions,
            class_count=len(data.class_names),
            repeats=int(utility_profile["grouped_bootstrap_repeats"]),
            seed=20260717,
            interval_level=float(config["primary_gate"]["paired_interval_level"]),
        )
    gate = apply_utility_gate(
        comparisons, config, scientific_decision_allowed=args.profile == "full"
    )

    args.output_dir.mkdir(parents=True)
    prediction_path = args.output_dir / "validation_predictions.npz"
    np.savez_compressed(
        prediction_path,
        validation_row_indices=data.validation_indices,
        validation_labels=validation_labels,
        validation_groups=groups,
        **{
            f"{method}__{fraction:g}__s{seed}": values
            for (method, fraction, seed), values in predictions.items()
        },
    )
    row_path = args.output_dir / "embedding_row_indices.npy"
    np.save(row_path, selected)
    embedding_paths = []
    for method in ("C1", "C0", "random", "pca96"):
        path = args.output_dir / f"embeddings_{method.lower()}.npy"
        np.save(path, subset_features[method])
        embedding_paths.append(path)
    figure_path = args.output_dir / "collapse_utility.png"
    _plot(figure_path, results, gate, method_order)
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "protocol": config["study"]["protocol"],
        "profile": args.profile,
        "dataset": config["study"]["real_dataset"],
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "parent_dataset_manifest_sha256": config["study"]["parent_manifest_sha256"],
        "promotion_report_sha256": _sha256(args.promotion_report),
        "endpoint_scope": config["study"]["endpoint_scope"],
        "class_names": data.class_names,
        "n_train_events": int(data.train_indices.size),
        "n_validation_events": int(data.validation_indices.size),
        "methods": method_order,
        "method_metadata": method_metadata,
        "results": results,
        "label_efficiency_auc": label_efficiency_auc(results, "method"),
        "development_variability": real_variability_summary(data),
        "primary_gate": gate,
        "decision": gate["decision"],
        "sealed_splits_used": [],
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs = [metrics_path, prediction_path, row_path, figure_path, *embedding_paths]
    run = {
        "schema_version": 1,
        "project": "unsupervised-learning-flow-cytometry",
        "run_id": args.run_id,
        "dataset": config["study"]["real_dataset"],
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(repo_root),
            "particles2SNR-pipeline": _revision(repo_root.parent / "particles2SNR-pipeline"),
        },
        "command": " ".join(sys.argv),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "profile": args.profile,
        "code_tree_clean": code_tree_clean,
        "config_sha256": _sha256(args.config),
        "promotion_report_sha256": _sha256(args.promotion_report),
        "source_run_manifests": {
            cell: {
                "run_id": source_manifests[cell]["run_id"],
                "run_json_sha256": _sha256(run_dir / "run.json"),
                "checkpoint_sha256": _sha256(run_dir / "checkpoint.pt"),
            }
            for cell, run_dir in (("C0", args.c0_run), ("C1", args.c1_run))
        },
        "sealed_splits_used": [],
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"primary_gate": gate, "run": run}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
