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

from p3_ssl.collapse_utility import paired_group_macro_f1_difference
from p3_ssl.collapse_utility_audit import classify_supplement, validation_block_support
from p3_ssl.config import load_config
from p3_ssl.followup_features import extract_feature_families, feature_matrix
from p3_ssl.study_baselines import (
    fit_linear_probe,
    handcrafted_features,
    load_baseline_data,
    prediction_metrics,
    sample_record_groups,
)
from p3_ssl.study_evaluation import calibration_metrics


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
        "scripts/audit_yeast_collapse_utility_supplement.py",
        "configs/yeast_ssl_collapse_utility_supplement_v1.yaml",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return not status.strip()


def _validate_source(source: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((source / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("profile") != "full"
        or manifest.get("sealed_splits_used") != []
        or metrics.get("protocol") != config["study"]["source_protocol"]
        or metrics.get("decision") != "reject_mask_only_rescue"
        or metrics.get("sealed_splits_used") != []
    ):
        raise ValueError("Source utility run is not the complete frozen rejection")
    for name, expected in manifest.get("outputs", {}).items():
        if _sha256(source / name) != expected:
            raise ValueError(f"Source utility checksum mismatch: {name}")
    if manifest.get("dataset_manifest_sha256") != config["study"]["dataset_manifest_sha256"]:
        raise ValueError("Source utility dataset manifest differs from supplement config")
    return manifest, metrics


def _plot(
    path: Path,
    source_results: list[dict[str, Any]],
    supplement_results: list[dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    all_results = [
        {**row, "method": "C1"}
        for row in source_results
        if row["method"] == "C1"
    ] + supplement_results
    for method in ("C1", "handcrafted_signal", "handcrafted_full", "handcrafted_full_plus_C1"):
        rows = [row for row in all_results if row["method"] == method]
        fractions = sorted({float(row["label_fraction"]) for row in rows})
        values = [
            np.mean([row["macro_f1"] for row in rows if row["label_fraction"] == fraction])
            for fraction in fractions
        ]
        axes[0].plot(fractions, values, marker="o", label=method)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Labeled training fraction")
    axes[0].set_ylabel("Development macro-F1")
    axes[0].set_title("Post-hoc baseline completion")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    names = list(comparisons)
    gains = np.asarray([comparisons[name]["gain"] for name in names])
    lower = np.asarray([comparisons[name]["paired_interval"][0] for name in names])
    upper = np.asarray([comparisons[name]["paired_interval"][1] for name in names])
    positions = np.arange(len(names))
    axes[1].bar(positions, gains, color="#0077b6")
    axes[1].errorbar(
        positions,
        gains,
        yerr=[gains - lower, upper - gains],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(0.03, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xticks(positions, names, rotation=15, ha="right")
    axes[1].set_ylabel("Macro-F1 difference")
    axes[1].set_title("10% labels; block intervals are descriptive")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete the yeast C1 utility baselines.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/yeast_ssl_collapse_utility_supplement_v1.yaml"),
    )
    parser.add_argument("--source-utility-run", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output already exists: {args.output_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    code_tree_clean = _code_tree_is_clean(repo_root)
    if not code_tree_clean and not args.allow_dirty_smoke:
        raise RuntimeError("The complete supplement requires committed code and config")
    config = load_config(args.config)
    if args.dataset_manifest_sha256 != config["study"]["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the frozen supplement config")
    source_manifest, source_metrics = _validate_source(args.source_utility_run, config)
    data = load_baseline_data(args.real_root, max_per_class=None, seed=42)
    selected = np.concatenate([data.train_indices, data.validation_indices])
    source_rows = np.load(args.source_utility_run / "embedding_row_indices.npy")
    if not np.array_equal(selected, source_rows):
        raise ValueError("Supplement rows do not match the source utility embeddings")
    signals = np.asarray(data.signals[selected], dtype=np.float32)
    selected_metadata = [data.rows[int(index)] for index in selected]
    c1 = np.load(args.source_utility_run / "embeddings_c1.npy")
    if c1.shape[0] != selected.size:
        raise ValueError("C1 embedding rows do not match the development dataset")

    signal_features = handcrafted_features(signals)
    families, family_names = extract_feature_families(signals, selected_metadata)
    full_features = feature_matrix(families)
    methods = {
        "handcrafted_signal": signal_features,
        "handcrafted_full": full_features,
        "handcrafted_full_plus_C1": np.concatenate([full_features, c1], axis=1),
    }
    evaluation = config["evaluation"]
    fractions = [float(value) for value in evaluation["label_fractions"]]
    seeds = [int(value) for value in evaluation["probe_seeds"]]
    validation = data.validation_indices
    validation_labels = data.labels[validation]
    validation_rows = [data.rows[int(index)] for index in validation]
    validation_groups = np.asarray(
        [row.get("capture_block_id") or row["record_id"] for row in validation_rows],
        dtype=str,
    )

    results: list[dict[str, Any]] = []
    predictions: dict[tuple[str, float, int], np.ndarray] = {}
    arrays: dict[str, np.ndarray] = {
        "validation_row_indices": validation,
        "validation_labels": validation_labels,
        "validation_groups": validation_groups,
    }
    for method, values in methods.items():
        full = np.empty((len(data.rows), values.shape[1]), dtype=np.float32)
        full[selected] = values
        for fraction in fractions:
            for seed in seeds:
                model, train = fit_linear_probe(full, data, fraction=fraction, seed=seed)
                probabilities = model.predict_proba(full[validation]).astype(np.float32)
                prediction = probabilities.argmax(axis=1).astype(np.int16)
                key = f"{method}__{fraction:g}__s{seed}"
                arrays[f"predictions__{key}"] = prediction
                arrays[f"probabilities__{key}"] = probabilities
                arrays[f"train_rows__{key}"] = sample_record_groups(
                    data.rows, data.labels, data.train_indices, fraction, seed
                )
                predictions[(method, fraction, seed)] = prediction
                results.append(
                    {
                        "method": method,
                        "label_fraction": fraction,
                        "seed": seed,
                        "n_probe_events": int(train.size),
                        "n_probe_records": len(
                            {data.rows[int(index)]["record_id"] for index in train}
                        ),
                        "probe_optimization": model.probe_optimization_,
                        "calibration": calibration_metrics(validation_labels, probabilities),
                        **prediction_metrics(
                            validation_labels, prediction, data.class_names
                        ),
                    }
                )

    primary_fraction = float(evaluation["primary_fraction"])
    with np.load(args.source_utility_run / "validation_predictions.npz") as source_predictions:
        if (
            not np.array_equal(source_predictions["validation_row_indices"], validation)
            or not np.array_equal(source_predictions["validation_labels"], validation_labels)
            or not np.array_equal(source_predictions["validation_groups"], validation_groups)
        ):
            raise ValueError("Source utility prediction alignment differs from the supplement")
        c1_predictions = [
            source_predictions[f"C1__{primary_fraction:g}__s{seed}"].copy()
            for seed in seeds
        ]

    comparison_kwargs = {
        "class_count": len(data.class_names),
        "repeats": int(evaluation["grouped_bootstrap_repeats"]),
        "seed": 20260717,
        "interval_level": float(evaluation["paired_interval_level"]),
    }
    c1_vs_handcrafted = {
        method: paired_group_macro_f1_difference(
            validation_labels,
            validation_groups,
            c1_predictions,
            [predictions[(method, primary_fraction, seed)] for seed in seeds],
            **comparison_kwargs,
        )
        for method in ("handcrafted_signal", "handcrafted_full")
    }
    fusion_vs_handcrafted = paired_group_macro_f1_difference(
        validation_labels,
        validation_groups,
        [
            predictions[("handcrafted_full_plus_C1", primary_fraction, seed)]
            for seed in seeds
        ],
        [predictions[("handcrafted_full", primary_fraction, seed)] for seed in seeds],
        **comparison_kwargs,
    )
    support = validation_block_support(data.rows, data.labels, validation, data.class_names)
    all_converged = all(
        bool(row["probe_optimization"]["converged"])
        for row in results
        if row["label_fraction"] == primary_fraction
    )
    decision = classify_supplement(
        source_decision=source_metrics["decision"],
        c1_vs_handcrafted=c1_vs_handcrafted,
        fusion_vs_handcrafted=fusion_vs_handcrafted,
        all_probes_converged=all_converged,
        minimum_blocks_per_class=int(support["minimum_blocks_per_class"]),
        required_blocks_per_class=int(evaluation["required_blocks_per_class"]),
        minimum_complementarity_gain=float(evaluation["minimum_complementarity_gain"]),
    )

    args.output_dir.mkdir(parents=True)
    arrays_path = args.output_dir / "probe_audit_arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    features_path = args.output_dir / "feature_manifest.json"
    features_path.write_text(
        json.dumps(
            {
                "handcrafted_signal_dimensions": int(signal_features.shape[1]),
                "handcrafted_full_dimensions": int(full_features.shape[1]),
                "families": family_names,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    comparisons = {
        "C1-minus-signal": c1_vs_handcrafted["handcrafted_signal"],
        "C1-minus-full": c1_vs_handcrafted["handcrafted_full"],
        "fusion-minus-full": fusion_vs_handcrafted,
    }
    figure_path = args.output_dir / "collapse_utility_supplement.png"
    _plot(figure_path, source_metrics["results"], results, comparisons)
    payload = {
        "schema_version": 1,
        "run_id": args.run_id,
        "protocol": config["study"]["protocol"],
        "endpoint_scope": config["study"]["endpoint_scope"],
        "dataset": config["study"]["dataset"],
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "source_utility_run": source_manifest["run_id"],
        "source_utility_run_json_sha256": _sha256(args.source_utility_run / "run.json"),
        "source_utility_metrics_sha256": _sha256(args.source_utility_run / "metrics.json"),
        "class_names": data.class_names,
        "results": results,
        "comparisons": comparisons,
        "validation_block_support": support,
        "decision_audit": decision,
        "all_primary_probes_converged": all_converged,
        "interpretation": (
            "Post-hoc baseline completion. Point estimates are useful diagnostics, but the "
            "paired intervals are descriptive because one proxy class has insufficient "
            "independent validation blocks."
        ),
        "sealed_splits_used": [],
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs = (metrics_path, arrays_path, features_path, figure_path)
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
        "profile": "post-hoc-baseline-completion",
        "code_tree_clean": code_tree_clean,
        "config_sha256": _sha256(args.config),
        "source_utility_run_json_sha256": _sha256(args.source_utility_run / "run.json"),
        "sealed_splits_used": [],
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"decision_audit": decision, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
