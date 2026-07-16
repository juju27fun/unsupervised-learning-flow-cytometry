#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from p3_ssl.followup_features import (
    FEATURE_FAMILY_ORDER,
    extract_feature_families,
    feature_matrix,
    fit_probe,
    load_followup_development,
    load_historical_embeddings,
    sample_record_groups,
    write_feature_manifest,
)


FRACTIONS = (0.01, 0.05, 0.10, 0.25, 1.00)
SEEDS = (42, 43, 44)


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cluster_bootstrap_delta(
    *,
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    blocks: np.ndarray,
    n_classes: int,
    seed: int,
    repetitions: int = 1000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    unique = np.asarray(sorted(set(blocks.tolist())))
    by_block = {block: np.flatnonzero(blocks == block) for block in unique}
    deltas = []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_block[block] for block in sampled])
        kwargs = {"labels": np.arange(n_classes), "average": "macro", "zero_division": 0}
        deltas.append(
            f1_score(labels[indices], left[indices], **kwargs)
            - f1_score(labels[indices], right[indices], **kwargs)
        )
    array = np.asarray(deltas)
    return {
        "mean": float(np.mean(array)),
        "ci_95_low": float(np.quantile(array, 0.025)),
        "ci_95_high": float(np.quantile(array, 0.975)),
        "probability_positive": float(np.mean(array > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit handcrafted families and historical SSL complementarity.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--historical-embedding-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mlp-fractions", default="0.10", help="Comma-separated fixed MLP sensitivity fractions")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    data = load_followup_development(args.dataset_root)
    families, names = extract_feature_families(data.signals, data.rows)
    write_feature_manifest(args.output_dir / "feature_manifest.json", names)
    all_handcrafted = feature_matrix(families)
    methods: dict[str, np.ndarray] = {
        **{f"family_{name}": values for name, values in families.items()},
        "handcrafted_all": all_handcrafted,
        "rms_only": families["energy_amplitude"][:, :1],
    }
    for excluded in FEATURE_FAMILY_ORDER:
        included = tuple(name for name in FEATURE_FAMILY_ORDER if name != excluded)
        methods[f"handcrafted_without_{excluded}"] = feature_matrix(families, include=included)
    embeddings = load_historical_embeddings(
        artifact_root=args.historical_embedding_root,
        followup_rows=data.rows,
    )
    for name, values in embeddings.items():
        methods[name] = values
        methods[f"handcrafted_plus_{name}"] = np.concatenate([all_handcrafted, values], axis=1)

    mlp_fractions = {float(value) for value in args.mlp_fractions.split(",") if value}
    metrics_rows: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, float, int], np.ndarray] = {}
    for method, values in methods.items():
        representation_seed = int(method.rsplit("_s", 1)[1]) if "_s" in method else None
        for fraction in FRACTIONS:
            for probe_seed in SEEDS:
                train = sample_record_groups(
                    data.rows, data.labels, data.train_indices, fraction, probe_seed
                )
                probes = ("linear", "mlp") if fraction in mlp_fractions else ("linear",)
                for probe in probes:
                    result, predictions = fit_probe(
                        values[train],
                        data.labels[train],
                        values[data.validation_indices],
                        data.labels[data.validation_indices],
                        probe=probe,
                        seed=probe_seed,
                        class_names=data.class_names,
                    )
                    metrics_rows.append(
                        {
                            "method": method,
                            "probe": probe,
                            "label_fraction": fraction,
                            "probe_seed": probe_seed,
                            "representation_seed": representation_seed if representation_seed else "",
                            "n_train_events": len(train),
                            "macro_f1": result["macro_f1"],
                            "balanced_accuracy_observed_classes": result[
                                "balanced_accuracy_observed_classes"
                            ],
                            "multiclass_log_loss": result["multiclass_log_loss"],
                            "converged": result["converged"],
                            "convergence_warning_count": result["convergence_warning_count"],
                            "n_iter_json": json.dumps(result["n_iter"]),
                            "per_class_recall_json": json.dumps(result["per_class_recall"], sort_keys=True),
                        }
                    )
                    if probe == "linear":
                        prediction_cache[(method, fraction, probe_seed)] = predictions
    _write_csv(args.output_dir / "probe_metrics.csv", metrics_rows)

    validation_labels = data.labels[data.validation_indices]
    validation_blocks = np.asarray(
        [data.rows[int(index)]["capture_block_id"] for index in data.validation_indices]
    )
    comparisons = []
    for cell in ("A3", "A4"):
        for representation_seed in SEEDS:
            learned = f"{cell}_s{representation_seed}"
            fused = f"handcrafted_plus_{learned}"
            for fraction in FRACTIONS:
                for probe_seed in SEEDS:
                    key = (fused, fraction, probe_seed)
                    comparison = _cluster_bootstrap_delta(
                        labels=validation_labels,
                        left=prediction_cache[key],
                        right=prediction_cache[("handcrafted_all", fraction, probe_seed)],
                        blocks=validation_blocks,
                        n_classes=len(data.class_names),
                        seed=representation_seed * 1000 + probe_seed + int(fraction * 100),
                    )
                    comparisons.append(
                        {
                            "cell": cell,
                            "representation_seed": representation_seed,
                            "probe_seed": probe_seed,
                            "label_fraction": fraction,
                            **comparison,
                        }
                    )
    _write_csv(args.output_dir / "fusion_cluster_bootstrap.csv", comparisons)

    linear = [row for row in metrics_rows if row["probe"] == "linear"]
    method_summary = []
    for method in sorted({row["method"] for row in linear}):
        for fraction in FRACTIONS:
            values = [
                float(row["macro_f1"])
                for row in linear
                if row["method"] == method and float(row["label_fraction"]) == fraction
            ]
            method_summary.append(
                {
                    "method": method,
                    "label_fraction": fraction,
                    "macro_f1_mean": float(np.mean(values)),
                    "macro_f1_std": float(np.std(values)),
                    "n": len(values),
                }
            )
    _write_csv(args.output_dir / "method_summary.csv", method_summary)
    core = [row for row in comparisons if float(row["label_fraction"]) == 0.10]
    by_cell = {
        cell: {
            "mean_delta": float(np.mean([row["mean"] for row in core if row["cell"] == cell])),
            "worst_ci_low": float(min(row["ci_95_low"] for row in core if row["cell"] == cell)),
            "best_ci_high": float(max(row["ci_95_high"] for row in core if row["cell"] == cell)),
        }
        for cell in ("A3", "A4")
    }
    if max(item["mean_delta"] for item in by_cell.values()) >= 0.03:
        classification = "complementary_signal_present"
    else:
        mlp_rows = [
            row for row in metrics_rows
            if row["probe"] == "mlp" and float(row["label_fraction"]) == 0.10
        ]
        mlp_gain = []
        for cell in ("A3", "A4"):
            fused = [float(row["macro_f1"]) for row in mlp_rows if row["method"].startswith(f"handcrafted_plus_{cell}")]
            base = [float(row["macro_f1"]) for row in mlp_rows if row["method"] == "handcrafted_all"]
            if fused and base:
                mlp_gain.append(float(np.mean(fused) - np.mean(base)))
        classification = "nonlinear_only_signal" if mlp_gain and max(mlp_gain) >= 0.03 else "redundant_with_handcrafted"
    summary = {
        "schema_version": 1,
        "status": "complete",
        "endpoint": "followup_validation_only",
        "sealed_splits_used": [],
        "class_names": data.class_names,
        "validation_proxy_missing": "shmoo",
        "historical_representation_classification": classification,
        "fusion_at_10_percent": by_cell,
        "all_probes_converged": all(bool(row["converged"]) for row in metrics_rows),
        "n_probe_fits": len(metrics_rows),
    }
    (args.output_dir / "complementarity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run = {
        "schema_version": 1,
        "run_id": args.run_id,
        "project": "unsupervised-learning-flow-cytometry",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "audit_yeast_followup_complementarity.py",
        "dataset": "yeast-events-followup@v2",
        "profile": "week1-diagnostic",
        "sealed_splits_used": [],
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(Path(__file__).resolve().parents[1]),
            "particles2SNR-pipeline": _revision(Path(__file__).resolve().parents[2] / "particles2SNR-pipeline"),
        },
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "run.json"
        },
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
