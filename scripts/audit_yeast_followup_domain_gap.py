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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from p3_ssl.followup_domain import (
    OBSERVABLE_NAMES,
    fit_domain_probe,
    matched_pairs,
    signal_observables,
    signal_summary_features,
)
from p3_ssl.followup_features import load_followup_development


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, text=True, capture_output=True
    ).stdout.strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_indices(length: int, maximum: int, seed: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(length, size=maximum, replace=False))


def _load_synthetic(root: Path, split: str) -> tuple[np.ndarray, list[dict[str, str]]]:
    rows = [row for row in _read_csv(root / "simulation_metadata.csv") if row["split"] == split]
    if any("view_index" in row and int(row["view_index"]) != 0 for row in rows):
        rows = [row for row in rows if int(row.get("view_index", 0)) == 0]
    signals = np.load(root / "signals.npy", mmap_mode="r")
    return np.asarray([signals[int(row["signal_row"])] for row in rows], dtype=np.float32), rows


def _group_bootstrap_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    repetitions: int = 1000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    by_domain = {}
    for domain in (0, 1):
        domain_groups = sorted(set(groups[labels == domain].tolist()))
        by_domain[domain] = {
            group: np.flatnonzero((labels == domain) & (groups == group)) for group in domain_groups
        }
    values = []
    for _ in range(repetitions):
        indices = []
        for domain in (0, 1):
            mapping = by_domain[domain]
            keys = np.asarray(list(mapping))
            sampled = rng.choice(keys, size=len(keys), replace=True)
            indices.extend(mapping[group] for group in sampled)
        selected = np.concatenate(indices)
        values.append(roc_auc_score(labels[selected], probabilities[selected]))
    array = np.asarray(values)
    return {
        "ci_95_low": float(np.quantile(array, 0.025)),
        "ci_95_high": float(np.quantile(array, 0.975)),
    }


def _synthetic_group(row: dict[str, str]) -> str:
    return row.get("template_source_record_id") or row.get("latent_id") or str(row["signal_row"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditional simulation-real domain-gap audit.")
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--analytic-root", type=Path, required=True)
    parser.add_argument("--template-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-validation", type=int, default=1000)
    parser.add_argument("--caliper", type=float, default=1.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    real = load_followup_development(args.real_root)
    real_split = np.asarray([row["development_split"] for row in real.rows])
    real_sets = {}
    for split, name, maximum, seed in (
        ("followup_train", "train", args.max_train, 42),
        ("followup_validation", "validation", args.max_validation, 43),
    ):
        available = np.flatnonzero(real_split == split)
        selected = available[_balanced_indices(len(available), maximum, seed)]
        real_sets[name] = (
            real.signals[selected],
            [real.rows[int(index)] for index in selected],
        )

    result_rows: list[dict[str, Any]] = []
    matching_reports = {}
    importance_reports = {}
    for source_name, source_root, retained_factors in (
        ("analytic", args.analytic_root, True),
        ("template_diagnostic", args.template_root, False),
    ):
        synthetic_sets = {}
        for split, maximum, seed in (
            ("train", args.max_train, 52),
            ("validation", args.max_validation, 53),
        ):
            signals, rows = _load_synthetic(source_root, split)
            selected = _balanced_indices(len(signals), maximum, seed)
            synthetic_sets[split] = (signals[selected], [rows[int(index)] for index in selected])

        train_real_signals, train_real_rows = real_sets["train"]
        validation_real_signals, validation_real_rows = real_sets["validation"]
        train_synthetic_signals, train_synthetic_rows = synthetic_sets["train"]
        validation_synthetic_signals, validation_synthetic_rows = synthetic_sets["validation"]
        train_real_observables = signal_observables(train_real_signals)
        validation_real_observables = signal_observables(validation_real_signals)
        train_synthetic_observables = signal_observables(train_synthetic_signals)
        validation_synthetic_observables = signal_observables(validation_synthetic_signals)
        scaler = StandardScaler().fit(
            np.concatenate([train_real_observables, train_synthetic_observables])
        )
        train_real_match, train_synthetic_match, train_match_report = matched_pairs(
            train_real_observables,
            train_synthetic_observables,
            scaler=scaler,
            caliper=args.caliper,
        )
        validation_real_match, validation_synthetic_match, validation_match_report = matched_pairs(
            validation_real_observables,
            validation_synthetic_observables,
            scaler=scaler,
            caliper=args.caliper,
        )
        matching_reports[source_name] = {
            "observable_names": list(OBSERVABLE_NAMES),
            "train": train_match_report,
            "validation": validation_match_report,
            "common_support_sufficient": bool(
                train_match_report["real_retained_fraction"] >= 0.50
                and validation_match_report["real_retained_fraction"] >= 0.50
                and train_match_report["post_match_smd_max"] <= 0.25
                and validation_match_report["post_match_smd_max"] <= 0.25
            ),
            "frozen_sufficiency_rule": "retained fraction >= 0.50 and max post-match SMD <= 0.25 in train and validation",
        }

        train_real_summary, summary_names = signal_summary_features(train_real_signals)
        validation_real_summary, _ = signal_summary_features(validation_real_signals)
        train_synthetic_summary, _ = signal_summary_features(train_synthetic_signals)
        validation_synthetic_summary, _ = signal_summary_features(validation_synthetic_signals)
        feature_sets = {
            "observables": (
                train_real_observables,
                train_synthetic_observables,
                validation_real_observables,
                validation_synthetic_observables,
                list(OBSERVABLE_NAMES),
            ),
            "signal_summary": (
                train_real_summary,
                train_synthetic_summary,
                validation_real_summary,
                validation_synthetic_summary,
                summary_names,
            ),
            "downsampled_signal": (
                train_real_signals[:, ::16],
                train_synthetic_signals[:, ::16],
                validation_real_signals[:, ::16],
                validation_synthetic_signals[:, ::16],
                [f"sample_{index * 16}" for index in range(train_real_signals[:, ::16].shape[1])],
            ),
        }
        for match_state in ("unmatched", "matched"):
            for feature_set, values in feature_sets.items():
                train_r, train_s, validation_r, validation_s, feature_names = values
                if match_state == "matched":
                    train_r = train_r[train_real_match]
                    train_s = train_s[train_synthetic_match]
                    validation_r = validation_r[validation_real_match]
                    validation_s = validation_s[validation_synthetic_match]
                    selected_real_rows = [validation_real_rows[int(index)] for index in validation_real_match]
                    selected_synthetic_rows = [
                        validation_synthetic_rows[int(index)] for index in validation_synthetic_match
                    ]
                else:
                    selected_real_rows = validation_real_rows
                    selected_synthetic_rows = validation_synthetic_rows
                for model in ("linear", "forest"):
                    result = fit_domain_probe(
                        train_r,
                        train_s,
                        validation_r,
                        validation_s,
                        feature_names=feature_names,
                        model=model,
                        seed=42,
                        compute_importance=feature_set != "downsampled_signal",
                    )
                    labels = np.concatenate(
                        [np.zeros(len(validation_r), dtype=int), np.ones(len(validation_s), dtype=int)]
                    )
                    groups = np.asarray(
                        [row["capture_block_id"] for row in selected_real_rows]
                        + [_synthetic_group(row) for row in selected_synthetic_rows]
                    )
                    interval = _group_bootstrap_auc(labels, result.probabilities, groups, seed=44)
                    key = f"{source_name}:{match_state}:{feature_set}:{model}"
                    importance_reports[key] = dict(list(result.importance.items())[:15])
                    result_rows.append(
                        {
                            "simulation_source": source_name,
                            "retained_physical_factors": retained_factors,
                            "match_state": match_state,
                            "feature_set": feature_set,
                            "model": model,
                            "validation_roc_auc": result.roc_auc,
                            **interval,
                            "converged": result.converged,
                            "n_train_per_domain": min(len(train_r), len(train_s)),
                            "n_validation_per_domain": min(len(validation_r), len(validation_s)),
                        }
                    )

    with (args.output_dir / "domain_probe_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    (args.output_dir / "matching_report.json").write_text(
        json.dumps(matching_reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "feature_importance.json").write_text(
        json.dumps(importance_reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    primary = next(
        row
        for row in result_rows
        if row["simulation_source"] == "analytic"
        and row["match_state"] == "matched"
        and row["feature_set"] == "signal_summary"
        and row["model"] == "linear"
    )
    auc = float(primary["validation_roc_auc"])
    analytic_support = bool(matching_reports["analytic"]["common_support_sufficient"])
    if not analytic_support:
        decision = "major_simulator_mismatch_no_common_support"
    elif auc <= 0.70:
        decision = "no_simulator_branch_prioritize_collapse_and_spectral_ssl"
    elif auc <= 0.85:
        decision = "authorize_one_measured_simulator_calibration_ablation"
    else:
        decision = "major_simulator_mismatch_correct_before_extended_real_adaptation"
    summary = {
        "schema_version": 1,
        "status": "complete",
        "sealed_splits_used": [],
        "primary_conditional_auc": auc if analytic_support else None,
        "primary_conditional_auc_ci_95": (
            [primary["ci_95_low"], primary["ci_95_high"]] if analytic_support else None
        ),
        "exploratory_matched_subset_auc": auc,
        "analytic_common_support_sufficient": analytic_support,
        "analytic_common_support_conclusion": (
            "Conditional AUC is not interpretable because observable overlap and balance failed."
            if not analytic_support
            else "Conditional AUC is interpretable under the frozen support rule."
        ),
        "triage_decision": decision,
        "analytic_retains_controlled_factors": True,
        "template_comparator_retains_controlled_factors": False,
        "historical_context": {
            "moment_visual_overlap_but_domain_auc": 0.9907,
            "conv1d_domain_auc": 0.9896,
            "real_vs_real_auc": "approximately 0.52",
            "contract": "obsolete 2.048 ms/window-z-score; diagnostic context only",
        },
    }
    (args.output_dir / "domain_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    run = {
        "schema_version": 1,
        "run_id": args.run_id,
        "project": "unsupervised-learning-flow-cytometry",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "audit_yeast_followup_domain_gap.py",
        "dataset": "yeast-events-followup@v2 + yeast-passage-simulations@v2 + yeast-template-comparator@v2",
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
