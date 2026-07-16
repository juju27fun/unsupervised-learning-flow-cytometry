from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .followup_domain import (
    OBSERVABLE_NAMES,
    fit_domain_probe,
    matched_pairs,
    signal_observables,
    signal_summary_features,
)
from .followup_features import load_followup_development


SOURCE_COLORS = {"baseline_v1": "#4C78A8", "corrected_v2": "#E15759"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _balanced_indices(length: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(length, size=maximum, replace=False))


def _load_synthetic(
    root: Path, split: str, *, maximum: int, seed: int
) -> tuple[np.ndarray, list[dict[str, str]]]:
    rows = [row for row in _read_csv(root / "simulation_metadata.csv") if row["split"] == split]
    rows = [row for row in rows if int(row.get("view_index", 0)) == 0]
    selected = _balanced_indices(len(rows), maximum, seed)
    chosen = [rows[int(index)] for index in selected]
    signals = np.load(root / "signals.npy", mmap_mode="r")
    values = np.asarray([signals[int(row["signal_row"])] for row in chosen], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite synthetic signals in {root} / {split}")
    return values, chosen


def _synthetic_group(row: dict[str, str]) -> str:
    return row.get("latent_id") or str(row["signal_row"])


def _group_bootstrap_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    mappings = {}
    for domain in (0, 1):
        keys = sorted(set(groups[labels == domain].tolist()))
        mappings[domain] = {
            key: np.flatnonzero((labels == domain) & (groups == key)) for key in keys
        }
    values = []
    for _ in range(repetitions):
        indices = []
        for domain in (0, 1):
            mapping = mappings[domain]
            keys = np.asarray(list(mapping))
            sampled = rng.choice(keys, size=len(keys), replace=True)
            indices.extend(mapping[key] for key in sampled)
        selected = np.concatenate(indices)
        values.append(float(roc_auc_score(labels[selected], probabilities[selected])))
    low, high = np.quantile(values, (0.025, 0.975))
    return {"ci_95_low": float(low), "ci_95_high": float(high)}


def _support_pass(report: dict[str, Any], gate: dict[str, float]) -> bool:
    return bool(
        report["train"]["real_retained_fraction"]
        >= gate["minimum_retained_fraction_train"]
        and report["validation"]["real_retained_fraction"]
        >= gate["minimum_retained_fraction_validation"]
        and report["train"]["post_match_smd_max"]
        <= gate["maximum_post_match_smd_train"]
        and report["validation"]["post_match_smd_max"]
        <= gate["maximum_post_match_smd_validation"]
    )


def _observable_distribution(values: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: {
            key: float(value)
            for key, value in zip(
                ("p05", "p50", "p95"), np.quantile(values[:, index], (0.05, 0.50, 0.95))
            )
        }
        for index, name in enumerate(OBSERVABLE_NAMES)
    }


def _evaluate_source(
    *,
    source_name: str,
    simulation_root: Path,
    real_sets: dict[str, tuple[np.ndarray, list[dict[str, str]]]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, float]]]:
    started = time.perf_counter()
    evaluation = config["evaluation"]
    seeds = evaluation["sample_seeds"]
    synthetic_sets = {
        "train": _load_synthetic(
            simulation_root,
            "train",
            maximum=int(evaluation["max_train_per_domain"]),
            seed=int(seeds["simulation_train"]),
        ),
        "validation": _load_synthetic(
            simulation_root,
            "validation",
            maximum=int(evaluation["max_validation_per_domain"]),
            seed=int(seeds["simulation_validation"]),
        ),
    }
    real_observables = {
        split: signal_observables(values[0]) for split, values in real_sets.items()
    }
    synthetic_observables = {
        split: signal_observables(values[0]) for split, values in synthetic_sets.items()
    }
    scaler = StandardScaler().fit(
        np.concatenate([real_observables["train"], synthetic_observables["train"]])
    )
    calipers = [float(evaluation["primary_caliper"]), *map(float, evaluation["sensitivity_calipers"])]
    matching = {}
    primary_indices = None
    for caliper in dict.fromkeys(calipers):
        try:
            train_real, train_synthetic, train_report = matched_pairs(
                real_observables["train"],
                synthetic_observables["train"],
                scaler=scaler,
                caliper=caliper,
            )
            validation_real, validation_synthetic, validation_report = matched_pairs(
                real_observables["validation"],
                synthetic_observables["validation"],
                scaler=scaler,
                caliper=caliper,
            )
            matching[f"{caliper:.2f}"] = {
                "status": "ok",
                "train": train_report,
                "validation": validation_report,
            }
            if caliper == float(evaluation["primary_caliper"]):
                primary_indices = (
                    train_real,
                    train_synthetic,
                    validation_real,
                    validation_synthetic,
                )
        except ValueError as error:
            matching[f"{caliper:.2f}"] = {"status": "no_pairs", "error": str(error)}
    if primary_indices is None:
        raise ValueError(f"No matched pairs at the primary caliper for {source_name}")
    primary = matching[f"{float(evaluation['primary_caliper']):.2f}"]
    primary["support_pass"] = _support_pass(primary, evaluation["support_gate"])

    real_summary = {
        split: signal_summary_features(values[0]) for split, values in real_sets.items()
    }
    synthetic_summary = {
        split: signal_summary_features(values[0]) for split, values in synthetic_sets.items()
    }
    feature_sets = {
        "observables": (
            real_observables,
            synthetic_observables,
            list(OBSERVABLE_NAMES),
        ),
        "signal_summary": (
            {split: values[0] for split, values in real_summary.items()},
            {split: values[0] for split, values in synthetic_summary.items()},
            real_summary["train"][1],
        ),
        "downsampled_signal": (
            {split: values[0][:, ::16] for split, values in real_sets.items()},
            {split: values[0][:, ::16] for split, values in synthetic_sets.items()},
            [f"sample_{index * 16}" for index in range(real_sets["train"][0][:, ::16].shape[1])],
        ),
    }
    train_real_match, train_synthetic_match, validation_real_match, validation_synthetic_match = (
        primary_indices
    )
    probe_rows = []
    importance = {}
    repetitions = int(evaluation["domain_probe"]["grouped_bootstrap_repetitions"])
    bootstrap_seed = int(evaluation["domain_probe"]["grouped_bootstrap_seed"])
    for match_state in ("unmatched", "matched"):
        for feature_set in evaluation["domain_probe"]["feature_sets"]:
            real_features, synthetic_features, feature_names = feature_sets[feature_set]
            train_real = real_features["train"]
            train_synthetic = synthetic_features["train"]
            validation_real = real_features["validation"]
            validation_synthetic = synthetic_features["validation"]
            real_rows = real_sets["validation"][1]
            synthetic_rows = synthetic_sets["validation"][1]
            if match_state == "matched":
                train_real = train_real[train_real_match]
                train_synthetic = train_synthetic[train_synthetic_match]
                validation_real = validation_real[validation_real_match]
                validation_synthetic = validation_synthetic[validation_synthetic_match]
                real_rows = [real_rows[int(index)] for index in validation_real_match]
                synthetic_rows = [synthetic_rows[int(index)] for index in validation_synthetic_match]
            for model in evaluation["domain_probe"]["models"]:
                result = fit_domain_probe(
                    train_real,
                    train_synthetic,
                    validation_real,
                    validation_synthetic,
                    feature_names=feature_names,
                    model=model,
                    seed=42,
                    compute_importance=feature_set != "downsampled_signal",
                )
                labels = np.concatenate(
                    [
                        np.zeros(len(validation_real), dtype=np.int64),
                        np.ones(len(validation_synthetic), dtype=np.int64),
                    ]
                )
                groups = np.asarray(
                    [row["capture_block_id"] for row in real_rows]
                    + [_synthetic_group(row) for row in synthetic_rows]
                )
                interval = _group_bootstrap_auc(
                    labels,
                    result.probabilities,
                    groups,
                    seed=bootstrap_seed,
                    repetitions=repetitions,
                )
                key = f"{match_state}:{feature_set}:{model}"
                importance[key] = dict(list(result.importance.items())[:15])
                probe_rows.append(
                    {
                        "simulation_source": source_name,
                        "match_state": match_state,
                        "feature_set": feature_set,
                        "model": model,
                        "validation_roc_auc": result.roc_auc,
                        **interval,
                        "converged": result.converged,
                        "n_train_per_domain": min(len(train_real), len(train_synthetic)),
                        "n_validation_per_domain": min(
                            len(validation_real), len(validation_synthetic)
                        ),
                        "conditional_interpretation_valid": bool(
                            match_state == "matched" and primary["support_pass"]
                        ),
                    }
                )
    source_result = {
        "simulation_root": str(simulation_root),
        "matching_by_caliper": matching,
        "observable_distributions": {
            split: {
                "real": _observable_distribution(real_observables[split]),
                "synthetic": _observable_distribution(synthetic_observables[split]),
            }
            for split in ("train", "validation")
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    return source_result, probe_rows, importance


def evaluate_week3(
    *,
    real_root: Path,
    baseline_root: Path,
    corrected_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    evaluation = config["evaluation"]
    seeds = evaluation["sample_seeds"]
    real = load_followup_development(real_root)
    real_sets = {}
    for split, indices, maximum, seed_name in (
        ("train", real.train_indices, evaluation["max_train_per_domain"], "real_train"),
        (
            "validation",
            real.validation_indices,
            evaluation["max_validation_per_domain"],
            "real_validation",
        ),
    ):
        selected = indices[
            _balanced_indices(len(indices), int(maximum), int(seeds[seed_name]))
        ]
        real_sets[split] = (
            real.signals[selected],
            [real.rows[int(index)] for index in selected],
        )
    source_results = {}
    probe_rows = []
    importance = {}
    for source_name, root in (("baseline_v1", baseline_root), ("corrected_v2", corrected_root)):
        result, rows, source_importance = _evaluate_source(
            source_name=source_name,
            simulation_root=root,
            real_sets=real_sets,
            config=config,
        )
        source_results[source_name] = result
        probe_rows.extend(rows)
        importance[source_name] = source_importance
    primary_caliper = f"{float(evaluation['primary_caliper']):.2f}"
    baseline_primary = source_results["baseline_v1"]["matching_by_caliper"][primary_caliper]
    corrected_primary = source_results["corrected_v2"]["matching_by_caliper"][primary_caliper]
    primary_probe = next(
        row
        for row in probe_rows
        if row["simulation_source"] == "corrected_v2"
        and row["match_state"] == "matched"
        and row["feature_set"] == evaluation["domain_probe"]["primary_feature_set"]
        and row["model"] == evaluation["domain_probe"]["primary_model"]
    )
    support_pass = bool(corrected_primary["support_pass"])
    primary_auc = float(primary_probe["validation_roc_auc"])
    bands = evaluation["domain_probe"]["interpretation_bands"]
    if not support_pass:
        decision = "stop_common_support_failed"
        valid_auc = None
    elif primary_auc <= float(bands["future_bridge_plausible_max_auc"]):
        decision = "future_simulator_bridge_plausible_no_current_retraining"
        valid_auc = primary_auc
    elif primary_auc <= float(bands["residual_gap_max_auc"]):
        decision = "residual_domain_gap_no_current_retraining"
        valid_auc = primary_auc
    else:
        decision = "stop_major_domain_gap"
        valid_auc = primary_auc
    comparison = {
        "train_retained_fraction_change": corrected_primary["train"]["real_retained_fraction"]
        - baseline_primary["train"]["real_retained_fraction"],
        "validation_retained_fraction_change": corrected_primary["validation"][
            "real_retained_fraction"
        ]
        - baseline_primary["validation"]["real_retained_fraction"],
        "train_max_smd_change": corrected_primary["train"]["post_match_smd_max"]
        - baseline_primary["train"]["post_match_smd_max"],
        "validation_max_smd_change": corrected_primary["validation"]["post_match_smd_max"]
        - baseline_primary["validation"]["post_match_smd_max"],
    }
    return {
        "schema_version": 1,
        "protocol": config["study"]["protocol"],
        "status": "complete",
        "datasets": {
            "real": config["study"]["real_dataset"],
            "baseline": config["study"]["baseline_simulation_dataset"],
            "corrected": config["study"]["corrected_simulation_dataset"],
        },
        "real_samples": {split: len(values[0]) for split, values in real_sets.items()},
        "source_results": source_results,
        "probe_results": probe_rows,
        "feature_importance": importance,
        "comparison_corrected_minus_baseline": comparison,
        "decision": {
            "classification": decision,
            "corrected_common_support_pass": support_pass,
            "primary_conditional_auc": valid_auc,
            "exploratory_matched_signal_summary_linear_auc": primary_auc,
            "representation_training_authorized": False,
            "followup_test_opening_authorized": False,
        },
        "sealed_splits_used": [],
        "runtime_seconds": time.perf_counter() - started,
    }


def plot_week3(payload: dict[str, Any], output_prefix: Path) -> list[Path]:
    primary = "1.50"
    sources = ("baseline_v1", "corrected_v2")
    labels = ("v1 Gaussian", "v2 finite support")
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), constrained_layout=True)
    x = np.arange(2)
    width = 0.34
    for source_index, (source, label) in enumerate(zip(sources, labels)):
        report = payload["source_results"][source]["matching_by_caliper"][primary]
        offset = (source_index - 0.5) * width
        axes[0, 0].bar(
            x + offset,
            [report[split]["real_retained_fraction"] for split in ("train", "validation")],
            width,
            label=label,
            color=SOURCE_COLORS[source],
        )
        axes[0, 1].bar(
            x + offset,
            [report[split]["post_match_smd_max"] for split in ("train", "validation")],
            width,
            label=label,
            color=SOURCE_COLORS[source],
        )
    axes[0, 0].axhline(0.50, color="#555555", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("Retained fraction")
    axes[0, 1].axhline(0.25, color="#555555", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("Maximum post-match SMD")
    for axis in axes[0]:
        axis.set_xticks(x, ("Train", "Validation"))
    axes[0, 1].legend(frameon=False, fontsize=8)

    observable_x = np.arange(len(OBSERVABLE_NAMES))
    for source_index, (source, label) in enumerate(zip(sources, labels)):
        report = payload["source_results"][source]["matching_by_caliper"][primary]
        axes[1, 0].bar(
            observable_x + (source_index - 0.5) * width,
            report["validation"]["post_match_smd"],
            width,
            label=label,
            color=SOURCE_COLORS[source],
        )
    axes[1, 0].axhline(0.25, color="#555555", linestyle="--", linewidth=1)
    axes[1, 0].set_xticks(
        observable_x,
        ("Duration", "Frequency", "RMS", "SNR", "Peak count", "Offset"),
        rotation=25,
        ha="right",
    )
    axes[1, 0].set_ylabel("Validation post-match SMD")

    feature_sets = ("observables", "signal_summary", "downsampled_signal")
    feature_x = np.arange(len(feature_sets))
    for source_index, (source, label) in enumerate(zip(sources, labels)):
        values = []
        for feature_set in feature_sets:
            row = next(
                item
                for item in payload["probe_results"]
                if item["simulation_source"] == source
                and item["match_state"] == "matched"
                and item["feature_set"] == feature_set
                and item["model"] == "linear"
            )
            values.append(row["validation_roc_auc"])
        axes[1, 1].bar(
            feature_x + (source_index - 0.5) * width,
            values,
            width,
            label=label,
            color=SOURCE_COLORS[source],
        )
    axes[1, 1].axhline(0.50, color="#555555", linewidth=1)
    axes[1, 1].axhline(0.70, color="#777777", linestyle="--", linewidth=1)
    axes[1, 1].axhline(0.85, color="#777777", linestyle=":", linewidth=1)
    axes[1, 1].set_xticks(feature_x, ("Observables", "Signal summary", "Downsampled"))
    axes[1, 1].set_ylabel("Exploratory matched ROC AUC")
    axes[1, 1].set_ylim(0.45, 1.02)

    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
    fig.suptitle("Week 3 simulator support correction", fontsize=13)
    outputs = []
    for suffix in (".png", ".pdf"):
        path = output_prefix.with_suffix(suffix)
        fig.savefig(path, dpi=220 if suffix == ".png" else None)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_week3_markdown(path: Path, payload: dict[str, Any]) -> None:
    primary = "1.50"
    decision = payload["decision"]
    lines = [
        "# Yeast SSL Week 3 Simulator Decision",
        "",
        f"**Decision:** `{decision['classification']}`",
        "",
        "Only `followup_train` fitted the correction and only `followup_validation` evaluated it. Final splits remained sealed.",
        "",
        "![Week 3 simulator comparison](week3_simulator_comparison.png)",
        "",
        "## Frozen common-support gate",
        "",
        "| Source | Train retained | Validation retained | Train max SMD | Validation max SMD | Pass |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for source, label in (("baseline_v1", "v1 Gaussian"), ("corrected_v2", "v2 finite support")):
        report = payload["source_results"][source]["matching_by_caliper"][primary]
        lines.append(
            f"| {label} | {report['train']['real_retained_fraction']:.3f} | "
            f"{report['validation']['real_retained_fraction']:.3f} | "
            f"{report['train']['post_match_smd_max']:.3f} | "
            f"{report['validation']['post_match_smd_max']:.3f} | "
            f"{'yes' if report['support_pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Corrected validation residuals",
            "",
            "| Observable | Post-match SMD |",
            "|---|---:|",
        ]
    )
    corrected = payload["source_results"]["corrected_v2"]["matching_by_caliper"][primary]
    lines.extend(
        f"| {name} | {value:.3f} |"
        for name, value in zip(OBSERVABLE_NAMES, corrected["validation"]["post_match_smd"])
    )
    primary_probe = next(
        row
        for row in payload["probe_results"]
        if row["simulation_source"] == "corrected_v2"
        and row["match_state"] == "matched"
        and row["feature_set"] == "signal_summary"
        and row["model"] == "linear"
    )
    lines.extend(
        [
            "",
            "## Domain separability",
            "",
            f"Corrected matched signal-summary linear AUC: {primary_probe['validation_roc_auc']:.3f} "
            f"[95% {primary_probe['ci_95_low']:.3f}, {primary_probe['ci_95_high']:.3f}].",
            "",
        ]
    )
    if not decision["corrected_common_support_pass"]:
        lines.append(
            "This AUC is exploratory because the frozen common-support gate failed; it cannot be interpreted as a conditional domain estimate."
        )
    lines.extend(
        [
            "",
            "## Consequence",
            "",
            "Representation retraining and final-test opening remain unauthorized. The result diagnoses simulator adequacy for future work only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
