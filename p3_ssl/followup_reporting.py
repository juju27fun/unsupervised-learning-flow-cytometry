from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CELLS = ("R0", "R1", "R2", "R3")
CELL_COLORS = {"R0": "#4C78A8", "R1": "#59A14F", "R2": "#F28E2B", "R3": "#E15759"}


def _checkpoint_by_cell_seed(payload: dict[str, Any]) -> dict[tuple[str, int], tuple[str, dict[str, Any]]]:
    output = {}
    for name, metadata in payload["checkpoint_results"].items():
        key = (metadata["cell"], int(metadata["seed"]))
        if key in output:
            raise ValueError(f"Duplicate checkpoint cell/seed: {key}")
        output[key] = (name, metadata)
    expected = {(cell, seed) for cell in CELLS for seed in (42, 43, 44)}
    if set(output) != expected:
        raise ValueError(f"Incomplete R0-R3 matrix: missing={sorted(expected - set(output))}")
    return output


def _probe_lookup(payload: dict[str, Any]) -> dict[tuple[str, int, str, str, float, int], dict[str, Any]]:
    output = {}
    for row in payload["probe_results"]:
        key = (
            row["cell"],
            int(row["representation_seed"]),
            row["method"],
            row["probe"],
            float(row["label_fraction"]),
            int(row["probe_seed"]),
        )
        if key in output:
            raise ValueError(f"Duplicate probe row: {key}")
        output[key] = row
    return output


def _hierarchical_paired_interval(
    differences: dict[int, list[float]], *, repeats: int, seed: int
) -> dict[str, Any]:
    representation_seeds = np.asarray(sorted(differences))
    if representation_seeds.size != 3 or any(not differences[int(value)] for value in representation_seeds):
        raise ValueError("Paired uncertainty requires all three representation seeds")
    rng = np.random.default_rng(seed)
    bootstraps = []
    for _ in range(repeats):
        sampled_seeds = rng.choice(representation_seeds, representation_seeds.size, replace=True)
        sampled_values = []
        for representation_seed in sampled_seeds:
            candidates = np.asarray(differences[int(representation_seed)], dtype=float)
            sampled_values.extend(rng.choice(candidates, len(candidates), replace=True).tolist())
        bootstraps.append(float(np.mean(sampled_values)))
    low, high = np.quantile(bootstraps, (0.025, 0.975))
    observed = np.concatenate([np.asarray(values, dtype=float) for values in differences.values()])
    return {
        "mean_difference": float(observed.mean()),
        "ci95": [float(low), float(high)],
        "n_representation_seeds": 3,
        "n_paired_values": int(observed.size),
        "repeats": repeats,
    }


def _seed_interval(differences: dict[int, float], *, repeats: int, seed: int) -> dict[str, Any]:
    nested = {key: [value] for key, value in differences.items()}
    return _hierarchical_paired_interval(nested, repeats=repeats, seed=seed)


def _label_auc(rows: list[dict[str, Any]], cell: str, representation_seed: int, probe_seed: int) -> float:
    selected = sorted(
        (
            row for row in rows
            if row["cell"] == cell
            and int(row["representation_seed"]) == representation_seed
            and row["method"] == "learned"
            and row["probe"] == "linear"
            and int(row["probe_seed"]) == probe_seed
        ),
        key=lambda row: float(row["label_fraction"]),
    )
    fractions = np.asarray([float(row["label_fraction"]) for row in selected])
    scores = np.asarray([float(row["macro_f1"]) for row in selected])
    return float(np.trapezoid(scores, fractions) / (fractions[-1] - fractions[0]))


def summarize_week2(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    checkpoints = _checkpoint_by_cell_seed(payload)
    probes = _probe_lookup(payload)
    uncertainty = config["evaluation"]["paired_uncertainty"]
    repeats = int(uncertainty["repeats"])
    bootstrap_seed = int(uncertainty["seed"])
    rows = []
    for cell in CELLS:
        for representation_seed in (42, 43, 44):
            _, metadata = checkpoints[(cell, representation_seed)]
            f1_values = [
                probes[(cell, representation_seed, "learned", "linear", 0.10, probe_seed)][
                    "macro_f1"
                ]
                for probe_seed in (42, 43, 44)
            ]
            fusion_deltas = [
                probes[
                    (cell, representation_seed, "handcrafted_plus_learned", "linear", 0.10, probe_seed)
                ]["macro_f1"]
                - probes[(cell, representation_seed, "handcrafted", "linear", 0.10, probe_seed)][
                    "macro_f1"
                ]
                for probe_seed in (42, 43, 44)
            ]
            auc_values = [
                _label_auc(payload["probe_results"], cell, representation_seed, probe_seed)
                for probe_seed in (42, 43, 44)
            ]
            rows.append(
                {
                    "cell": cell,
                    "representation_seed": representation_seed,
                    "converged": bool(metadata["training_convergence"]["converged"]),
                    "effective_rank": float(
                        metadata["real_validation_embedding_health"]["effective_rank"]
                    ),
                    "mean_dimension_std": float(
                        metadata["real_validation_embedding_health"]["mean_dimension_std"]
                    ),
                    "mean_absolute_off_diagonal_covariance": float(
                        metadata["real_validation_embedding_health"][
                            "mean_absolute_off_diagonal_covariance"
                        ]
                    ),
                    "mean_cosine_similarity": float(
                        metadata["real_validation_embedding_health"][
                            "mean_off_diagonal_cosine_similarity"
                        ]
                    ),
                    "macro_f1_10pct_mean": float(np.mean(f1_values)),
                    "macro_f1_10pct_std_probe_seed": float(np.std(f1_values)),
                    "label_efficiency_auc_mean": float(np.mean(auc_values)),
                    "handcrafted_fusion_delta_mean": float(np.mean(fusion_deltas)),
                    "continuous_retention_mean": float(
                        metadata["mean_continuous_relative_mse_reduction"]
                    ),
                    "component_balanced_accuracy": float(
                        metadata["component_count_balanced_accuracy"]
                    ),
                    "retrieval_topk_purity": float(
                        metadata["cross_recording_retrieval"]["topk_label_purity"]
                    ),
                    "runtime_seconds": float(metadata["training_runtime"]["wall_seconds"]),
                    "optimizer_steps": int(metadata["training_runtime"]["optimizer_steps"]),
                }
            )

    f1_differences: dict[int, list[float]] = defaultdict(list)
    fusion_differences: dict[int, list[float]] = defaultdict(list)
    auc_differences: dict[int, list[float]] = defaultdict(list)
    for representation_seed in (42, 43, 44):
        for probe_seed in (42, 43, 44):
            for method, destination in (
                ("learned", f1_differences),
                ("handcrafted_plus_learned", fusion_differences),
            ):
                r3 = probes[("R3", representation_seed, method, "linear", 0.10, probe_seed)][
                    "macro_f1"
                ]
                r0 = probes[("R0", representation_seed, method, "linear", 0.10, probe_seed)][
                    "macro_f1"
                ]
                if method == "handcrafted_plus_learned":
                    handcrafted = probes[
                        ("R3", representation_seed, "handcrafted", "linear", 0.10, probe_seed)
                    ]["macro_f1"]
                    destination[representation_seed].append((r3 - handcrafted) - (r0 - handcrafted))
                else:
                    destination[representation_seed].append(r3 - r0)
            auc_differences[representation_seed].append(
                _label_auc(payload["probe_results"], "R3", representation_seed, probe_seed)
                - _label_auc(payload["probe_results"], "R0", representation_seed, probe_seed)
            )
    retention_differences = {
        seed: checkpoints[("R3", seed)][1]["mean_continuous_relative_mse_reduction"]
        - checkpoints[("R0", seed)][1]["mean_continuous_relative_mse_reduction"]
        for seed in (42, 43, 44)
    }
    component_differences = {
        seed: checkpoints[("R3", seed)][1]["component_count_balanced_accuracy"]
        - checkpoints[("R0", seed)][1]["component_count_balanced_accuracy"]
        for seed in (42, 43, 44)
    }
    retrieval_differences = {
        seed: checkpoints[("R3", seed)][1]["cross_recording_retrieval"]["topk_label_purity"]
        - checkpoints[("R0", seed)][1]["cross_recording_retrieval"]["topk_label_purity"]
        for seed in (42, 43, 44)
    }
    comparisons = {
        "macro_f1_10pct": _hierarchical_paired_interval(
            f1_differences, repeats=repeats, seed=bootstrap_seed
        ),
        "label_efficiency_auc": _hierarchical_paired_interval(
            auc_differences, repeats=repeats, seed=bootstrap_seed + 1
        ),
        "handcrafted_fusion_delta": _hierarchical_paired_interval(
            fusion_differences, repeats=repeats, seed=bootstrap_seed + 2
        ),
        "continuous_retention": _seed_interval(
            retention_differences, repeats=repeats, seed=bootstrap_seed + 3
        ),
        "component_balanced_accuracy": _seed_interval(
            component_differences, repeats=repeats, seed=bootstrap_seed + 4
        ),
        "retrieval_topk_purity": _seed_interval(
            retrieval_differences, repeats=repeats, seed=bootstrap_seed + 5
        ),
    }
    r0_rank = np.median(
        [checkpoints[("R0", seed)][1]["real_validation_embedding_health"]["effective_rank"] for seed in (42, 43, 44)]
    )
    r3_rank = np.median(
        [checkpoints[("R3", seed)][1]["real_validation_embedding_health"]["effective_rank"] for seed in (42, 43, 44)]
    )
    rank_ratio = float(r3_rank / max(r0_rank, 1.0e-12))
    convergence_pass = all(checkpoints[("R3", seed)][1]["training_convergence"]["converged"] for seed in (42, 43, 44))
    retention_pass = (
        comparisons["continuous_retention"]["ci95"][1] >= 0.0
        and comparisons["component_balanced_accuracy"]["ci95"][1] >= 0.0
    )
    macro_f1_pass = comparisons["macro_f1_10pct"]["ci95"][1] >= 0.0
    positive_candidates = (
        "macro_f1_10pct",
        "label_efficiency_auc",
        "handcrafted_fusion_delta",
        "retrieval_topk_purity",
    )
    positive = [name for name in positive_candidates if comparisons[name]["ci95"][0] > 0.0]
    gate = {
        "effective_rank": {
            "r0_median": float(r0_rank),
            "r3_median": float(r3_rank),
            "ratio": rank_ratio,
            "threshold": float(
                config["evaluation"]["promotion_rules"]["effective_rank_ratio_R3_over_R0_min"]
            ),
            "pass": rank_ratio
            >= float(
                config["evaluation"]["promotion_rules"]["effective_rank_ratio_R3_over_R0_min"]
            ),
        },
        "retained_factors_pass": retention_pass,
        "macro_f1_preserved": macro_f1_pass,
        "positive_metrics_with_ci_above_zero": positive,
        "positive_metric_pass": bool(positive),
        "all_r3_seeds_converged": convergence_pass,
    }
    gate["r3_promoted"] = bool(
        gate["effective_rank"]["pass"]
        and retention_pass
        and macro_f1_pass
        and positive
        and convergence_pass
    )
    return {
        "protocol": payload["protocol"],
        "rows": rows,
        "paired_R3_minus_R0": comparisons,
        "gate": gate,
        "week3_quality_adaptation_authorized": gate["r3_promoted"],
        "week3_simulator_correction_required_before_transfer": True,
        "sealed_splits_used": [],
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_week2(summary: dict[str, Any], output_prefix: Path) -> list[Path]:
    rows = summary["rows"]
    metrics = (
        ("effective_rank", "Effective rank"),
        ("macro_f1_10pct_mean", "Macro F1 at 10% labels"),
        ("continuous_retention_mean", "Mean retained-factor gain"),
        ("handcrafted_fusion_delta_mean", "Fusion delta macro F1"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    rng = np.random.default_rng(20260716)
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        for index, cell in enumerate(CELLS):
            values = np.asarray([row[metric] for row in rows if row["cell"] == cell])
            jitter = rng.uniform(-0.06, 0.06, len(values))
            axis.scatter(
                np.full(len(values), index) + jitter,
                values,
                color=CELL_COLORS[cell],
                edgecolor="white",
                linewidth=0.6,
                s=48,
                zorder=3,
            )
            axis.plot(index, np.median(values), marker="_", color="black", markersize=18, zorder=4)
        axis.set_xticks(range(len(CELLS)), CELLS)
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        if "delta" in metric or "retention" in metric:
            axis.axhline(0.0, color="#777777", linewidth=0.8)
    fig.suptitle("Week 2 collapse-aware spectral SSL ablation", fontsize=13)
    outputs = []
    for suffix in (".png", ".pdf"):
        path = output_prefix.with_suffix(suffix)
        fig.savefig(path, dpi=220 if suffix == ".png" else None)
        outputs.append(path)
    plt.close(fig)
    return outputs


def write_decision_markdown(path: Path, summary: dict[str, Any]) -> None:
    gate = summary["gate"]
    decision = "PROMOTE R3" if gate["r3_promoted"] else "STOP R3 EXTENSION"
    lines = [
        "# Yeast SSL Week 2 Decision",
        "",
        f"**Decision:** {decision}",
        "",
        "This decision uses only `followup_validation`; no final split was opened.",
        "",
        "## Frozen gate",
        "",
        "| Criterion | Result | Pass |",
        "|---|---:|:---:|",
        f"| Median effective-rank ratio R3/R0 | {gate['effective_rank']['ratio']:.3f} | {'yes' if gate['effective_rank']['pass'] else 'no'} |",
        f"| Retained factors not degraded beyond uncertainty | n/a | {'yes' if gate['retained_factors_pass'] else 'no'} |",
        f"| 10%-label macro F1 not degraded beyond uncertainty | n/a | {'yes' if gate['macro_f1_preserved'] else 'no'} |",
        f"| At least one positive metric with CI above zero | {', '.join(gate['positive_metrics_with_ci_above_zero']) or 'none'} | {'yes' if gate['positive_metric_pass'] else 'no'} |",
        f"| All R3 seeds converged | n/a | {'yes' if gate['all_r3_seeds_converged'] else 'no'} |",
        "",
        "## Consequence",
        "",
    ]
    if gate["r3_promoted"]:
        lines.append("Week 3 quality-balanced adaptation is authorized, after the required targeted simulator correction.")
    else:
        lines.append("The expensive R3 transfer extension is not authorized. Week 3 is limited to the separately required targeted simulator correction and controlled interpretation of the negative result.")
    lines.extend(
        [
            "",
            "The source-group endpoint remains an acquisition-condition proxy; it is not a yeast-morphology claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
