from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "rms": "RMS",
    "raw": "Raw signal",
    "handcrafted": "Handcrafted",
    "moment": "MOMENT",
    "patchtst": "PatchTST",
    "random": "Random encoder",
    "conv1d": "Supervised Conv1D",
    "A1": "A1 real SSL",
    "A2": "A2 synthetic reconstruction",
    "A3": "A3 physics-informed",
    "A4": "A4 physics + adaptation",
}
COLORS = {
    "rms": "#999999",
    "raw": "#332288",
    "handcrafted": "#0072B2",
    "moment": "#E69F00",
    "patchtst": "#AA4499",
    "random": "#7F7F7F",
    "conv1d": "#117733",
    "A1": "#009E73",
    "A2": "#CC79A7",
    "A3": "#D55E00",
    "A4": "#56B4E9",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _representation_seed(payload: dict[str, Any], row: dict[str, Any]) -> int:
    checkpoint = row.get("checkpoint")
    metadata = payload.get("checkpoint_metadata", {}).get(checkpoint, {})
    return int(metadata.get("seed", 0))


def _matching_rows(
    payload: dict[str, Any], key: str, value: str, fraction: float
) -> list[dict[str, Any]]:
    return [
        row
        for row in payload["results"]
        if row[key] == value and float(row["label_fraction"]) == fraction
    ]


def paired_bootstrap_comparisons(
    a0: dict[str, Any], checkpoints: dict[str, Any]
) -> list[dict[str, Any]]:
    comparisons = (
        ("A4", "handcrafted", checkpoints, "cell", a0, "method"),
        ("A4", "moment", checkpoints, "cell", a0, "method"),
        ("A4", "A3", checkpoints, "cell", checkpoints, "cell"),
        ("A3", "A2", checkpoints, "cell", checkpoints, "cell"),
    )
    output = []
    for left, right, left_payload, left_key, right_payload, right_key in comparisons:
        left_rows = _matching_rows(left_payload, left_key, left, 0.1)
        right_rows = _matching_rows(right_payload, right_key, right, 0.1)
        paired_differences = []
        run_differences = []
        for left_row in left_rows:
            probe_seed = int(left_row.get("seed", 0))
            representation_seed = _representation_seed(left_payload, left_row)
            candidates = [
                row
                for row in right_rows
                if int(row.get("seed", 0)) == probe_seed
                and (
                    right_key != "cell"
                    or _representation_seed(right_payload, row) == representation_seed
                )
            ]
            if not candidates and len(right_rows) == 1:
                candidates = right_rows
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected one paired row for {left}/{right}, representation seed "
                    f"{representation_seed}, probe seed {probe_seed}; got {len(candidates)}"
                )
            right_row = candidates[0]
            left_values = np.asarray(
                left_row["grouped_bootstrap"]["metrics"]["macro_f1"]["replicates"]
            )
            right_values = np.asarray(
                right_row["grouped_bootstrap"]["metrics"]["macro_f1"]["replicates"]
            )
            if left_values.shape != right_values.shape:
                raise ValueError("Paired bootstrap arrays have different shapes")
            paired_differences.append(left_values - right_values)
            run_differences.append(float(left_row["macro_f1"] - right_row["macro_f1"]) if "macro_f1" in left_row else float(np.mean(left_values - right_values)))
        direct_mean = float(np.mean(run_differences))
        repeats = max(len(values) for values in paired_differences)
        rng = np.random.default_rng(20260715)
        hierarchical = np.empty(repeats, dtype=np.float64)
        for repeat in range(repeats):
            sampled_runs = rng.integers(0, len(paired_differences), size=len(paired_differences))
            hierarchical[repeat] = np.mean(
                [
                    paired_differences[index][
                        rng.integers(0, len(paired_differences[index]))
                    ]
                    for index in sampled_runs
                ]
            )
        interval = np.quantile(hierarchical, [0.025, 0.975])
        output.append(
            {
                "comparison": f"{left} - {right}",
                "left": left,
                "right": right,
                "mean_difference": direct_mean,
                "ci_95_low": float(interval[0]),
                "ci_95_high": float(interval[1]),
                "bootstrap_probability_gt_zero": float(np.mean(hierarchical > 0.0)),
                "n_repeats": int(hierarchical.size),
                "n_paired_runs": len(run_differences),
                "paired_run_mean_difference": direct_mean,
                "paired_run_std": (
                    float(np.std(run_differences, ddof=1))
                    if len(run_differences) > 1
                    else 0.0
                ),
                "uncertainty_method": (
                    "hierarchical paired bootstrap over representation/probe runs "
                    "and capture-block bootstrap replicates"
                ),
            }
        )
    return output


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [pdf.name, png.name]


def _plot_aggregate_curve(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    name: str,
) -> None:
    fractions = sorted({float(row["label_fraction"]) for row in rows})
    means = []
    deviations = []
    for fraction in fractions:
        values = np.asarray(
            [float(row["macro_f1"]) for row in rows if float(row["label_fraction"]) == fraction]
        )
        means.append(float(values.mean()))
        deviations.append(float(values.std(ddof=1)) if values.size > 1 else 0.0)
    ax.errorbar(
        [100.0 * value for value in fractions],
        means,
        yerr=deviations,
        marker="o",
        capsize=2,
        color=COLORS[name],
        label=DISPLAY[name],
    )


def _label_efficiency_figure(a0: dict[str, Any], checkpoints: dict[str, Any], output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for name in ("rms", "raw", "handcrafted", "random", "moment", "patchtst", "conv1d"):
        rows = [row for row in a0["results"] if row["method"] == name]
        _plot_aggregate_curve(ax, rows, name=name)
    for name in ("A1", "A2", "A3", "A4"):
        rows = [row for row in checkpoints["results"] if row["cell"] == name]
        _plot_aggregate_curve(ax, rows, name=name)
    ax.set_xlabel("Available proxy labels (%)")
    ax.set_ylabel("Development macro F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Development source-condition proxy label efficiency")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, output_dir, "development_label_efficiency")


def _method_summary_rows(
    a0: dict[str, Any], checkpoints: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = [
        {**row, "display_method": DISPLAY[row["method"]]}
        for row in a0["results"]
    ] + [
        {**row, "method": row["cell"], "display_method": DISPLAY[row["cell"]]}
        for row in checkpoints["results"]
    ]
    output = []
    for method in (*a0["methods"], "A1", "A2", "A3", "A4"):
        for fraction in sorted(
            {float(row["label_fraction"]) for row in rows if row["method"] == method}
        ):
            selected = [
                float(row["macro_f1"])
                for row in rows
                if row["method"] == method
                and float(row["label_fraction"]) == fraction
            ]
            output.append(
                {
                    "method": method,
                    "display_method": DISPLAY[method],
                    "label_fraction": fraction,
                    "n_runs": len(selected),
                    "macro_f1_mean": float(np.mean(selected)),
                    "macro_f1_std": (
                        float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0
                    ),
                    "macro_f1_min": float(np.min(selected)),
                    "macro_f1_max": float(np.max(selected)),
                }
            )
    return output


def _checkpoint_diagnostic_rows(checkpoints: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for name, metadata in sorted(checkpoints["checkpoint_metadata"].items()):
        physics = metadata["development_physical_fidelity"]
        retained = physics["retained_factor_linear_probes"]
        robustness = [
            float(value["prediction_agreement"])
            for value in metadata["development_robustness"]["perturbations"].values()
        ]
        output.append(
            {
                "checkpoint": name,
                "cell": metadata["cell"],
                "representation_seed": metadata["seed"],
                "domain_roc_auc": metadata["simulation_real_domain_probe"]["roc_auc"],
                "domain_balanced_accuracy": metadata["simulation_real_domain_probe"][
                    "balanced_accuracy"
                ],
                "real_effective_rank": metadata["real_embedding_health"]["effective_rank"],
                "real_embedding_dimension": metadata["real_embedding_health"][
                    "embedding_dimension"
                ],
                "real_mean_cosine_similarity": metadata["real_embedding_health"][
                    "mean_off_diagonal_cosine_similarity"
                ],
                "retrieval_top1_proxy_purity": metadata["development_retrieval"][
                    "top1_label_purity"
                ],
                "retrieval_top1_quality_purity": metadata["development_retrieval"][
                    "top1_quality_purity"
                ],
                "retained_factor_mean_relative_mse_reduction": float(
                    np.mean(
                        [
                            value["relative_mse_reduction_vs_constant"]
                            for value in retained.values()
                        ]
                    )
                ),
                "component_count_balanced_accuracy": physics["component_count_probe"][
                    "balanced_accuracy"
                ],
                "robustness_prediction_agreement_mean": float(np.mean(robustness)),
                "robustness_prediction_agreement_min": float(np.min(robustness)),
            }
        )
    return output


def _domain_physics_figure(
    checkpoints: dict[str, Any], output_dir: Path
) -> list[str]:
    rows = _checkpoint_diagnostic_rows(checkpoints)
    fig, ax = plt.subplots(figsize=(7.3, 4.7), constrained_layout=True)
    for cell in ("A1", "A2", "A3", "A4"):
        selected = [row for row in rows if row["cell"] == cell]
        physics = np.asarray(
            [row["retained_factor_mean_relative_mse_reduction"] for row in selected]
        )
        domain = np.asarray([row["domain_roc_auc"] for row in selected])
        ax.errorbar(
            float(physics.mean()),
            float(domain.mean()),
            xerr=float(physics.std(ddof=1)),
            yerr=float(domain.std(ddof=1)),
            fmt="o",
            capsize=3,
            color=COLORS[cell],
            markersize=7,
            label=DISPLAY[cell],
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="chance")
    ax.set_xlabel("Mean retained-factor MSE reduction versus constant prior")
    ax.set_ylabel("Simulation-versus-real ROC AUC")
    ax.set_ylim(0.5, 1.01)
    ax.set_title("Physical information versus simulation-real separability")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, output_dir, "development_domain_vs_physics")


def _embedding_health_figure(
    a0: dict[str, Any], checkpoints: dict[str, Any], output_dir: Path
) -> list[str]:
    points: list[tuple[str, list[dict[str, Any]]]] = []
    for name in ("random", "moment", "patchtst"):
        health = a0["method_metadata"][name]["embedding_health"]
        points.append((name, [health]))
    for cell in ("A1", "A2", "A3", "A4"):
        points.append(
            (
                cell,
                [
                    metadata["real_embedding_health"]
                    for metadata in checkpoints["checkpoint_metadata"].values()
                    if metadata["cell"] == cell
                ],
            )
        )
    fig, ax = plt.subplots(figsize=(7.3, 4.7), constrained_layout=True)
    for name, health_rows in points:
        rank_fraction = np.asarray(
            [row["effective_rank"] / row["embedding_dimension"] for row in health_rows]
        )
        cosine = np.asarray(
            [row["mean_off_diagonal_cosine_similarity"] for row in health_rows]
        )
        ax.errorbar(
            float(rank_fraction.mean()),
            float(cosine.mean()),
            xerr=float(rank_fraction.std(ddof=1)) if len(rank_fraction) > 1 else 0.0,
            yerr=float(cosine.std(ddof=1)) if len(cosine) > 1 else 0.0,
            fmt="o",
            capsize=3,
            color=COLORS[name],
            markersize=7,
            label=DISPLAY[name],
        )
    ax.set_xlabel("Effective rank / embedding dimension")
    ax.set_ylabel("Mean off-diagonal cosine similarity")
    ax.set_ylim(0.7, 1.005)
    ax.set_title("Development embedding anisotropy diagnostic")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    return _save(fig, output_dir, "development_embedding_health")


def _class_recall_figure(
    a0: dict[str, Any], checkpoints: dict[str, Any], output_dir: Path
) -> list[str]:
    methods = [*a0["methods"], "A1", "A2", "A3", "A4"]
    rows = []
    for method in methods:
        source = a0 if method in a0["methods"] else checkpoints
        key = "method" if source is a0 else "cell"
        rows.append(
            [
                float(
                    np.mean(
                        [
                            row["per_class_recall"][class_name]
                            for row in source["results"]
                            if row[key] == method
                            and float(row["label_fraction"]) == 0.1
                        ]
                    )
                )
                for class_name in a0["class_names"]
            ]
        )
    values = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(7.6, 5.3), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(a0["class_names"])), a0["class_names"])
    ax.set_yticks(np.arange(len(methods)), [DISPLAY[method] for method in methods])
    ax.set_title("Mean development recall by source-condition proxy at 10% labels")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", color="white" if values[row, column] < 0.55 else "black", fontsize=7)
    fig.colorbar(image, ax=ax, label="recall")
    return _save(fig, output_dir, "development_class_recall")


def _paired_figure(rows: list[dict[str, Any]], output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(7.3, 3.8), constrained_layout=True)
    positions = np.arange(len(rows))
    means = np.asarray([row["mean_difference"] for row in rows])
    lower = means - np.asarray([row["ci_95_low"] for row in rows])
    upper = np.asarray([row["ci_95_high"] for row in rows]) - means
    ax.errorbar(means, positions, xerr=np.vstack([lower, upper]), fmt="o", color="#0072B2", capsize=3)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(positions, [row["comparison"] for row in rows])
    ax.set_xlabel("Paired macro-F1 difference (capture-block bootstrap 95% interval)")
    ax.set_title("Development paired 10%-label comparisons")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, output_dir, "development_paired_differences")


def _retrieval_figure(a0: dict[str, Any], checkpoints: dict[str, Any], output_dir: Path) -> list[str]:
    points = []
    for name in ("handcrafted", "moment", "random"):
        metric = a0["method_metadata"][name]["development_retrieval"]
        points.append((name, metric))
    for key, value in checkpoints["checkpoint_metadata"].items():
        points.append((value["cell"], value["development_retrieval"], value.get("seed")))
    fig, ax = plt.subplots(figsize=(7.0, 4.7), constrained_layout=True)
    for point in points:
        name, metric = point[:2]
        seed = point[2] if len(point) > 2 else None
        ax.scatter(
            metric["top1_quality_purity"],
            metric["top1_label_purity"],
            color=COLORS[name],
            s=48,
        )
        label = DISPLAY[name] if seed is None else f"{name} s{seed}"
        ax.annotate(label, (metric["top1_quality_purity"], metric["top1_label_purity"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axhline(0.25, color="black", linestyle="--", linewidth=0.8, label="four-class chance")
    ax.set_xlabel("Top-1 quality-stratum purity")
    ax.set_ylabel("Top-1 source-condition purity")
    ax.set_title("Cross-recording retrieval: proxy signal versus quality shortcut")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    return _save(fig, output_dir, "development_retrieval")


def _robustness_figure(checkpoints: dict[str, Any], output_dir: Path) -> list[str]:
    cells = sorted(
        checkpoints["checkpoint_metadata"],
        key=lambda key: (
            checkpoints["checkpoint_metadata"][key]["cell"],
            int(checkpoints["checkpoint_metadata"][key].get("seed", 0)),
        ),
    )
    perturbations = list(
        checkpoints["checkpoint_metadata"][cells[0]]["development_robustness"]["perturbations"]
    )
    agreement = np.asarray(
        [
            [
                checkpoints["checkpoint_metadata"][cell]["development_robustness"]["perturbations"][name]["prediction_agreement"]
                for name in perturbations
            ]
            for cell in cells
        ]
    )
    fig, ax = plt.subplots(figsize=(9.0, 3.8), constrained_layout=True)
    image = ax.imshow(agreement, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(perturbations)), perturbations, rotation=35, ha="right")
    ax.set_yticks(
        np.arange(len(cells)),
        [
            f"{checkpoints['checkpoint_metadata'][key]['cell']} s{checkpoints['checkpoint_metadata'][key].get('seed', '?')}"
            for key in cells
        ],
    )
    ax.set_title("Development prediction agreement under bounded perturbations")
    for row in range(agreement.shape[0]):
        for column in range(agreement.shape[1]):
            ax.text(column, row, f"{agreement[row, column]:.2f}", ha="center", va="center", color="white" if agreement[row, column] < 0.7 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="prediction agreement")
    return _save(fig, output_dir, "development_robustness")


def _physical_figure(checkpoints: dict[str, Any], output_dir: Path) -> list[str]:
    cells = sorted(
        checkpoints["checkpoint_metadata"],
        key=lambda key: (
            checkpoints["checkpoint_metadata"][key]["cell"],
            int(checkpoints["checkpoint_metadata"][key].get("seed", 0)),
        ),
    )
    factors = list(
        checkpoints["checkpoint_metadata"][cells[0]]["development_physical_fidelity"]["retained_factor_linear_probes"]
    )
    values = np.asarray(
        [
            [
                checkpoints["checkpoint_metadata"][cell]["development_physical_fidelity"]["retained_factor_linear_probes"][factor]["relative_mse_reduction_vs_constant"]
                for factor in factors
            ]
            for cell in cells
        ]
    )
    fig, ax = plt.subplots(figsize=(9.2, 3.8), constrained_layout=True)
    image = ax.imshow(np.clip(values, -1.0, 1.0), aspect="auto", vmin=-1.0, vmax=1.0, cmap="RdBu")
    ax.set_xticks(np.arange(len(factors)), [factor.replace("_", " ") for factor in factors], rotation=30, ha="right")
    ax.set_yticks(
        np.arange(len(cells)),
        [
            f"{checkpoints['checkpoint_metadata'][key]['cell']} s{checkpoints['checkpoint_metadata'][key].get('seed', '?')}"
            for key in cells
        ],
    )
    ax.set_title("Retained-factor MSE reduction versus constant prior")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="relative MSE reduction")
    return _save(fig, output_dir, "development_physical_fidelity")


def _decision_summary(
    a0: dict[str, Any],
    checkpoints: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = ("rms", "raw", "handcrafted", "random", "moment", "patchtst")
    baseline_means = {
        method: float(
            np.mean(
                [
                    row["macro_f1"]
                    for row in a0["results"]
                    if row["method"] == method
                    and float(row["label_fraction"]) == 0.1
                ]
            )
        )
        for method in eligible
    }
    strongest = max(baseline_means, key=baseline_means.get)
    primary = next(
        row for row in comparisons if row["left"] == "A4" and row["right"] == strongest
    )
    adaptation = next(
        row for row in comparisons if row["left"] == "A4" and row["right"] == "A3"
    )
    physics = {}
    for cell in ("A2", "A4"):
        metadata = [
            value
            for value in checkpoints["checkpoint_metadata"].values()
            if value["cell"] == cell
        ]
        factors = metadata[0]["development_physical_fidelity"][
            "retained_factor_linear_probes"
        ]
        physics[cell] = {
            factor: float(
                np.mean(
                    [
                        value["development_physical_fidelity"][
                            "retained_factor_linear_probes"
                        ][factor]["relative_mse_reduction_vs_constant"]
                        for value in metadata
                    ]
                )
            )
            for factor in factors
        }
    domain_auc = {
        cell: [
            value["simulation_real_domain_probe"]["roc_auc"]
            for value in checkpoints["checkpoint_metadata"].values()
            if value["cell"] == cell
        ]
        for cell in ("A3", "A4")
    }
    minimum_effect = 0.03
    return {
        "schema_version": 1,
        "scope": "single-acquisition development proxy endpoint; not morphology or OOD",
        "minimum_effect_macro_f1": minimum_effect,
        "strongest_eligible_frozen_baseline": strongest,
        "baseline_10pct_macro_f1_means": baseline_means,
        "criteria": {
            "primary_effect_at_least_minimum": bool(
                primary["mean_difference"] >= minimum_effect
            ),
            "primary_interval_excludes_zero": bool(primary["ci_95_low"] > 0.0),
            "a4_improves_over_a3": bool(adaptation["mean_difference"] > 0.0),
            "a4_vs_a3_interval_excludes_zero": bool(adaptation["ci_95_low"] > 0.0),
            "all_retained_factor_means_improve_over_a2": bool(all(
                physics["A4"][factor] > physics["A2"][factor]
                for factor in physics["A2"]
            )),
            "simulation_real_separability_reported": True,
            "adaptation_reduces_mean_domain_auc": bool(
                np.mean(domain_auc["A4"]) < np.mean(domain_auc["A3"])
            ),
        },
        "primary_comparison": primary,
        "adaptation_comparison": adaptation,
        "retained_factor_mean_recovery": physics,
        "simulation_real_domain_auc": domain_auc,
        "promotion_decision": "do_not_promote_a4",
        "decision_reason": (
            "A4 does not exceed the strongest eligible frozen baseline by the "
            "predeclared 0.03 macro-F1 effect and its paired interval includes zero."
        ),
        "interpretation_limits": [
            "The endpoint is a source-condition proxy from one acquisition, not yeast morphology.",
            "No acquisition-OOD test exists, so the decision is in-session only.",
            "Perfect quality-stratum retrieval purity indicates a strong quality shortcut.",
            "High simulation-real AUC indicates that real adaptation did not close the domain gap.",
        ],
    }


def build_development_report(a0_path: Path, checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    a0 = _read_json(a0_path)
    checkpoints = _read_json(checkpoint_path)
    comparisons = paired_bootstrap_comparisons(a0, checkpoints)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "paired_bootstrap_differences.csv", comparisons)
    _write_csv(
        output_dir / "development_method_summary.csv",
        _method_summary_rows(a0, checkpoints),
    )
    _write_csv(
        output_dir / "development_checkpoint_diagnostics.csv",
        _checkpoint_diagnostic_rows(checkpoints),
    )

    probe_rows = []
    for row in a0["results"]:
        probe_rows.append(
            {
                "method": row["method"],
                "representation_seed": "",
                "label_fraction": row["label_fraction"],
                "seed": row["seed"],
                "macro_f1": row["macro_f1"],
                "balanced_accuracy": row["balanced_accuracy"],
                "ece": row.get("calibration", {}).get("expected_calibration_error", ""),
                "brier": row.get("calibration", {}).get("multiclass_brier", ""),
            }
        )
    for row in checkpoints["results"]:
        probe_rows.append(
            {
                "method": row["cell"],
                "representation_seed": _representation_seed(checkpoints, row),
                "label_fraction": row["label_fraction"],
                "seed": row["seed"],
                "macro_f1": row["macro_f1"],
                "balanced_accuracy": row["balanced_accuracy"],
                "ece": row["calibration"]["expected_calibration_error"],
                "brier": row["calibration"]["multiclass_brier"],
            }
        )
    _write_csv(output_dir / "development_probe_metrics.csv", probe_rows)
    decision = _decision_summary(a0, checkpoints, comparisons)
    (output_dir / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs = [
        "paired_bootstrap_differences.csv",
        "development_method_summary.csv",
        "development_checkpoint_diagnostics.csv",
        "development_probe_metrics.csv",
        "decision_summary.json",
    ]
    outputs += _label_efficiency_figure(a0, checkpoints, output_dir)
    outputs += _paired_figure(comparisons, output_dir)
    outputs += _retrieval_figure(a0, checkpoints, output_dir)
    outputs += _robustness_figure(checkpoints, output_dir)
    outputs += _physical_figure(checkpoints, output_dir)
    outputs += _domain_physics_figure(checkpoints, output_dir)
    outputs += _embedding_health_figure(a0, checkpoints, output_dir)
    outputs += _class_recall_figure(a0, checkpoints, output_dir)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "development source-condition proxy labels; single acquisition, not a morphology or OOD result",
        "source_a0": str(a0_path),
        "source_checkpoints": str(checkpoint_path),
        "paired_comparisons": comparisons,
        "decision": decision,
        "outputs": outputs,
        "sealed_splits_used": [],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
