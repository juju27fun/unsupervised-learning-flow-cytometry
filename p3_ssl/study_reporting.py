from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "handcrafted": "Handcrafted",
    "moment": "MOMENT",
    "random": "Random encoder",
    "A1": "A1 real SSL",
    "A2": "A2 synthetic reconstruction",
    "A3": "A3 physics-informed",
    "A4": "A4 physics + adaptation",
}
COLORS = {
    "handcrafted": "#0072B2",
    "moment": "#E69F00",
    "random": "#7F7F7F",
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


def _probe_row(payload: dict[str, Any], key: str, value: str, fraction: float) -> dict[str, Any]:
    return next(
        row
        for row in payload["results"]
        if row[key] == value and float(row["label_fraction"]) == fraction
    )


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
        left_row = _probe_row(left_payload, left_key, left, 0.1)
        right_row = _probe_row(right_payload, right_key, right, 0.1)
        left_values = np.asarray(
            left_row["grouped_bootstrap"]["metrics"]["macro_f1"]["replicates"]
        )
        right_values = np.asarray(
            right_row["grouped_bootstrap"]["metrics"]["macro_f1"]["replicates"]
        )
        if left_values.shape != right_values.shape:
            raise ValueError("Paired bootstrap arrays have different shapes")
        differences = left_values - right_values
        interval = np.quantile(differences, [0.025, 0.975])
        output.append(
            {
                "comparison": f"{left} - {right}",
                "left": left,
                "right": right,
                "mean_difference": float(differences.mean()),
                "ci_95_low": float(interval[0]),
                "ci_95_high": float(interval[1]),
                "bootstrap_probability_gt_zero": float(np.mean(differences > 0.0)),
                "n_repeats": int(differences.size),
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


def _label_efficiency_figure(a0: dict[str, Any], checkpoints: dict[str, Any], output_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for name in ("handcrafted", "moment", "random"):
        rows = sorted(
            [row for row in a0["results"] if row["method"] == name],
            key=lambda row: float(row["label_fraction"]),
        )
        ax.plot(
            [100.0 * float(row["label_fraction"]) for row in rows],
            [row["macro_f1"] for row in rows],
            marker="o",
            color=COLORS[name],
            label=DISPLAY[name],
        )
    for name in ("A1", "A2", "A3", "A4"):
        rows = sorted(
            [row for row in checkpoints["results"] if row["cell"] == name],
            key=lambda row: float(row["label_fraction"]),
        )
        ax.plot(
            [100.0 * float(row["label_fraction"]) for row in rows],
            [row["macro_f1"] for row in rows],
            marker="o",
            color=COLORS[name],
            label=DISPLAY[name],
        )
    ax.set_xlabel("Available proxy labels (%)")
    ax.set_ylabel("Development macro F1")
    ax.set_ylim(0.0, 0.42)
    ax.set_title("Development smoke: source-condition proxy label efficiency")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, output_dir, "development_label_efficiency")


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
    ax.set_title("Development smoke: paired 10%-label comparisons")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, output_dir, "development_paired_differences")


def _retrieval_figure(a0: dict[str, Any], checkpoints: dict[str, Any], output_dir: Path) -> list[str]:
    points = []
    for name in ("handcrafted", "moment", "random"):
        metric = a0["method_metadata"][name]["development_retrieval"]
        points.append((name, metric))
    for key, value in checkpoints["checkpoint_metadata"].items():
        points.append((value["cell"], value["development_retrieval"]))
    fig, ax = plt.subplots(figsize=(7.0, 4.7), constrained_layout=True)
    for name, metric in points:
        ax.scatter(
            metric["top1_quality_purity"],
            metric["top1_label_purity"],
            color=COLORS[name],
            s=48,
        )
        ax.annotate(DISPLAY[name], (metric["top1_quality_purity"], metric["top1_label_purity"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0.25, color="black", linestyle="--", linewidth=0.8, label="four-class chance")
    ax.set_xlabel("Top-1 quality-stratum purity")
    ax.set_ylabel("Top-1 source-condition purity")
    ax.set_title("Cross-recording retrieval: proxy signal versus quality shortcut")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    return _save(fig, output_dir, "development_retrieval")


def _robustness_figure(checkpoints: dict[str, Any], output_dir: Path) -> list[str]:
    cells = ("a1", "a2", "a3", "a4")
    perturbations = list(
        checkpoints["checkpoint_metadata"]["a1"]["development_robustness"]["perturbations"]
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
    ax.set_yticks(np.arange(len(cells)), [cell.upper() for cell in cells])
    ax.set_title("Development prediction agreement under bounded perturbations")
    for row in range(agreement.shape[0]):
        for column in range(agreement.shape[1]):
            ax.text(column, row, f"{agreement[row, column]:.2f}", ha="center", va="center", color="white" if agreement[row, column] < 0.7 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="prediction agreement")
    return _save(fig, output_dir, "development_robustness")


def _physical_figure(checkpoints: dict[str, Any], output_dir: Path) -> list[str]:
    cells = ("a1", "a2", "a3", "a4")
    factors = list(
        checkpoints["checkpoint_metadata"]["a1"]["development_physical_fidelity"]["retained_factor_linear_probes"]
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
    ax.set_yticks(np.arange(len(cells)), [cell.upper() for cell in cells])
    ax.set_title("Retained-factor MSE reduction versus constant prior")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="relative MSE reduction")
    return _save(fig, output_dir, "development_physical_fidelity")


def build_development_report(a0_path: Path, checkpoint_path: Path, output_dir: Path) -> dict[str, Any]:
    a0 = _read_json(a0_path)
    checkpoints = _read_json(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    comparisons = paired_bootstrap_comparisons(a0, checkpoints)
    _write_csv(output_dir / "paired_bootstrap_differences.csv", comparisons)

    probe_rows = []
    for row in a0["results"]:
        probe_rows.append(
            {
                "method": row["method"],
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
                "label_fraction": row["label_fraction"],
                "seed": row["seed"],
                "macro_f1": row["macro_f1"],
                "balanced_accuracy": row["balanced_accuracy"],
                "ece": row["calibration"]["expected_calibration_error"],
                "brier": row["calibration"]["multiclass_brier"],
            }
        )
    _write_csv(output_dir / "development_probe_metrics.csv", probe_rows)
    outputs = ["paired_bootstrap_differences.csv", "development_probe_metrics.csv"]
    outputs += _label_efficiency_figure(a0, checkpoints, output_dir)
    outputs += _paired_figure(comparisons, output_dir)
    outputs += _retrieval_figure(a0, checkpoints, output_dir)
    outputs += _robustness_figure(checkpoints, output_dir)
    outputs += _physical_figure(checkpoints, output_dir)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "scope": "development smoke on source-condition proxy labels; not a morphology or OOD result",
        "source_a0": str(a0_path),
        "source_checkpoints": str(checkpoint_path),
        "paired_comparisons": comparisons,
        "outputs": outputs,
        "sealed_splits_used": [],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
