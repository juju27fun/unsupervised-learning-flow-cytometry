from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "handcrafted": "Handcrafted",
    "A3": "A3 physics-informed",
    "A4": "A4 physics + adaptation",
}
COLORS = {"handcrafted": "#0072B2", "A3": "#D55E00", "A4": "#56B4E9"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [pdf.name, png.name]


def build_final_report(metrics_path: Path, output_dir: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if payload["sealed_splits_used"] != ["in_session_test"]:
        raise ValueError("Expected the one-time in-session test metrics")
    output_dir.mkdir(parents=True, exist_ok=False)
    methods = payload["methods"]
    method_rows = []
    class_rows = []
    for method in methods:
        selected = [row for row in payload["results"] if row["method"] == method]
        values = np.asarray([row["macro_f1"] for row in selected])
        method_rows.append(
            {
                "method": method,
                "display_method": DISPLAY[method],
                "n_runs": len(selected),
                "macro_f1_mean": float(values.mean()),
                "macro_f1_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "macro_f1_min": float(values.min()),
                "macro_f1_max": float(values.max()),
            }
        )
        for class_name in payload["class_names"]:
            recalls = np.asarray(
                [row["per_class_recall"][class_name] for row in selected]
            )
            class_rows.append(
                {
                    "method": method,
                    "class_name": class_name,
                    "recall_mean": float(recalls.mean()),
                    "recall_std": (
                        float(recalls.std(ddof=1)) if len(recalls) > 1 else 0.0
                    ),
                }
            )
    _write_csv(output_dir / "final_method_summary.csv", method_rows)
    _write_csv(output_dir / "final_class_recall.csv", class_rows)

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    positions = np.arange(len(method_rows))
    ax.bar(
        positions,
        [row["macro_f1_mean"] for row in method_rows],
        yerr=[row["macro_f1_std"] for row in method_rows],
        color=[COLORS[row["method"]] for row in method_rows],
        capsize=4,
    )
    ax.set_xticks(positions, [row["display_method"] for row in method_rows])
    ax.set_ylim(0.0, 0.55)
    ax.set_ylabel("In-session test macro F1")
    ax.set_title("Frozen 10%-label confirmatory endpoint")
    ax.grid(axis="y", alpha=0.2)
    outputs = _save(fig, output_dir, "final_macro_f1")

    comparison_rows = payload["paired_comparisons"]
    fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)
    means = np.asarray([row["mean_difference"] for row in comparison_rows])
    lower = means - np.asarray([row["ci_95_low"] for row in comparison_rows])
    upper = np.asarray([row["ci_95_high"] for row in comparison_rows]) - means
    positions = np.arange(len(comparison_rows))
    ax.errorbar(
        means,
        positions,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#0072B2",
        capsize=4,
    )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(positions, [row["comparison"] for row in comparison_rows])
    ax.set_xlabel("Paired macro-F1 difference (95% hierarchical bootstrap interval)")
    ax.set_title("One-time in-session test comparisons")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    outputs += _save(fig, output_dir, "final_paired_differences")

    values = np.asarray(
        [
            [
                next(
                    row["recall_mean"]
                    for row in class_rows
                    if row["method"] == method and row["class_name"] == class_name
                )
                for class_name in payload["class_names"]
            ]
            for method in methods
        ]
    )
    fig, ax = plt.subplots(figsize=(7.0, 3.3), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(payload["class_names"])), payload["class_names"])
    ax.set_yticks(np.arange(len(methods)), [DISPLAY[method] for method in methods])
    ax.set_title("In-session test recall by source-condition proxy")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if values[row, column] < 0.55 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="recall")
    outputs += _save(fig, output_dir, "final_class_recall")

    primary = comparison_rows[0]
    adaptation = comparison_rows[1]
    decision = {
        "promotion_decision": "do_not_promote_a4",
        "primary_interval_position": (
            "entirely_below_zero"
            if primary["ci_95_high"] < 0.0
            else "entirely_above_zero"
            if primary["ci_95_low"] > 0.0
            else "crosses_zero"
        ),
        "a4_is_significantly_worse_than_handcrafted": primary["ci_95_high"] < 0.0,
        "a4_improves_over_a3": adaptation["ci_95_low"] > 0.0,
        "a4_vs_a3_effect_reaches_0p03": adaptation["mean_difference"] >= 0.03,
        "scope": "in-session source-condition proxy endpoint; not morphology or OOD",
    }
    (output_dir / "final_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs = ["final_method_summary.csv", "final_class_recall.csv", "final_decision.json", *outputs]
    summary = {
        "schema_version": 1,
        "status": "complete",
        "source_metrics": str(metrics_path),
        "n_test_events": payload["n_test_events"],
        "n_test_capture_blocks": payload["n_test_capture_blocks"],
        "paired_comparisons": comparison_rows,
        "decision": decision,
        "outputs": outputs,
        "sealed_splits_used": ["in_session_test"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
