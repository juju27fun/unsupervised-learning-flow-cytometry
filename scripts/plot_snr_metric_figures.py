#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
P0_ROOT = REPO_ROOT / "P0"
for path_entry in (ROOT, P0_ROOT):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

from scripts.evaluate_snr_embedding_impact import (  # noqa: E402
    load_impact_bundle,
    precompute_neighbor_indices,
)
from scripts.plot_snr_threshold_manifolds import (  # noqa: E402
    BASE_CLASS_COLORS,
    MODEL_DISPLAY,
    PARTICLE_CLASS_NAMES,
    DatasetSpec,
    align_embeddings_to_metadata,
    apply_pub_style,
    build_datasets,
    parse_csv_strings,
    set_axis_style,
)
from scripts.run_ssl_assessment_figures import normalized_embedding_matrix  # noqa: E402


METRIC_LABELS = {
    "knn_impurity_delta": "Delta kNN impurity",
    "probe_error_lift": "Delta probe error",
    "top1_same_class_delta": "Delta top-1 same class",
}


def parse_quantiles(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one quantile")
    for value in values:
        if not 0.0 < value < 1.0:
            raise ValueError(f"Quantiles must be in (0, 1), got {value}")
    return values


def quantile_tag(value: float) -> str:
    return f"q{int(round(value * 100)):02d}"


def nearest_metric_row(metrics: pd.DataFrame, mode: str, model_key: str, quantile: float) -> pd.Series:
    rows = metrics[(metrics["mode"] == mode) & (metrics["model"] == model_key)].copy()
    if rows.empty:
        raise ValueError(f"No metric rows for mode={mode}, model={model_key}")
    idx = (rows["quantile"].astype(float) - float(quantile)).abs().idxmin()
    row = rows.loc[idx]
    if abs(float(row["quantile"]) - float(quantile)) > 1.0e-4:
        raise ValueError(f"No metric row close to quantile={quantile} for mode={mode}, model={model_key}")
    return row


def format_delta(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):+.3f}"


def most_enriched_class(row: pd.Series, class_names: tuple[str, str, str]) -> tuple[str, float]:
    best_name = class_names[0]
    best_value = float("nan")
    for class_name in class_names:
        key = f"{class_name}_low_snr_enrichment"
        if key not in row:
            continue
        value = float(row[key])
        if math.isfinite(value) and (not math.isfinite(best_value) or value > best_value):
            best_name = class_name
            best_value = value
    return best_name, best_value


def load_fixed_reductions(path: Path, model_keys: list[str]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixed reductions: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = {"selected_index", "event_id", "labels", "snr_value"}
        for model_key in model_keys:
            required.add(f"{model_key}_pca")
            required.add(f"{model_key}_tsne")
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        payload: dict[str, Any] = {
            "selected_index": np.asarray(data["selected_index"], dtype=np.int64),
            "event_id": np.asarray(data["event_id"]).astype(str),
            "labels": np.asarray(data["labels"], dtype=np.int64),
            "snr_value": np.asarray(data["snr_value"], dtype=np.float64),
        }
        for model_key in model_keys:
            payload[f"{model_key}_pca"] = np.asarray(data[f"{model_key}_pca"], dtype=np.float32)
            payload[f"{model_key}_tsne"] = np.asarray(data[f"{model_key}_tsne"], dtype=np.float32)
    return payload


def compute_low_snr_confused_masks(
    dataset: DatasetSpec,
    reductions: dict[str, Any],
    model_keys: list[str],
    quantiles: list[float],
) -> dict[str, dict[float, dict[str, np.ndarray | float]]]:
    selected_idx = np.asarray(reductions["selected_index"], dtype=np.int64)
    labels = np.asarray(reductions["labels"], dtype=np.int64)
    snr = np.asarray(reductions["snr_value"], dtype=np.float64)
    result: dict[str, dict[float, dict[str, np.ndarray | float]]] = {}
    for model_key in model_keys:
        bundle = load_impact_bundle(dataset.embedding_root / model_key / "all_embeddings.npz", model_key)
        aligned = align_embeddings_to_metadata(bundle, dataset.metadata)
        selected_embeddings = aligned[selected_idx]
        x_norm = normalized_embedding_matrix(selected_embeddings)
        neighbor_idx = precompute_neighbor_indices(x_norm, k=1)
        if neighbor_idx.shape[1] == 0:
            top1_label = np.full(labels.shape, -1, dtype=np.int64)
        else:
            top1_label = labels[neighbor_idx[:, 0]]
        result[model_key] = {}
        for quantile in quantiles:
            threshold = float(np.quantile(snr, quantile))
            low = snr <= threshold
            confused = low & (top1_label != labels)
            result[model_key][float(quantile)] = {
                "threshold": threshold,
                "low": low,
                "confused": confused,
            }
    return result


def axis_limits(coords_by_model: dict[str, dict[str, np.ndarray]]) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    limits: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for model_key, reductions in coords_by_model.items():
        for reduction_name, coords in reductions.items():
            x = coords[:, 0]
            y = coords[:, 1]
            x_pad = max(float(np.ptp(x)) * 0.05, 1.0e-6)
            y_pad = max(float(np.ptp(y)) * 0.05, 1.0e-6)
            limits[(model_key, reduction_name)] = (
                float(np.min(x) - x_pad),
                float(np.max(x) + x_pad),
                float(np.min(y) - y_pad),
                float(np.max(y) + y_pad),
            )
    return limits


def scatter_enriched(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    low: np.ndarray,
    confused: np.ndarray,
    class_names: tuple[str, str, str],
    point_size: float = 9.0,
) -> list[Any]:
    artists: list[Any] = []
    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        artist = ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=0.80,
            linewidths=0,
            c=BASE_CLASS_COLORS.get(class_id, "#555555"),
            label=class_name,
        )
        artists.append(artist)
    if bool(np.any(low)):
        low_artist = ax.scatter(
            coords[low, 0],
            coords[low, 1],
            s=point_size * 2.5,
            facecolors="none",
            edgecolors="black",
            linewidths=0.60,
            label="low SNR",
        )
        artists.append(low_artist)
    if bool(np.any(confused)):
        confused_artist = ax.scatter(
            coords[confused, 0],
            coords[confused, 1],
            s=point_size * 2.0,
            marker="x",
            c="#CC0033",
            linewidths=0.85,
            label="low SNR + top-1 cross-class",
        )
        artists.append(confused_artist)
    return artists


def add_metric_box(ax: plt.Axes, row: pd.Series, class_names: tuple[str, str, str]) -> None:
    class_name, enrichment = most_enriched_class(row, class_names)
    text = (
        f"n_low={int(row['n_low_snr'])}/{int(row['n_total'])}\n"
        f"Delta kNN={format_delta(float(row['knn_impurity_delta']))}\n"
        f"Delta err={format_delta(float(row['probe_error_lift']))}\n"
        f"enrich: {class_name} x{enrichment:.2f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.86, "linewidth": 0.5},
    )


def plot_enriched_projection(
    dataset: DatasetSpec,
    reductions: dict[str, Any],
    metrics: pd.DataFrame,
    masks: dict[str, dict[float, dict[str, np.ndarray | float]]],
    model_keys: list[str],
    mode: str,
    quantile: float,
    output_pdf: Path,
    output_png: Path,
) -> None:
    labels = np.asarray(reductions["labels"], dtype=np.int64)
    coords_by_model = {
        model_key: {
            "pca": np.asarray(reductions[f"{model_key}_pca"], dtype=np.float32),
            "tsne": np.asarray(reductions[f"{model_key}_tsne"], dtype=np.float32),
        }
        for model_key in model_keys
    }
    limits = axis_limits(coords_by_model)
    apply_pub_style()
    fig_width = max(10.8, 5.1 * len(model_keys))
    fig, axes = plt.subplots(2, len(model_keys), figsize=(fig_width, 6.5), squeeze=False, constrained_layout=False)
    legend_artists: list[Any] = []
    for col, model_key in enumerate(model_keys):
        row = nearest_metric_row(metrics, mode=mode, model_key=model_key, quantile=quantile)
        low = np.asarray(masks[model_key][float(quantile)]["low"], dtype=bool)
        confused = np.asarray(masks[model_key][float(quantile)]["confused"], dtype=bool)
        for row_i, reduction_name in enumerate(("pca", "tsne")):
            ax = axes[row_i, col]
            coords = coords_by_model[model_key][reduction_name]
            artists = scatter_enriched(ax, coords, labels, low, confused, dataset.class_names)
            if not legend_artists:
                legend_artists = artists
            xmin, xmax, ymin, ymax = limits[(model_key, reduction_name)]
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            title = MODEL_DISPLAY.get(model_key, model_key) if row_i == 0 else ""
            ax.set_title(f"{title}\n{reduction_name.upper()}" if title else reduction_name.upper(), pad=4)
            add_metric_box(ax, row, dataset.class_names)
            set_axis_style(ax)
    if legend_artists:
        labels_legend = [artist.get_label() for artist in legend_artists]
        fig.legend(
            legend_artists,
            labels_legend,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=min(6, len(legend_artists)),
            frameon=False,
            fontsize=8,
            markerscale=1.5,
            columnspacing=1.1,
            handletextpad=0.35,
        )
    fig.suptitle(f"{dataset.display_name} - SNR metric overlay {quantile_tag(quantile)} ({mode})", fontsize=12, y=0.965)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.18, wspace=0.22, hspace=0.34)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def plot_impact_curves(dataset: DatasetSpec, metrics: pd.DataFrame, model_keys: list[str], mode: str, output_pdf: Path, output_png: Path) -> None:
    apply_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    mode_rows = metrics[metrics["mode"] == mode]
    for model_key in model_keys:
        rows = mode_rows[mode_rows["model"] == model_key].sort_values("quantile")
        x = rows["quantile"].to_numpy(dtype=float) * 100.0
        axes[0].plot(x, rows["knn_impurity_delta"].to_numpy(dtype=float), marker="o", linewidth=1.25, label=MODEL_DISPLAY.get(model_key, model_key))
        axes[1].plot(x, rows["probe_error_lift"].to_numpy(dtype=float), marker="o", linewidth=1.25, label=MODEL_DISPLAY.get(model_key, model_key))
    for ax, ylabel in zip(axes, ("Low - high kNN impurity", "Low - high probe error")):
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.set_xlabel("SNR threshold quantile (%)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=7)
        set_axis_style(ax)
    fig.suptitle(f"{dataset.display_name} SNR impact curves ({mode})", fontsize=11)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def plot_enrichment_heatmap(dataset: DatasetSpec, metrics: pd.DataFrame, mode: str, output_pdf: Path, output_png: Path) -> None:
    rows = metrics[metrics["mode"] == mode].copy()
    if rows.empty:
        raise ValueError(f"No rows for mode={mode}")
    first_model = str(rows["model"].iloc[0])
    rows = rows[rows["model"] == first_model].sort_values("quantile")
    values = np.asarray(
        [
            [float(row[f"{class_name}_low_snr_enrichment"]) for row in rows.to_dict("records")]
            for class_name in dataset.class_names
        ],
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    spread = max(float(np.max(np.abs(finite - 1.0))) if finite.size else 0.25, 0.25)
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(8.4, 3.2), constrained_layout=True)
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=1.0 - spread, vmax=1.0 + spread)
    ax.set_yticks(np.arange(len(dataset.class_names)))
    ax.set_yticklabels(dataset.class_names)
    ax.set_xticks(np.arange(rows.shape[0]))
    ax.set_xticklabels([f"{float(v) * 100:.0f}" for v in rows["quantile"].tolist()], fontsize=7)
    ax.set_xlabel("SNR threshold quantile (%)")
    ax.set_title(f"{dataset.display_name} class enrichment among low-SNR events ({mode})", fontsize=10)
    fig.colorbar(image, ax=ax, label="P(class | low SNR) / P(class)")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if math.isfinite(float(value)):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6, color="black")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def plot_confusion_summary(dataset: DatasetSpec, details: dict[str, Any], model_keys: list[str], mode: str, quantiles: list[float], output_pdf: Path, output_png: Path) -> None:
    apply_pub_style()
    fig, axes = plt.subplots(len(model_keys), len(quantiles), figsize=(3.0 * len(quantiles), 2.45 * len(model_keys)), squeeze=False, constrained_layout=True)
    for row_i, model_key in enumerate(model_keys):
        model_details = details["models"][model_key][mode]
        for col_i, quantile in enumerate(quantiles):
            key = quantile_tag(quantile)
            matrix = np.asarray(model_details["thresholds"][key]["low_top1_retrieval_matrix_row_normalized"], dtype=np.float64)
            off_diag = matrix.copy()
            np.fill_diagonal(off_diag, np.nan)
            ax = axes[row_i, col_i]
            image = ax.imshow(off_diag, vmin=0.0, vmax=np.nanmax(off_diag) if np.isfinite(off_diag).any() else 1.0, cmap="magma", interpolation="nearest")
            ax.set_xticks(np.arange(len(dataset.class_names)))
            ax.set_yticks(np.arange(len(dataset.class_names)))
            ax.set_xticklabels(dataset.class_names, fontsize=7, rotation=35, ha="right")
            ax.set_yticklabels(dataset.class_names, fontsize=7)
            ax.set_title(f"{MODEL_DISPLAY.get(model_key, model_key)} {key}", fontsize=8)
            for y in range(off_diag.shape[0]):
                for x in range(off_diag.shape[1]):
                    value = off_diag[y, x]
                    if math.isfinite(float(value)):
                        ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white" if value > 0.35 else "black")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="low-SNR top-1 cross-class rate", shrink=0.82)
    fig.suptitle(f"{dataset.display_name} low-SNR top-1 cross-class retrieval ({mode})", fontsize=11)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def write_compact_table(dataset: DatasetSpec, metrics: pd.DataFrame, model_keys: list[str], modes: list[str], quantiles: list[float], output_csv: Path, output_pdf: Path) -> None:
    rows_out: list[dict[str, Any]] = []
    for mode in modes:
        for model_key in model_keys:
            for quantile in quantiles:
                row = nearest_metric_row(metrics, mode=mode, model_key=model_key, quantile=quantile)
                enriched_class, enriched_value = most_enriched_class(row, dataset.class_names)
                rows_out.append(
                    {
                        "dataset": dataset.key,
                        "mode": mode,
                        "model": model_key,
                        "quantile": float(quantile),
                        "knn_impurity_delta": float(row["knn_impurity_delta"]),
                        "probe_error_lift": float(row["probe_error_lift"]),
                        "top1_same_class_delta": float(row["top1_same_class_delta"]),
                        "most_enriched_class": enriched_class,
                        "most_enriched_value": float(enriched_value),
                    }
                )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    display_rows = [
        [
            row["mode"],
            row["model"],
            quantile_tag(float(row["quantile"])),
            format_delta(float(row["knn_impurity_delta"])),
            format_delta(float(row["probe_error_lift"])),
            format_delta(float(row["top1_same_class_delta"])),
            f"{row['most_enriched_class']} x{float(row['most_enriched_value']):.2f}",
        ]
        for row in rows_out
    ]
    apply_pub_style()
    fig_height = max(3.2, 0.31 * len(display_rows) + 1.1)
    fig, ax = plt.subplots(figsize=(11.2, fig_height), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=display_rows,
        colLabels=["mode", "model", "q", "Delta kNN", "Delta err", "Delta top1", "enrichment"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)
    ax.set_title(f"{dataset.display_name} compact SNR metric table", fontsize=11, pad=10)
    fig.savefig(output_pdf)
    plt.close(fig)


def process_dataset(dataset: DatasetSpec, args: argparse.Namespace, model_keys: list[str], modes: list[str], quantiles: list[float]) -> dict[str, Any]:
    dataset_dir = args.output_dir / dataset.key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_root / dataset.key / "snr_embedding_impact_metrics.csv"
    details_path = args.metrics_root / dataset.key / "snr_embedding_impact_metrics.json"
    reductions_path = args.manifold_root / dataset.key / "fixed_reductions.npz"
    metrics = pd.read_csv(metrics_path)
    details = json.load(details_path.open())
    reductions = load_fixed_reductions(reductions_path, model_keys)
    masks = compute_low_snr_confused_masks(dataset, reductions, model_keys, quantiles)

    with PdfPages(dataset_dir / f"{dataset.key}_enriched_projection_q20_q50_q80.pdf") as pdf:
        for quantile in quantiles:
            stem = f"{dataset.key}_{quantile_tag(quantile)}_{args.overlay_mode}_enriched_pca_tsne"
            pdf_path = dataset_dir / f"{stem}.pdf"
            png_path = dataset_dir / f"{stem}.png"
            plot_enriched_projection(dataset, reductions, metrics, masks, model_keys, args.overlay_mode, quantile, pdf_path, png_path)
            image = plt.imread(png_path)
            fig, ax = plt.subplots(figsize=(11.0, 4.7), constrained_layout=True)
            ax.imshow(image)
            ax.set_axis_off()
            pdf.savefig(fig)
            plt.close(fig)

    for mode in modes:
        plot_impact_curves(
            dataset,
            metrics,
            model_keys,
            mode,
            dataset_dir / f"{dataset.key}_{mode}_snr_impact_curves.pdf",
            dataset_dir / f"{dataset.key}_{mode}_snr_impact_curves.png",
        )
        plot_enrichment_heatmap(
            dataset,
            metrics,
            mode,
            dataset_dir / f"{dataset.key}_{mode}_class_enrichment_heatmap.pdf",
            dataset_dir / f"{dataset.key}_{mode}_class_enrichment_heatmap.png",
        )
        plot_confusion_summary(
            dataset,
            details,
            model_keys,
            mode,
            quantiles,
            dataset_dir / f"{dataset.key}_{mode}_top1_confusion_summary.pdf",
            dataset_dir / f"{dataset.key}_{mode}_top1_confusion_summary.png",
        )
    write_compact_table(
        dataset,
        metrics,
        model_keys,
        modes,
        quantiles,
        dataset_dir / f"{dataset.key}_compact_metric_table.csv",
        dataset_dir / f"{dataset.key}_compact_metric_table.pdf",
    )
    return {
        "dataset": dataset.key,
        "output_dir": str(dataset_dir),
        "quantiles": [float(q) for q in quantiles],
        "modes": modes,
        "models": model_keys,
    }


def run(args: argparse.Namespace) -> None:
    model_keys = parse_csv_strings(args.models)
    modes = parse_csv_strings(args.modes)
    quantiles = parse_quantiles(args.threshold_quantiles)
    if args.overlay_mode not in modes:
        modes = [args.overlay_mode] + modes
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {"datasets": {}, "models": model_keys, "modes": modes, "quantiles": quantiles}
    for dataset in build_datasets(args):
        summaries["datasets"][dataset.key] = process_dataset(dataset, args, model_keys, modes, quantiles)
    with (args.output_dir / "snr_metric_figures_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
    print(json.dumps({"output_dir": str(args.output_dir), "datasets": sorted(summaries["datasets"].keys())}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot SNR-conditioned metric figures and enriched PCA/t-SNE overlays.")
    parser.add_argument("--particle-root", type=Path, default=ROOT / "outputs" / "pretrained_backbones" / "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap")
    parser.add_argument("--particle-manifest", type=Path, default=REPO_ROOT / "particles2SNR_pipeline" / "output" / "p0_c1_Particles2SNR_F" / "event_classification_dataset" / "event_manifest.csv")
    parser.add_argument("--yeast-event-root", type=Path, default=ROOT / "outputs" / "pretrained_backbones" / "yeast_passage_events_p3_4096")
    parser.add_argument("--yeast-embedding-root", type=Path, default=ROOT / "outputs" / "pretrained_backbones" / "particles2snr_f_3class_plus_yeast_moment_patchtst_conv1dgap")
    parser.add_argument("--metrics-root", type=Path, default=ROOT / "outputs" / "snr_threshold_metrics")
    parser.add_argument("--manifold-root", type=Path, default=ROOT / "outputs" / "snr_threshold_manifolds")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "snr_metric_figures")
    parser.add_argument("--models", default="moment_official,patchtst_pretrained,conv1dgap_same_input_3class")
    parser.add_argument("--modes", default="visual_subset,full_dataset")
    parser.add_argument("--overlay-mode", default="visual_subset")
    parser.add_argument("--threshold-quantiles", default="0.20,0.50,0.80")
    parser.add_argument("--yeast-classes", default="mix,budding,shmoo2")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
