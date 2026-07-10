#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

MODEL_DISPLAY = {
    "moment_official": "MOMENT frozen pretrained",
    "patchtst_pretrained": "PatchTST frozen pretrained",
    "conv1dgap_same_input_3class": "Conv1D-GAP supervised",
}

MODEL_ORDER = ("moment_official", "patchtst_pretrained", "conv1dgap_same_input_3class")
BASE_CLASS_COLORS = {
    0: "#0072B2",
    1: "#009E73",
    2: "#D55E00",
}

PARTICLE_CLASS_NAMES = ("2um", "4um", "10um")
YEAST_CLASS_NAMES = ("mix", "budding", "shmoo2")
QUANTILE_VALUES = tuple(round(v, 2) for v in np.arange(0.05, 0.81, 0.05))


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    snr_column: str
    snr_label: str
    class_names: tuple[str, str, str]
    embedding_root: Path
    metadata: pd.DataFrame


@dataclass(frozen=True)
class EmbeddingBundle:
    model_key: str
    embeddings: np.ndarray
    event_id: np.ndarray


def parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def quantile_grid() -> np.ndarray:
    return np.asarray(QUANTILE_VALUES, dtype=np.float32)


def apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 9,
            "axes.labelsize": 8,
        }
    )


def set_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#555555")
    ax.grid(False)


def balanced_indices(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(set(int(v) for v in labels.tolist())):
        idx = np.flatnonzero(labels == class_id)
        if max_per_class > 0 and idx.size > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.extend(int(i) for i in idx.tolist())
    arr = np.asarray(selected, dtype=np.int64)
    arr.sort()
    return arr


def quantile_thresholds(values: np.ndarray, quantiles: np.ndarray | None = None) -> np.ndarray:
    q = quantile_grid() if quantiles is None else np.asarray(quantiles, dtype=np.float32)
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("Cannot compute SNR thresholds on an empty array")
    if not np.isfinite(x).all():
        raise ValueError("SNR values contain NaN or inf")
    return np.quantile(x, q).astype(np.float64)


def threshold_summary_rows(
    dataset_key: str,
    snr_values: np.ndarray,
    labels: np.ndarray,
    class_names: tuple[str, str, str],
    thresholds: np.ndarray,
    quantiles: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    q = quantile_grid() if quantiles is None else np.asarray(quantiles, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    n_total = int(labels.size)
    for quantile, threshold in zip(q.tolist(), thresholds.tolist()):
        low = snr_values <= float(threshold)
        row: dict[str, Any] = {
            "dataset": dataset_key,
            "quantile": float(quantile),
            "threshold": float(threshold),
            "n_total": n_total,
            "n_low_snr": int(low.sum()),
            "low_snr_fraction": float(low.mean()) if n_total else float("nan"),
        }
        for class_id, class_name in enumerate(class_names):
            class_mask = labels == class_id
            n_class = int(class_mask.sum())
            n_low = int(np.logical_and(low, class_mask).sum())
            row[f"{class_name}_n"] = n_class
            row[f"{class_name}_low_snr_n"] = n_low
            row[f"{class_name}_low_snr_fraction"] = float(n_low / n_class) if n_class else float("nan")
        rows.append(row)
    return rows


def load_npz_bundle(path: Path, model_key: str) -> EmbeddingBundle:
    if not path.is_file():
        raise FileNotFoundError(f"Missing embedding file for {model_key}: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = {"embeddings", "event_id"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        event_id = np.asarray(data["event_id"]).astype(str)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings for {model_key}, got {embeddings.shape}")
    if embeddings.shape[0] != event_id.shape[0]:
        raise ValueError(f"Embedding/event_id row mismatch for {model_key}")
    return EmbeddingBundle(model_key=model_key, embeddings=embeddings, event_id=event_id)


def align_embeddings_to_metadata(bundle: EmbeddingBundle, metadata: pd.DataFrame) -> np.ndarray:
    event_to_idx = {event_id: idx for idx, event_id in enumerate(bundle.event_id.tolist())}
    indices: list[int] = []
    missing: list[str] = []
    for event_id in metadata["event_id"].astype(str).tolist():
        idx = event_to_idx.get(event_id)
        if idx is None:
            missing.append(event_id)
        else:
            indices.append(idx)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{bundle.model_key} is missing {len(missing)} metadata events, e.g. {preview}")
    return bundle.embeddings[np.asarray(indices, dtype=np.int64)]


def build_particle_metadata(particle_root: Path, particle_manifest: Path) -> pd.DataFrame:
    events_path = particle_root / "events_metadata.csv"
    if not events_path.is_file():
        raise FileNotFoundError(f"Missing particle events metadata: {events_path}")
    if not particle_manifest.is_file():
        raise FileNotFoundError(f"Missing Particles2SNR_F event manifest: {particle_manifest}")

    events = pd.read_csv(events_path)
    manifest = pd.read_csv(particle_manifest)
    events["join_key"] = events["sample_id"].astype(str)
    manifest["join_key"] = manifest["output_filename"].astype(str).str.replace(".npy", "", regex=False)
    merged = events.merge(
        manifest[["join_key", "snr_db", "width_ms", "frequency"]],
        on="join_key",
        how="left",
        validate="one_to_one",
    )
    if merged["snr_db"].isna().any():
        missing = int(merged["snr_db"].isna().sum())
        raise ValueError(f"Particle SNR join failed for {missing}/{len(merged)} events")
    class_map = {name: idx for idx, name in enumerate(PARTICLE_CLASS_NAMES)}
    merged = merged[merged["class_name"].isin(class_map)].copy()
    merged["plot_class_id"] = merged["class_name"].map(class_map).astype(np.int64)
    merged["plot_class_name"] = merged["class_name"]
    merged["snr_value"] = merged["snr_db"].astype(float)
    return merged.reset_index(drop=True)


def build_yeast_metadata(yeast_event_root: Path, yeast_classes: tuple[str, str, str] = YEAST_CLASS_NAMES) -> pd.DataFrame:
    events_path = yeast_event_root / "events_metadata.csv"
    if not events_path.is_file():
        raise FileNotFoundError(f"Missing yeast events metadata: {events_path}")
    events = pd.read_csv(events_path)
    required = {"event_id", "source_group", "snr_proxy"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"{events_path} is missing columns: {sorted(missing)}")
    class_map = {name: idx for idx, name in enumerate(yeast_classes)}
    filtered = events[events["source_group"].isin(class_map)].copy()
    if filtered.empty:
        raise ValueError(f"No yeast events found for classes {yeast_classes}")
    filtered["plot_class_id"] = filtered["source_group"].map(class_map).astype(np.int64)
    filtered["plot_class_name"] = filtered["source_group"].astype(str)
    filtered["snr_value"] = filtered["snr_proxy"].astype(float)
    return filtered.reset_index(drop=True)


def reduce_embeddings(embeddings: np.ndarray, seed: int, run_tsne: bool = True) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x = StandardScaler().fit_transform(np.asarray(embeddings, dtype=np.float32)).astype(np.float32)
    n_samples, n_features = x.shape
    if n_samples < 2:
        coords = np.zeros((n_samples, 2), dtype=np.float32)
        return coords, coords.copy(), {"pca_explained_variance_ratio_sum": float("nan"), "trustworthiness": float("nan")}
    pca = PCA(n_components=2, random_state=seed)
    pca_coords = pca.fit_transform(x).astype(np.float32)
    metrics: dict[str, float] = {"pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_))}
    if not run_tsne or n_samples < 5:
        metrics["trustworthiness"] = float("nan")
        metrics["tsne_perplexity"] = float("nan")
        return pca_coords, pca_coords.copy(), metrics
    pre_dim = min(50, n_features, n_samples - 1)
    x_pre = PCA(n_components=pre_dim, random_state=seed).fit_transform(x) if pre_dim < n_features else x
    perplexity = min(30, max(2, (n_samples - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
    tsne_coords = tsne.fit_transform(x_pre).astype(np.float32)
    trust_neighbors = min(10, max(1, (n_samples // 2) - 1))
    metrics["trustworthiness"] = float(trustworthiness(x, tsne_coords, n_neighbors=trust_neighbors)) if trust_neighbors >= 1 else float("nan")
    metrics["tsne_perplexity"] = float(perplexity)
    return pca_coords, tsne_coords, metrics


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


def scatter_classes_with_overlay(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: np.ndarray,
    low_mask: np.ndarray,
    class_names: tuple[str, str, str],
    point_size: float = 9.0,
) -> tuple[list[Any], Any | None]:
    artists: list[Any] = []
    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        artist = ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=0.82,
            linewidths=0,
            c=BASE_CLASS_COLORS[class_id],
            label=class_name,
        )
        artists.append(artist)
    overlay_artist = None
    if bool(np.any(low_mask)):
        overlay_artist = ax.scatter(
            coords[low_mask, 0],
            coords[low_mask, 1],
            s=point_size * 2.5,
            facecolors="none",
            edgecolors="black",
            linewidths=0.65,
            label="low SNR",
        )
    return artists, overlay_artist


def add_legend(fig: plt.Figure, class_artists: list[Any], overlay_artist: Any | None) -> None:
    artists = list(class_artists)
    if overlay_artist is not None:
        artists.append(overlay_artist)
    if not artists:
        return
    fig.legend(
        artists,
        [artist.get_label() for artist in artists],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=len(artists),
        frameon=False,
        fontsize=8,
        markerscale=1.5,
        handletextpad=0.35,
        columnspacing=1.2,
    )


def plot_threshold_figure(
    dataset: DatasetSpec,
    selected_labels: np.ndarray,
    selected_snr: np.ndarray,
    coords_by_model: dict[str, dict[str, np.ndarray]],
    model_keys: list[str],
    threshold: float,
    quantile: float,
    output_pdf: Path,
    output_png: Path,
) -> None:
    apply_pub_style()
    low_mask = selected_snr <= float(threshold)
    fig_width = max(10.8, 5.1 * len(model_keys))
    fig, axes = plt.subplots(2, len(model_keys), figsize=(fig_width, 6.3), squeeze=False, constrained_layout=False)
    limits = axis_limits(coords_by_model)
    legend_artists: list[Any] = []
    legend_overlay = None
    for col, model_key in enumerate(model_keys):
        reductions = coords_by_model[model_key]
        for row, reduction_name in enumerate(("pca", "tsne")):
            ax = axes[row, col]
            coords = reductions[reduction_name]
            class_artists, overlay_artist = scatter_classes_with_overlay(
                ax,
                coords,
                selected_labels,
                low_mask,
                dataset.class_names,
            )
            if not legend_artists:
                legend_artists = class_artists
            if legend_overlay is None and overlay_artist is not None:
                legend_overlay = overlay_artist
            xmin, xmax, ymin, ymax = limits[(model_key, reduction_name)]
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(f"{MODEL_DISPLAY.get(model_key, model_key)}\n{reduction_name.upper()}" if row == 0 else reduction_name.upper(), pad=4)
            set_axis_style(ax)
    n_low = int(low_mask.sum())
    fig.suptitle(
        f"{dataset.display_name} - SNR overlay q={quantile:.2f}, "
        f"{dataset.snr_label} <= {threshold:.3g} ({n_low}/{selected_snr.size})",
        fontsize=12,
        y=0.965,
    )
    add_legend(fig, legend_artists, legend_overlay)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.18, wspace=0.22, hspace=0.34)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def process_dataset(
    dataset: DatasetSpec,
    model_keys: list[str],
    output_dir: Path,
    max_plot_per_class: int,
    seed: int,
    run_tsne: bool,
) -> dict[str, Any]:
    dataset_dir = output_dir / dataset.key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    labels = dataset.metadata["plot_class_id"].to_numpy(dtype=np.int64)
    selected_idx = balanced_indices(labels, max_per_class=max_plot_per_class, seed=seed)
    selected_meta = dataset.metadata.iloc[selected_idx].reset_index(drop=True)
    selected_labels = selected_meta["plot_class_id"].to_numpy(dtype=np.int64)
    selected_snr = selected_meta["snr_value"].to_numpy(dtype=np.float64)
    selected_event_id = selected_meta["event_id"].astype(str).to_numpy()

    coords_by_model: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, Any] = {}
    reductions_payload: dict[str, np.ndarray] = {
        "selected_index": selected_idx.astype(np.int64),
        "event_id": selected_event_id,
        "labels": selected_labels.astype(np.int64),
        "snr_value": selected_snr.astype(np.float32),
        "class_name": selected_meta["plot_class_name"].astype(str).to_numpy(),
    }

    for model_key in model_keys:
        bundle = load_npz_bundle(dataset.embedding_root / model_key / "all_embeddings.npz", model_key)
        embeddings = align_embeddings_to_metadata(bundle, dataset.metadata)
        selected_embeddings = embeddings[selected_idx]
        pca_coords, tsne_coords, reduction_metrics = reduce_embeddings(selected_embeddings, seed=seed, run_tsne=run_tsne)
        coords_by_model[model_key] = {"pca": pca_coords, "tsne": tsne_coords}
        reductions_payload[f"{model_key}_pca"] = pca_coords.astype(np.float32)
        reductions_payload[f"{model_key}_tsne"] = tsne_coords.astype(np.float32)
        metrics[model_key] = reduction_metrics

    np.savez_compressed(dataset_dir / "fixed_reductions.npz", **reductions_payload)
    thresholds = quantile_thresholds(selected_snr)
    rows = threshold_summary_rows(dataset.key, selected_snr, selected_labels, dataset.class_names, thresholds)
    write_csv(dataset_dir / "snr_threshold_summary.csv", rows)

    with PdfPages(dataset_dir / f"{dataset.key}_snr_overlay_16_thresholds.pdf") as pdf:
        for row in rows:
            quantile = float(row["quantile"])
            threshold = float(row["threshold"])
            stem = f"{dataset.key}_q{int(round(quantile * 100)):02d}_snr_overlay_pca_tsne"
            pdf_path = dataset_dir / f"{stem}.pdf"
            png_path = dataset_dir / f"{stem}.png"
            plot_threshold_figure(
                dataset=dataset,
                selected_labels=selected_labels,
                selected_snr=selected_snr,
                coords_by_model=coords_by_model,
                model_keys=model_keys,
                threshold=threshold,
                quantile=quantile,
                output_pdf=pdf_path,
                output_png=png_path,
            )
            image = plt.imread(png_path)
            fig, ax = plt.subplots(figsize=(11.0, 4.55), constrained_layout=True)
            ax.imshow(image)
            ax.set_axis_off()
            pdf.savefig(fig)
            plt.close(fig)

    summary = {
        "dataset": dataset.key,
        "display_name": dataset.display_name,
        "embedding_root": str(dataset.embedding_root),
        "n_events_available": int(dataset.metadata.shape[0]),
        "n_events_plotted": int(selected_idx.size),
        "max_plot_per_class": int(max_plot_per_class),
        "snr_column": dataset.snr_column,
        "snr_label": dataset.snr_label,
        "class_names": list(dataset.class_names),
        "quantiles": [float(v) for v in quantile_grid().tolist()],
        "threshold_source": "balanced plotted subset",
        "reduction_metrics": metrics,
    }
    with (dataset_dir / "snr_threshold_summary.json").open("w") as f:
        json.dump({**summary, "thresholds": rows}, f, indent=2, sort_keys=True)
    return summary


def build_datasets(args: argparse.Namespace) -> list[DatasetSpec]:
    particle_metadata = build_particle_metadata(args.particle_root, args.particle_manifest)
    yeast_metadata = build_yeast_metadata(args.yeast_event_root, tuple(parse_csv_strings(args.yeast_classes)))
    return [
        DatasetSpec(
            key="particles",
            display_name="Particles2SNR_F",
            snr_column="snr_db",
            snr_label="SNR dB",
            class_names=PARTICLE_CLASS_NAMES,
            embedding_root=args.particle_root,
            metadata=particle_metadata,
        ),
        DatasetSpec(
            key="yeast",
            display_name="Yeast",
            snr_column="snr_proxy",
            snr_label="SNR proxy",
            class_names=tuple(parse_csv_strings(args.yeast_classes)),  # type: ignore[arg-type]
            embedding_root=args.yeast_embedding_root,
            metadata=yeast_metadata,
        ),
    ]


def run(args: argparse.Namespace) -> None:
    model_keys = parse_csv_strings(args.models)
    missing_model_keys = [key for key in model_keys if key not in MODEL_DISPLAY]
    if missing_model_keys:
        raise ValueError(f"Unsupported model keys: {missing_model_keys}")
    summaries: dict[str, Any] = {"models": model_keys, "datasets": {}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in build_datasets(args):
        summaries["datasets"][dataset.key] = process_dataset(
            dataset=dataset,
            model_keys=model_keys,
            output_dir=args.output_dir,
            max_plot_per_class=args.max_plot_per_class,
            seed=args.seed,
            run_tsne=not args.skip_tsne,
        )
    with (args.output_dir / "snr_threshold_manifolds_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
    print(json.dumps({"output_dir": str(args.output_dir), "datasets": sorted(summaries["datasets"].keys())}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot fixed PCA/t-SNE manifolds with 16 SNR-threshold overlays for particles and yeast.")
    parser.add_argument("--particle-root", type=Path, default=ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones" / "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap")
    parser.add_argument("--particle-manifest", type=Path, default=REPO_ROOT / "artifacts" / "particles2SNR-pipeline" / "runs" / "p0_c1_Particles2SNR_F" / "event_classification_dataset" / "event_manifest.csv")
    parser.add_argument("--yeast-event-root", type=Path, default=ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones" / "yeast_passage_events_p3_4096")
    parser.add_argument("--yeast-embedding-root", type=Path, default=ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones" / "particles2snr_f_3class_plus_yeast_moment_patchtst_conv1dgap")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "snr_threshold_manifolds")
    parser.add_argument("--models", default="moment_official,patchtst_pretrained,conv1dgap_same_input_3class")
    parser.add_argument("--yeast-classes", default="mix,budding,shmoo2")
    parser.add_argument("--max-plot-per-class", type=int, default=500)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
