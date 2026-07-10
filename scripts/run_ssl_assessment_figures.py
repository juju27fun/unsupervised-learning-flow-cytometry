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
import torch
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import balanced_accuracy_score, f1_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
P0_ROOT = REPO_ROOT / "P0"
for path_entry in (ROOT, P0_ROOT):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

from p3_ssl.serialization import json_safe


CLASS_NAMES = {
    0: "2um",
    1: "4um",
    2: "10um",
    3: "unclear",
}

CLASS_COLORS = {
    0: "#0072B2",
    1: "#009E73",
    2: "#D55E00",
    3: "#CC79A7",
}

DISPLAY_NAMES = {
    "moment_official": "MOMENT frozen pretrained",
    "patchtst_pretrained": "PatchTST frozen pretrained",
    "patchtst_pretrained_full": "PatchTST full fine-tuned",
    "p3_ssl_moment_like": "P3 SSL MOMENT-like",
    "conv1dgap_same_input_3class": "Conv1D-GAP supervised",
    "conv1dgap_4class": "Conv1D-GAP supervised",
    "raw_signal": "Raw signal",
    "random_projection": "Random projection",
}


@dataclass(frozen=True)
class EmbeddingBundle:
    key: str
    display_name: str
    embeddings: np.ndarray
    labels: np.ndarray
    split: np.ndarray
    event_id: np.ndarray
    class_name: np.ndarray


def parse_csv_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def parse_csv_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def output_bases(output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"


def write_strict_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


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


def embedding_file_candidates(embedding_root: Path, key: str) -> list[Path]:
    candidates = [
        embedding_root / key / "all_embeddings.npz",
        embedding_root / key / "embeddings.npz",
        embedding_root / key / "zero_shot" / "embeddings.npz",
    ]
    if key == "patchtst_pretrained_full":
        candidates.extend(
            [
                embedding_root / "patchtst_pretrained_full" / "all_embeddings.npz",
                embedding_root / "patchtst_pretrained" / "full_finetune" / "embeddings.npz",
                embedding_root / "patchtst_full_20ep" / "patchtst_pretrained" / "full_finetune" / "embeddings.npz",
            ]
        )
    if key == "p3_ssl_moment_like":
        candidates.extend(
            [
                embedding_root / "p3_ssl_moment_like" / "all_embeddings.npz",
                embedding_root / "embedding_space" / "embeddings_all.npz",
                embedding_root / "reconstruction_moment_like_60ep" / "embedding_space" / "embeddings_all.npz",
                embedding_root / "embeddings_all.npz",
            ]
        )
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def resolve_embedding_file(embedding_root: Path, key: str) -> Path:
    candidates = embedding_file_candidates(embedding_root, key)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Missing embedding file for " + key + ": tried " + ", ".join(str(path) for path in candidates))


def load_npz_bundle(path: Path, key: str) -> EmbeddingBundle:
    if not path.is_file():
        raise FileNotFoundError(f"Missing embedding file: {path}")
    with np.load(path, allow_pickle=True) as data:
        if key == "p3_ssl_moment_like" and "moment_embeddings" in data.files:
            required = {"moment_embeddings", "class_id", "split", "event_id"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path} is missing keys: {sorted(missing)}")
            embeddings = np.asarray(data["moment_embeddings"], dtype=np.float32)
            labels = np.asarray(data["class_id"], dtype=np.int64)
            split = np.asarray(data["split"]).astype(str)
            event_id = np.asarray(data["event_id"]).astype(str)
            class_name = np.asarray(data["class_name"]).astype(str) if "class_name" in data.files else np.asarray([CLASS_NAMES.get(int(v), str(v)) for v in labels])
        else:
            required = {"embeddings", "labels", "split", "event_id"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path} is missing keys: {sorted(missing)}")
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            labels = np.asarray(data["labels"], dtype=np.int64)
            split = np.asarray(data["split"]).astype(str)
            event_id = np.asarray(data["event_id"]).astype(str)
            class_name = np.asarray(data["class_name"]).astype(str) if "class_name" in data.files else np.asarray([CLASS_NAMES.get(int(v), str(v)) for v in labels])
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings in {path}, got {embeddings.shape}")
    return EmbeddingBundle(
        key=key,
        display_name=DISPLAY_NAMES.get(key, key),
        embeddings=embeddings,
        labels=labels,
        split=split,
        event_id=event_id,
        class_name=class_name,
    )


def load_aligned_inputs(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None or not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as data:
        required = {"signals", "labels", "split"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        return {
            "signals": np.asarray(data["signals"], dtype=np.float32),
            "labels": np.asarray(data["labels"], dtype=np.int64),
            "split": np.asarray(data["split"]).astype(str),
        }


def random_projection(signals: np.ndarray, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / math.sqrt(signals.shape[1]), size=(signals.shape[1], dim)).astype(np.float32)
    return signals.astype(np.float32) @ proj


def load_embedding_bundles(
    embedding_root: Path,
    model_keys: list[str],
    aligned_inputs: dict[str, np.ndarray] | None,
    include_raw_baseline: bool,
    include_random_baseline: bool,
    random_projection_dim: int,
    seed: int,
) -> list[EmbeddingBundle]:
    bundles = [
        load_npz_bundle(resolve_embedding_file(embedding_root, key), key)
        for key in model_keys
    ]
    if not bundles:
        raise ValueError("At least one model embedding bundle is required")
    reference = bundles[0]
    for bundle in bundles[1:]:
        if not np.array_equal(bundle.labels, reference.labels):
            raise ValueError(f"Label order mismatch between {reference.key} and {bundle.key}")
        if not np.array_equal(bundle.split, reference.split):
            raise ValueError(f"Split order mismatch between {reference.key} and {bundle.key}")
        if not np.array_equal(bundle.event_id, reference.event_id):
            raise ValueError(f"Event order mismatch between {reference.key} and {bundle.key}")

    if aligned_inputs is not None:
        signals = aligned_inputs["signals"]
        if signals.shape[0] != reference.labels.shape[0]:
            raise ValueError("Aligned inputs and embeddings have different row counts")
        if not np.array_equal(aligned_inputs["labels"], reference.labels):
            raise ValueError("Aligned input labels do not match embedding labels")
        if not np.array_equal(aligned_inputs["split"], reference.split):
            raise ValueError("Aligned input splits do not match embedding splits")
        if include_raw_baseline:
            bundles.append(
                EmbeddingBundle(
                    key="raw_signal",
                    display_name=DISPLAY_NAMES["raw_signal"],
                    embeddings=signals.astype(np.float32),
                    labels=reference.labels,
                    split=reference.split,
                    event_id=reference.event_id,
                    class_name=reference.class_name,
                )
            )
        if include_random_baseline:
            bundles.append(
                EmbeddingBundle(
                    key="random_projection",
                    display_name=DISPLAY_NAMES["random_projection"],
                    embeddings=random_projection(signals, random_projection_dim, seed=seed),
                    labels=reference.labels,
                    split=reference.split,
                    event_id=reference.event_id,
                    class_name=reference.class_name,
                )
            )
    return bundles


def read_event_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", newline="") as f:
        return {row["event_id"]: row for row in csv.DictReader(f)}


def balanced_indices(labels: np.ndarray, max_per_class: int | None, seed: int, pool: np.ndarray | None = None) -> np.ndarray:
    base = np.arange(labels.shape[0], dtype=np.int64) if pool is None else np.asarray(pool, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(set(int(labels[i]) for i in base.tolist())):
        idx = base[labels[base] == class_id]
        if max_per_class is not None and max_per_class > 0 and idx.size > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.extend(int(i) for i in idx.tolist())
    arr = np.asarray(selected, dtype=np.int64)
    rng.shuffle(arr)
    return arr


def scale_embeddings(embeddings: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(embeddings).astype(np.float32)


def reduce_embeddings(embeddings: np.ndarray, seed: int, run_tsne: bool = True) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x = scale_embeddings(embeddings)
    n_samples, n_features = x.shape
    if n_samples < 2:
        coords = np.zeros((n_samples, 2), dtype=np.float32)
        return coords, coords.copy(), {"pca_explained_variance_ratio_sum": float("nan"), "trustworthiness": float("nan")}
    pca = PCA(n_components=2, random_state=seed)
    pca_coords = pca.fit_transform(x).astype(np.float32)
    metrics: dict[str, float] = {
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
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
    metrics["trustworthiness"] = (
        float(trustworthiness(x, tsne_coords, n_neighbors=trust_neighbors))
        if trust_neighbors >= 1
        else float("nan")
    )
    metrics["tsne_perplexity"] = float(perplexity)
    return pca_coords, tsne_coords, metrics


def knn_same_class_rate(embeddings: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    if labels.shape[0] <= 1:
        return float("nan")
    x = scale_embeddings(embeddings)
    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0e-8)
    sim = x_norm @ x_norm.T
    np.fill_diagonal(sim, -np.inf)
    k_eff = min(k, labels.shape[0] - 1)
    nn = np.argpartition(-sim, kth=np.arange(k_eff), axis=1)[:, :k_eff]
    return float(np.mean(labels[nn] == labels[:, None]))


def class_scatter(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, point_size: float = 9.0) -> list[Any]:
    artists: list[Any] = []
    for class_id in sorted(set(int(v) for v in labels.tolist())):
        mask = labels == class_id
        artist = ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=0.84,
            linewidths=0,
            c=CLASS_COLORS.get(class_id, "#555555"),
            label=CLASS_NAMES.get(class_id, str(class_id)),
        )
        artists.append(artist)
    return artists


def plot_manifold_figure(
    bundles: list[EmbeddingBundle],
    output_dir: Path,
    max_events_per_class: int,
    seed: int,
    run_tsne: bool,
) -> dict[str, Any]:
    selected = balanced_indices(bundles[0].labels, max_per_class=max_events_per_class, seed=seed)
    labels = bundles[0].labels[selected]
    row_count = 2 if run_tsne else 1
    apply_pub_style()
    fig_width = max(8.0, 3.2 * len(bundles))
    fig_height = 6.2 if run_tsne else 3.7
    fig, axes_raw = plt.subplots(row_count, len(bundles), figsize=(fig_width, fig_height), squeeze=False, constrained_layout=False)
    axes = np.asarray(axes_raw).reshape(row_count, len(bundles))
    metrics: dict[str, Any] = {
        "n_events_plotted": int(selected.size),
        "max_events_per_class": int(max_events_per_class),
        "run_tsne": bool(run_tsne),
        "class_counts_plotted": {
            CLASS_NAMES.get(int(class_id), str(class_id)): int(np.sum(labels == class_id))
            for class_id in sorted(set(int(v) for v in labels.tolist()))
        },
        "models": {},
    }
    reductions: dict[str, np.ndarray] = {"selected_index": selected.astype(np.int64), "labels": labels.astype(np.int64)}
    legend_artists: list[Any] = []
    for col, bundle in enumerate(bundles):
        x = bundle.embeddings[selected]
        pca_coords, tsne_coords, reduction_metrics = reduce_embeddings(x, seed=seed, run_tsne=run_tsne)
        model_metrics = {
            **reduction_metrics,
            "silhouette": safe_silhouette(x, labels),
            "knn5_same_class_rate": knn_same_class_rate(x, labels, k=5),
        }
        metrics["models"][bundle.key] = model_metrics
        reductions[f"{bundle.key}_pca"] = pca_coords
        panels = [(pca_coords, "PCA")]
        if run_tsne:
            reductions[f"{bundle.key}_tsne"] = tsne_coords
            panels.append((tsne_coords, "t-SNE"))
        for row, (coords, name) in enumerate(panels):
            ax = axes[row, col]
            artists = class_scatter(ax, coords, labels)
            if not legend_artists:
                legend_artists = artists
            if row == 0:
                title = (
                    f"{bundle.display_name}\n"
                    f"sil={model_metrics['silhouette']:.3f}, kNN={model_metrics['knn5_same_class_rate']:.3f}"
                )
            else:
                title = f"{name}, trust={model_metrics['trustworthiness']:.3f}"
            ax.set_title(title, pad=4)
            set_axis_style(ax)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.88 if run_tsne else 0.82, bottom=0.22, wspace=0.24, hspace=0.34)
    if legend_artists:
        fig.legend(
            legend_artists,
            [artist.get_label() for artist in legend_artists],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.065),
            ncol=len(legend_artists),
            frameon=False,
            fontsize=8,
            markerscale=1.5,
            columnspacing=1.2,
        )
    fig.suptitle("Representation manifolds" if run_tsne else "Representation manifolds (PCA only)", fontsize=12, y=0.965)
    pdf_path, png_path = output_bases(output_dir, "representation_manifold")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    np.savez_compressed(output_dir / "representation_manifold_reductions.npz", **reductions)
    write_strict_json(output_dir / "representation_manifold_metrics.json", metrics)
    return metrics

def safe_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2 or labels.shape[0] <= len(set(labels.tolist())):
        return float("nan")
    try:
        return float(silhouette_score(scale_embeddings(embeddings), labels))
    except ValueError:
        return float("nan")


def split_indices(split: np.ndarray, name: str) -> np.ndarray:
    return np.flatnonzero(split == name).astype(np.int64)


def stratified_fraction_indices(labels: np.ndarray, pool: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Label fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(set(int(labels[i]) for i in pool.tolist())):
        class_idx = pool[labels[pool] == class_id]
        n = class_idx.size if fraction >= 1.0 else max(1, int(round(class_idx.size * fraction)))
        n = min(n, class_idx.size)
        selected.extend(int(i) for i in rng.choice(class_idx, size=n, replace=False).tolist())
    arr = np.asarray(selected, dtype=np.int64)
    rng.shuffle(arr)
    return arr


def fit_probe(x_train: np.ndarray, y_train: np.ndarray) -> Any:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    clf.fit(x_train, y_train)
    return clf


def run_label_efficiency(
    bundles: list[EmbeddingBundle],
    output_dir: Path,
    fractions: list[float],
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    train_idx = split_indices(bundles[0].split, "train")
    test_idx = split_indices(bundles[0].split, "test")
    if test_idx.size == 0:
        test_idx = split_indices(bundles[0].split, "val")
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Label-efficiency requires non-empty train and test/val splits")

    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        for fraction in fractions:
            for repeat in range(repeats):
                sub_train = stratified_fraction_indices(
                    bundle.labels,
                    train_idx,
                    fraction=fraction,
                    seed=seed + repeat * 1009 + int(round(fraction * 1000)),
                )
                clf = fit_probe(bundle.embeddings[sub_train], bundle.labels[sub_train])
                pred = clf.predict(bundle.embeddings[test_idx])
                rows.append(
                    {
                        "model": bundle.key,
                        "display_name": bundle.display_name,
                        "fraction": float(fraction),
                        "repeat": int(repeat),
                        "n_train": int(sub_train.size),
                        "n_eval": int(test_idx.size),
                        "balanced_accuracy": float(balanced_accuracy_score(bundle.labels[test_idx], pred)),
                        "macro_f1": float(f1_score(bundle.labels[test_idx], pred, average="macro", zero_division=0)),
                    }
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "label_efficiency_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_label_efficiency(rows, output_dir)
    write_strict_json(output_dir / "label_efficiency_summary.json", summarize_probe_rows(rows))
    return rows


def summarize_probe_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model in sorted(set(str(row["model"]) for row in rows)):
        summary[model] = {}
        for fraction in sorted(set(float(row["fraction"]) for row in rows if row["model"] == model)):
            values = [float(row["macro_f1"]) for row in rows if row["model"] == model and float(row["fraction"]) == fraction]
            bal = [float(row["balanced_accuracy"]) for row in rows if row["model"] == model and float(row["fraction"]) == fraction]
            summary[model][str(fraction)] = {
                "macro_f1_mean": float(np.mean(values)),
                "macro_f1_std": float(np.std(values)),
                "balanced_accuracy_mean": float(np.mean(bal)),
                "balanced_accuracy_std": float(np.std(bal)),
                "n_repeats": len(values),
            }
    return summary


def plot_label_efficiency(rows: list[dict[str, Any]], output_dir: Path) -> None:
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for model in sorted(set(str(row["model"]) for row in rows)):
        model_rows = [row for row in rows if row["model"] == model]
        display = str(model_rows[0]["display_name"])
        fractions = sorted(set(float(row["fraction"]) for row in model_rows))
        means = []
        stds = []
        for fraction in fractions:
            values = [float(row["macro_f1"]) for row in model_rows if float(row["fraction"]) == fraction]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
        x = np.asarray(fractions, dtype=np.float32) * 100.0
        y = np.asarray(means, dtype=np.float32)
        s = np.asarray(stds, dtype=np.float32)
        ax.plot(x, y, marker="o", linewidth=1.4, label=display)
        ax.fill_between(x, y - s, y + s, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("Labeled train data used (%)")
    ax.set_ylabel("Test macro F1")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    set_axis_style(ax)
    pdf_path, png_path = output_bases(output_dir, "label_efficiency_curve")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def normalized_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    x = scale_embeddings(embeddings)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0e-8)


def nearest_indices_from_normalized(
    x_norm: np.ndarray,
    query_idx: int,
    k: int,
    candidate_idx: np.ndarray | None = None,
) -> np.ndarray:
    if candidate_idx is None:
        sim = x_norm @ x_norm[query_idx]
        sim[query_idx] = -np.inf
        k_eff = min(k, x_norm.shape[0] - 1)
        if k_eff <= 0:
            return np.array([], dtype=np.int64)
        idx = np.argpartition(-sim, kth=np.arange(k_eff))[:k_eff]
        idx = idx[np.argsort(-sim[idx])]
        return idx.astype(np.int64)

    candidates = np.asarray(candidate_idx, dtype=np.int64)
    sim = x_norm[candidates] @ x_norm[query_idx]
    sim[candidates == query_idx] = -np.inf
    k_eff = min(k, max(0, candidates.size - 1))
    if k_eff <= 0:
        return np.array([], dtype=np.int64)
    local = np.argpartition(-sim, kth=np.arange(k_eff))[:k_eff]
    local = local[np.argsort(-sim[local])]
    return candidates[local].astype(np.int64)


def nearest_indices(embeddings: np.ndarray, query_idx: int, k: int) -> np.ndarray:
    return nearest_indices_from_normalized(normalized_embedding_matrix(embeddings), query_idx, k)


def retrieval_metrics(bundle: EmbeddingBundle, k: int, metric_max_per_class: int, seed: int) -> dict[str, Any]:
    labels = bundle.labels
    metric_idx = balanced_indices(labels, max_per_class=metric_max_per_class, seed=seed)
    x_norm = normalized_embedding_matrix(bundle.embeddings)
    same_top1 = []
    same_topk = []
    class_ids = sorted(set(int(labels[i]) for i in metric_idx.tolist()))
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    class_to_pos = {class_id: pos for pos, class_id in enumerate(class_ids)}
    for i in metric_idx:
        nn = nearest_indices_from_normalized(x_norm, int(i), k=k, candidate_idx=metric_idx)
        same_top1.append(bool(labels[nn[0]] == labels[i]) if nn.size else False)
        same_topk.append(float(np.mean(labels[nn] == labels[i])) if nn.size else float("nan"))
        if nn.size:
            matrix[class_to_pos[int(labels[i])], class_to_pos[int(labels[nn[0]])]] += 1
    return {
        "n_queries": int(metric_idx.size),
        "metric_max_per_class": int(metric_max_per_class),
        "top1_same_class_rate": float(np.mean(same_top1)),
        f"top{k}_same_class_rate": float(np.nanmean(same_topk)),
        "class_ids": class_ids,
        "top1_retrieval_matrix": matrix.tolist(),
    }


def plot_retrieval_sheet(
    bundles: list[EmbeddingBundle],
    output_dir: Path,
    signals: np.ndarray | None,
    metadata: dict[str, dict[str, str]],
    queries_per_class: int,
    neighbors: int,
    metric_max_per_class: int,
    seed: int,
) -> dict[str, Any]:
    selected_queries = balanced_indices(bundles[0].labels, max_per_class=queries_per_class, seed=seed)
    metrics = {
        bundle.key: retrieval_metrics(bundle, k=neighbors, metric_max_per_class=metric_max_per_class, seed=seed)
        for bundle in bundles
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_strict_json(output_dir / "retrieval_metrics.json", metrics)

    if signals is not None:
        pdf_path = output_dir / "nearest_neighbor_retrieval_sheet.pdf"
        with PdfPages(pdf_path) as pdf:
            for bundle in bundles:
                x_norm = normalized_embedding_matrix(bundle.embeddings)
                for query_idx in selected_queries:
                    nn = nearest_indices_from_normalized(x_norm, int(query_idx), k=neighbors)
                    indices = np.concatenate([[query_idx], nn])
                    fig, axes = plt.subplots(1, indices.size, figsize=(2.7 * indices.size, 2.4), sharey=True, constrained_layout=True)
                    if indices.size == 1:
                        axes = np.asarray([axes])
                    for ax, idx, rank in zip(axes, indices, range(indices.size)):
                        ax.plot(signals[int(idx)], color="black", linewidth=0.8)
                        cls = CLASS_NAMES.get(int(bundle.labels[int(idx)]), str(bundle.labels[int(idx)]))
                        event_id = str(bundle.event_id[int(idx)])
                        sample = metadata.get(event_id, {}).get("sample_id", event_id.split("/")[-1])
                        prefix = "Q" if rank == 0 else f"N{rank}"
                        ax.set_title(f"{prefix}: {cls}\n{sample[:28]}", fontsize=7)
                        ax.set_xticks([])
                        set_axis_style(ax)
                    fig.suptitle(bundle.display_name, fontsize=10)
                    pdf.savefig(fig)
                    plt.close(fig)

    plot_retrieval_metric_summary(metrics, bundles, output_dir)
    return metrics


def plot_retrieval_metric_summary(metrics: dict[str, Any], bundles: list[EmbeddingBundle], output_dir: Path) -> None:
    apply_pub_style()
    model_keys = [bundle.key for bundle in bundles]
    top1 = [float(metrics[key]["top1_same_class_rate"]) for key in model_keys]
    topk_key = next(key for key in metrics[model_keys[0]].keys() if key.startswith("top") and key.endswith("_same_class_rate") and key != "top1_same_class_rate")
    topk = [float(metrics[key][topk_key]) for key in model_keys]
    x = np.arange(len(model_keys))
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    width = 0.36
    ax.bar(x - width / 2, top1, width=width, label="Top-1 same class")
    ax.bar(x + width / 2, topk, width=width, label=topk_key.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(key, key) for key in model_keys], rotation=20, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Retrieval purity")
    ax.legend(frameon=False, fontsize=8)
    set_axis_style(ax)
    pdf_path, png_path = output_bases(output_dir, "retrieval_purity")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def plot_reconstruction_diagnostic(
    output_dir: Path,
    history_json: Path | None,
    eval_metrics_json: Path | None,
) -> dict[str, Any]:
    history = []
    if history_json is not None and history_json.is_file():
        history = json.load(history_json.open())
    eval_metrics: dict[str, float] = {}
    if eval_metrics_json is not None and eval_metrics_json.is_file():
        eval_metrics = json.load(eval_metrics_json.open())

    apply_pub_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
    status = {
        "history_path": "" if history_json is None else str(history_json),
        "eval_metrics_path": "" if eval_metrics_json is None else str(eval_metrics_json),
        "history_found": bool(history),
        "eval_metrics_found": bool(eval_metrics),
    }
    if history:
        epochs = [int(row["epoch"]) for row in history]
        for key in ["train_loss", "masked_mse", "masked_mae", "derivative_mse"]:
            if key in history[0]:
                axes[0].plot(epochs, [float(row[key]) for row in history], marker="o", linewidth=1.2, label=key)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Metric")
        axes[0].legend(frameon=False, fontsize=7)
        axes[0].set_title("Masked reconstruction training")
    else:
        axes[0].text(0.5, 0.5, "No reconstruction history found", ha="center", va="center", fontsize=10)
        axes[0].set_axis_off()

    if eval_metrics:
        keys = list(eval_metrics.keys())
        values = [float(eval_metrics[key]) for key in keys]
        axes[1].bar(np.arange(len(keys)), values)
        axes[1].set_xticks(np.arange(len(keys)))
        axes[1].set_xticklabels(keys, rotation=35, ha="right", fontsize=7)
        axes[1].set_title("Validation reconstruction metrics")
    else:
        axes[1].text(0.5, 0.5, "No eval metrics found", ha="center", va="center", fontsize=10)
        axes[1].set_axis_off()
    for ax in axes:
        if ax.axison:
            set_axis_style(ax)
    pdf_path, png_path = output_bases(output_dir, "reconstruction_diagnostic")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    write_strict_json(output_dir / "reconstruction_diagnostic_summary.json", status)
    return status


def perturb_signal_batch(signals: np.ndarray, perturbation: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = signals.astype(np.float32, copy=True)
    if perturbation == "noise_0p10":
        scale = np.std(x, axis=1, keepdims=True).clip(min=1.0e-6)
        return x + rng.normal(0.0, 0.10, size=x.shape).astype(np.float32) * scale
    if perturbation == "scale_1p25":
        return x * 1.25
    if perturbation == "shift_8":
        return np.roll(x, shift=8, axis=1)
    if perturbation == "center_mask_64":
        width = min(64, x.shape[1])
        start = max(0, x.shape[1] // 2 - width // 2)
        end = min(x.shape[1], start + width)
        x[:, start:end] = 0.0
        return x
    raise ValueError(f"Unsupported perturbation: {perturbation}")


def cosine_distances_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1.0e-8)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1.0e-8)
    return (1.0 - np.sum(a_norm * b_norm, axis=1)).astype(np.float32)


@torch.no_grad()
def encode_pretrained_signals(model_key: str, signals: np.ndarray, batch_size: int, device: torch.device, cache_dir: Path) -> np.ndarray:
    from p3_ssl.pretrained_backbones import (
        encode_batch,
        load_moment_official_model,
        load_patchtst_1ch_model,
    )

    if model_key == "moment_official":
        model = load_moment_official_model(cache_dir=cache_dir, device=device, seq_len=signals.shape[1])
    elif model_key == "patchtst_pretrained":
        model, _ = load_patchtst_1ch_model(cache_dir=cache_dir, device=device)
    else:
        raise ValueError(f"Robustness encoding supports moment_official and patchtst_pretrained, got {model_key}")

    chunks: list[np.ndarray] = []
    for start in range(0, signals.shape[0], batch_size):
        batch = torch.from_numpy(signals[start : start + batch_size]).float()
        features = encode_batch(model_key, model, batch, device)
        chunks.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def run_robustness_grid(
    output_dir: Path,
    model_keys: list[str],
    signals: np.ndarray,
    labels: np.ndarray,
    max_events_per_class: int,
    batch_size: int,
    device: str,
    cache_dir: Path,
    seed: int,
) -> dict[str, Any]:
    selected = balanced_indices(labels, max_per_class=max_events_per_class, seed=seed)
    selected_signals = signals[selected]
    perturbations = ["noise_0p10", "scale_1p25", "shift_8", "center_mask_64"]
    result: dict[str, Any] = {"n_events": int(selected.size), "models": {}}

    for model_key in model_keys:
        base = encode_pretrained_signals(model_key, selected_signals, batch_size=batch_size, device=torch.device(device), cache_dir=cache_dir)
        model_rows: dict[str, Any] = {}
        for perturbation in perturbations:
            perturbed = perturb_signal_batch(selected_signals, perturbation, seed=seed)
            emb = encode_pretrained_signals(model_key, perturbed, batch_size=batch_size, device=torch.device(device), cache_dir=cache_dir)
            dist = cosine_distances_rows(base, emb)
            model_rows[perturbation] = {
                "cosine_distance_mean": float(np.mean(dist)),
                "cosine_distance_std": float(np.std(dist)),
                "cosine_distance_median": float(np.median(dist)),
            }
        result["models"][model_key] = model_rows

    output_dir.mkdir(parents=True, exist_ok=True)
    write_strict_json(output_dir / "robustness_metrics.json", result)
    plot_robustness_grid(result, output_dir)
    return result


def plot_robustness_grid(result: dict[str, Any], output_dir: Path) -> None:
    apply_pub_style()
    models = list(result["models"].keys())
    perturbations = list(next(iter(result["models"].values())).keys()) if models else []
    values = np.asarray(
        [
            [float(result["models"][model][perturbation]["cosine_distance_mean"]) for perturbation in perturbations]
            for model in models
        ],
        dtype=np.float32,
    )
    fig, ax = plt.subplots(figsize=(1.2 * max(4, len(perturbations)), 0.65 * max(4, len(models)) + 1.4), constrained_layout=True)
    im = ax.imshow(values, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(perturbations)))
    ax.set_xticklabels(perturbations, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels([DISPLAY_NAMES.get(model, model) for model in models])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=7)
    ax.set_title("Embedding sensitivity to nuisance perturbations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean cosine distance")
    pdf_path, png_path = output_bases(output_dir, "robustness_invariance_grid")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def write_robustness_placeholder(output_dir: Path, reason: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": "not_run", "reason": reason}
    write_strict_json(output_dir / "robustness_metrics.json", payload)
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.2), constrained_layout=True)
    ax.text(0.5, 0.5, reason, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    pdf_path, png_path = output_bases(output_dir, "robustness_invariance_grid")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return payload



def _summary_metric(label_summary: dict[str, Any], model_key: str, fraction: str) -> float | None:
    item = label_summary.get(model_key, {}).get(fraction)
    if item is None:
        return None
    return float(item.get("balanced_accuracy_mean", float("nan")))


def _robustness_metric(robustness: dict[str, Any], model_key: str, perturbation: str) -> float | None:
    item = robustness.get("models", {}).get(model_key, {}).get(perturbation)
    if item is None:
        return None
    return float(item.get("cosine_distance_mean", float("nan")))


def write_assessment_dashboard(
    output_dir: Path,
    bundles: list[EmbeddingBundle],
    manifold: dict[str, Any],
    label_summary: dict[str, Any],
    retrieval: dict[str, Any],
    robustness: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        model_metrics = manifold.get("models", {}).get(bundle.key, {})
        retrieval_metrics_for_model = retrieval.get(bundle.key, {})
        rows.append(
            {
                "model": bundle.key,
                "display_name": bundle.display_name,
                "probe_bal_acc_1pct": _summary_metric(label_summary, bundle.key, "0.01"),
                "probe_bal_acc_10pct": _summary_metric(label_summary, bundle.key, "0.1"),
                "probe_bal_acc_100pct": _summary_metric(label_summary, bundle.key, "1.0"),
                "knn5_same_class_rate": model_metrics.get("knn5_same_class_rate"),
                "retrieval_top1_same_class_rate": retrieval_metrics_for_model.get("top1_same_class_rate"),
                "shift8_cosine_distance": _robustness_metric(robustness, bundle.key, "shift_8"),
                "center_mask64_cosine_distance": _robustness_metric(robustness, bundle.key, "center_mask_64"),
            }
        )
    write_strict_json(output_dir / "assessment_dashboard.json", rows)

    columns = [
        ("probe_bal_acc_1pct", "Probe 1%", False),
        ("probe_bal_acc_10pct", "Probe 10%", False),
        ("probe_bal_acc_100pct", "Probe 100%", False),
        ("knn5_same_class_rate", "kNN@5", False),
        ("retrieval_top1_same_class_rate", "Retrieval@1", False),
        ("shift8_cosine_distance", "Shift dist", True),
        ("center_mask64_cosine_distance", "Mask dist", True),
    ]
    values = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
    texts: list[list[str]] = []
    for i, row in enumerate(rows):
        text_row: list[str] = []
        for j, (key, _, lower_is_better) in enumerate(columns):
            raw = row.get(key)
            if raw is None or not np.isfinite(float(raw)):
                text_row.append("--")
                continue
            value = float(raw)
            values[i, j] = max(0.0, 1.0 - min(value, 1.0)) if lower_is_better else value
            text_row.append(f"{value:.3f}")
        texts.append(text_row)

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(1.25 * len(columns) + 3.0, 0.55 * max(3, len(rows)) + 1.8), constrained_layout=True)
    masked = np.ma.masked_invalid(values)
    ax.imshow(masked, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels([label for _, label, _ in columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([row["display_name"] for row in rows])
    for i in range(len(rows)):
        for j in range(len(columns)):
            ax.text(j, i, texts[i][j], ha="center", va="center", fontsize=7, color="white" if np.isfinite(values[i, j]) and values[i, j] < 0.45 else "black")
    ax.set_title("Canonical representation assessment dashboard", pad=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    pdf_path, png_path = output_bases(output_dir, "assessment_dashboard")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return rows


def auto_detect_reconstruction_paths(output_dir: Path, embedding_root: Path, history_json: Path | None, eval_metrics_json: Path | None) -> tuple[Path | None, Path | None]:
    bases = [output_dir, output_dir.parent, output_dir.parent.parent, embedding_root, embedding_root.parent]
    if history_json is None:
        for base in bases:
            for candidate in [base / "history.json", base / "reconstruction_moment_like_60ep" / "history.json"]:
                if candidate.is_file():
                    history_json = candidate
                    break
            if history_json is not None:
                break
    if eval_metrics_json is None:
        for base in bases:
            for candidate in [base / "eval_val" / "metrics.json", base / "reconstruction_moment_like_60ep" / "eval_val" / "metrics.json"]:
                if candidate.is_file():
                    eval_metrics_json = candidate
                    break
            if eval_metrics_json is not None:
                break
    return history_json, eval_metrics_json

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate P3 SSL assessment figures from cached embeddings and optional checkpoints.")
    parser.add_argument("--embedding-root", type=Path, default=ROOT / "outputs" / "pretrained_backbones" / "particles2snr_f_3class_same_input_moment_patchtst_conv1dgap")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "ssl_assessment")
    parser.add_argument("--models", default="moment_official,patchtst_pretrained,conv1dgap_same_input_3class")
    parser.add_argument("--aligned-inputs", type=Path, default=None)
    parser.add_argument("--event-metadata", type=Path, default=None)
    parser.add_argument("--include-raw-baseline", action="store_true")
    parser.add_argument("--include-random-baseline", action="store_true")
    parser.add_argument("--random-projection-dim", type=int, default=128)
    parser.add_argument("--max-events-per-class", type=int, default=500)
    parser.add_argument("--label-fractions", default="0.01,0.05,0.10,0.25,1.0")
    parser.add_argument("--probe-repeats", type=int, default=5)
    parser.add_argument("--retrieval-queries-per-class", type=int, default=2)
    parser.add_argument("--retrieval-neighbors", type=int, default=5)
    parser.add_argument("--retrieval-metric-max-per-class", type=int, default=500)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--history-json", type=Path, default=None)
    parser.add_argument("--eval-metrics-json", type=Path, default=None)
    parser.add_argument("--run-robustness", action="store_true")
    parser.add_argument("--robustness-models", default="moment_official,patchtst_pretrained")
    parser.add_argument("--robustness-events-per-class", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs" / "hf_cache")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    aligned_path = args.aligned_inputs
    if aligned_path is None:
        for candidate_name in ("aligned_inputs.npz", "aligned_512_inputs.npz"):
            candidate = args.embedding_root / candidate_name
            if candidate.is_file():
                aligned_path = candidate
                break
    metadata_path = args.event_metadata
    if metadata_path is None:
        candidate = args.embedding_root / "events_metadata.csv"
        metadata_path = candidate if candidate.is_file() else None

    aligned = load_aligned_inputs(aligned_path)
    bundles = load_embedding_bundles(
        embedding_root=args.embedding_root,
        model_keys=parse_csv_strings(args.models),
        aligned_inputs=aligned,
        include_raw_baseline=args.include_raw_baseline,
        include_random_baseline=args.include_random_baseline,
        random_projection_dim=args.random_projection_dim,
        seed=args.seed,
    )
    metadata = read_event_metadata(metadata_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history_json, eval_metrics_json = auto_detect_reconstruction_paths(args.output_dir, args.embedding_root, args.history_json, args.eval_metrics_json)
    summary: dict[str, Any] = {
        "embedding_root": str(args.embedding_root),
        "aligned_inputs": "" if aligned_path is None else str(aligned_path),
        "event_metadata": "" if metadata_path is None else str(metadata_path),
        "models": [bundle.key for bundle in bundles],
        "class_counts": {
            CLASS_NAMES.get(int(class_id), str(class_id)): int(np.sum(bundles[0].labels == class_id))
            for class_id in sorted(set(int(v) for v in bundles[0].labels.tolist()))
        },
    }
    summary["reconstruction"] = plot_reconstruction_diagnostic(args.output_dir, history_json, eval_metrics_json)
    summary["manifold"] = plot_manifold_figure(
        bundles,
        output_dir=args.output_dir,
        max_events_per_class=args.max_events_per_class,
        seed=args.seed,
        run_tsne=not args.skip_tsne,
    )
    summary["label_efficiency"] = summarize_probe_rows(
        run_label_efficiency(
            bundles,
            output_dir=args.output_dir,
            fractions=parse_csv_floats(args.label_fractions),
            repeats=args.probe_repeats,
            seed=args.seed,
        )
    )
    signals = None if aligned is None else aligned["signals"]
    summary["retrieval"] = plot_retrieval_sheet(
        bundles,
        output_dir=args.output_dir,
        signals=signals,
        metadata=metadata,
        queries_per_class=args.retrieval_queries_per_class,
        neighbors=args.retrieval_neighbors,
        metric_max_per_class=args.retrieval_metric_max_per_class,
        seed=args.seed,
    )
    if args.run_robustness:
        if aligned is None:
            summary["robustness"] = write_robustness_placeholder(args.output_dir, "Robustness was requested, but aligned input signals were not provided.")
        else:
            summary["robustness"] = run_robustness_grid(
                output_dir=args.output_dir,
                model_keys=parse_csv_strings(args.robustness_models),
                signals=aligned["signals"],
                labels=aligned["labels"],
                max_events_per_class=args.robustness_events_per_class,
                batch_size=args.batch_size,
                device=args.device,
                cache_dir=args.cache_dir,
                seed=args.seed,
            )
    else:
        summary["robustness"] = write_robustness_placeholder(
            args.output_dir,
            "Robustness encodes perturbed signals and can be expensive; rerun with --run-robustness to generate model-based distances.",
        )

    summary["dashboard"] = write_assessment_dashboard(
        args.output_dir,
        bundles=bundles,
        manifold=summary["manifold"],
        label_summary=summary["label_efficiency"],
        retrieval=summary["retrieval"],
        robustness=summary["robustness"],
    )
    write_strict_json(args.output_dir / "ssl_assessment_summary.json", summary)
    print(json.dumps({"output_dir": str(args.output_dir), "models": summary["models"]}, sort_keys=True))


if __name__ == "__main__":
    main()
