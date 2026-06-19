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
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
P0_ROOT = REPO_ROOT / "P0"
for path_entry in (ROOT, P0_ROOT):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

from scripts.plot_snr_threshold_manifolds import (  # noqa: E402
    MODEL_DISPLAY,
    PARTICLE_CLASS_NAMES,
    YEAST_CLASS_NAMES,
    DatasetSpec,
    align_embeddings_to_metadata,
    apply_pub_style,
    balanced_indices,
    build_datasets,
    parse_csv_strings,
    quantile_grid,
    quantile_thresholds,
    set_axis_style,
    write_csv,
)
from scripts.run_ssl_assessment_figures import (  # noqa: E402
    fit_probe,
    nearest_indices_from_normalized,
    normalized_embedding_matrix,
    safe_silhouette,
)


@dataclass(frozen=True)
class ImpactBundle:
    model_key: str
    embeddings: np.ndarray
    event_id: np.ndarray
    split: np.ndarray | None


@dataclass(frozen=True)
class EvaluationSet:
    mode: str
    indices: np.ndarray
    labels: np.ndarray
    snr: np.ndarray
    event_id: np.ndarray


def load_impact_bundle(path: Path, model_key: str) -> ImpactBundle:
    if not path.is_file():
        raise FileNotFoundError(f"Missing embedding file for {model_key}: {path}")
    with np.load(path, allow_pickle=True) as data:
        required = {"embeddings", "event_id"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing keys: {sorted(missing)}")
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        event_id = np.asarray(data["event_id"]).astype(str)
        split = np.asarray(data["split"]).astype(str) if "split" in data.files else None
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings for {model_key}, got {embeddings.shape}")
    if embeddings.shape[0] != event_id.shape[0]:
        raise ValueError(f"Embedding/event_id row mismatch for {model_key}")
    if split is not None and split.shape[0] != event_id.shape[0]:
        raise ValueError(f"Split/event_id row mismatch for {model_key}")
    return ImpactBundle(model_key=model_key, embeddings=embeddings, event_id=event_id, split=split)


def align_split_to_metadata(bundle: ImpactBundle, metadata_event_id: np.ndarray) -> np.ndarray | None:
    if bundle.split is None:
        return None
    event_to_idx = {event_id: idx for idx, event_id in enumerate(bundle.event_id.tolist())}
    indices: list[int] = []
    missing: list[str] = []
    for event_id in metadata_event_id.astype(str).tolist():
        idx = event_to_idx.get(event_id)
        if idx is None:
            missing.append(event_id)
        else:
            indices.append(idx)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{bundle.model_key} is missing {len(missing)} metadata events, e.g. {preview}")
    return bundle.split[np.asarray(indices, dtype=np.int64)]


def class_enrichment(labels: np.ndarray, low_mask: np.ndarray, class_names: tuple[str, str, str]) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    n_total = int(labels.size)
    n_low = int(low_mask.sum())
    for class_id, class_name in enumerate(class_names):
        class_mask = labels == class_id
        n_class = int(class_mask.sum())
        n_class_low = int(np.logical_and(class_mask, low_mask).sum())
        baseline_fraction = float(n_class / n_total) if n_total else float("nan")
        low_fraction = float(n_class_low / n_low) if n_low else float("nan")
        enrichment = float(low_fraction / baseline_fraction) if baseline_fraction > 0 and math.isfinite(low_fraction) else float("nan")
        result[f"{class_name}_n"] = n_class
        result[f"{class_name}_low_snr_n"] = n_class_low
        result[f"{class_name}_baseline_fraction"] = baseline_fraction
        result[f"{class_name}_low_snr_fraction_of_low"] = low_fraction
        result[f"{class_name}_low_snr_enrichment"] = enrichment
    return result


def knn_impurity(
    x_norm: np.ndarray,
    labels: np.ndarray,
    query_idx: np.ndarray,
    candidate_idx: np.ndarray,
    k: int,
) -> float:
    query_idx = np.asarray(query_idx, dtype=np.int64)
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if query_idx.size == 0 or candidate_idx.size <= 1:
        return float("nan")
    values: list[float] = []
    for idx in query_idx.tolist():
        nn = nearest_indices_from_normalized(x_norm, int(idx), k=k, candidate_idx=candidate_idx)
        if nn.size:
            values.append(float(np.mean(labels[nn] != labels[int(idx)])))
    return float(np.mean(values)) if values else float("nan")


def precompute_neighbor_indices(x_norm: np.ndarray, k: int) -> np.ndarray:
    n_samples = x_norm.shape[0]
    if n_samples <= 1:
        return np.empty((n_samples, 0), dtype=np.int64)
    sim = x_norm @ x_norm.T
    np.fill_diagonal(sim, -np.inf)
    k_eff = min(k, n_samples - 1)
    nn = np.argpartition(-sim, kth=np.arange(k_eff), axis=1)[:, :k_eff]
    row = np.arange(n_samples)[:, None]
    order = np.argsort(-sim[row, nn], axis=1)
    return np.take_along_axis(nn, order, axis=1).astype(np.int64)


def impurity_from_neighbors(labels: np.ndarray, neighbor_idx: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0 or neighbor_idx.shape[1] == 0:
        return float("nan")
    same = labels[neighbor_idx[mask]] == labels[mask, None]
    return float(np.mean(~same))


def top1_confusion_from_neighbors(
    labels: np.ndarray,
    neighbor_idx: np.ndarray,
    mask: np.ndarray,
    class_ids: list[int],
) -> tuple[np.ndarray, float]:
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    class_to_pos = {class_id: pos for pos, class_id in enumerate(class_ids)}
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0 or neighbor_idx.shape[1] == 0:
        return matrix, float("nan")
    query_idx = np.flatnonzero(mask)
    nn = neighbor_idx[query_idx, 0]
    src_labels = labels[query_idx]
    dst_labels = labels[nn]
    for src, dst in zip(src_labels.tolist(), dst_labels.tolist()):
        matrix[class_to_pos[int(src)], class_to_pos[int(dst)]] += 1
    return matrix, float(np.mean(src_labels == dst_labels))


def top1_confusion(
    x_norm: np.ndarray,
    labels: np.ndarray,
    query_idx: np.ndarray,
    candidate_idx: np.ndarray,
    class_ids: list[int],
) -> tuple[np.ndarray, float]:
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    class_to_pos = {class_id: pos for pos, class_id in enumerate(class_ids)}
    same: list[bool] = []
    query_idx = np.asarray(query_idx, dtype=np.int64)
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if query_idx.size == 0 or candidate_idx.size <= 1:
        return matrix, float("nan")
    for idx in query_idx.tolist():
        nn = nearest_indices_from_normalized(x_norm, int(idx), k=1, candidate_idx=candidate_idx)
        if nn.size == 0:
            continue
        src = int(labels[int(idx)])
        dst = int(labels[int(nn[0])])
        matrix[class_to_pos[src], class_to_pos[dst]] += 1
        same.append(src == dst)
    return matrix, float(np.mean(same)) if same else float("nan")


def thresholded_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    max_events_per_class: int,
    seed: int,
) -> float:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) < 3:
        return float("nan")
    idx = np.flatnonzero(mask).astype(np.int64)
    if max_events_per_class > 0:
        local = balanced_indices(labels[idx], max_per_class=max_events_per_class, seed=seed)
        idx = idx[local]
    if idx.size < 3 or np.unique(labels[idx]).size < 2:
        return float("nan")
    return safe_silhouette(embeddings[idx], labels[idx])


def cap_indices_per_class(labels: np.ndarray, pool: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.int64)
    if max_per_class <= 0:
        return pool
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(set(int(labels[i]) for i in pool.tolist())):
        class_idx = pool[labels[pool] == class_id]
        if class_idx.size > max_per_class:
            class_idx = rng.choice(class_idx, size=max_per_class, replace=False)
        selected.extend(int(i) for i in class_idx.tolist())
    arr = np.asarray(selected, dtype=np.int64)
    rng.shuffle(arr)
    return arr


def split_for_probe(labels: np.ndarray, split: np.ndarray | None, seed: int, max_train_per_class: int = 0) -> tuple[np.ndarray, np.ndarray, str]:
    if split is not None:
        train_idx = np.flatnonzero(split == "train")
        test_idx = np.flatnonzero(split == "test")
        if test_idx.size == 0:
            test_idx = np.flatnonzero(split == "val")
        if train_idx.size > 0 and test_idx.size > 0 and np.unique(labels[train_idx]).size >= 2 and np.unique(labels[test_idx]).size >= 2:
            train_idx = cap_indices_per_class(labels, train_idx, max_train_per_class, seed=seed)
            return train_idx.astype(np.int64), test_idx.astype(np.int64), "existing_split"

    indices = np.arange(labels.shape[0], dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2 or counts.min() < 2:
        return indices, np.array([], dtype=np.int64), "unavailable"
    train_idx, test_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=labels)
    train_idx = cap_indices_per_class(labels, np.asarray(train_idx, dtype=np.int64), max_train_per_class, seed=seed)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64), "stratified_70_30"


def probe_predictions(
    embeddings: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray | None,
    seed: int,
    max_train_per_class: int = 0,
) -> dict[str, Any]:
    train_idx, test_idx, source = split_for_probe(labels, split, seed=seed, max_train_per_class=max_train_per_class)
    result: dict[str, Any] = {
        "split_source": source,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "pred": np.asarray([], dtype=np.int64),
    }
    if source == "unavailable" or test_idx.size == 0:
        return result
    clf = fit_probe(embeddings[train_idx], labels[train_idx])
    pred = clf.predict(embeddings[test_idx])
    result["balanced_accuracy"] = float(balanced_accuracy_score(labels[test_idx], pred))
    result["macro_f1"] = float(f1_score(labels[test_idx], pred, average="macro", zero_division=0))
    result["pred"] = np.asarray(pred, dtype=np.int64)
    return result


def error_rate(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(y_true[mask] != y_pred[mask]))


def add_probe_error_columns(
    row: dict[str, Any],
    probe: dict[str, Any],
    labels: np.ndarray,
    snr: np.ndarray,
    threshold: float,
    class_names: tuple[str, str, str],
) -> None:
    test_idx = np.asarray(probe["test_idx"], dtype=np.int64)
    pred = np.asarray(probe["pred"], dtype=np.int64)
    row["probe_split_source"] = str(probe["split_source"])
    row["probe_n_train"] = int(np.asarray(probe["train_idx"]).size)
    row["probe_n_eval"] = int(test_idx.size)
    row["probe_balanced_accuracy"] = float(probe["balanced_accuracy"])
    row["probe_macro_f1"] = float(probe["macro_f1"])
    if test_idx.size == 0:
        row["probe_error_low_snr"] = float("nan")
        row["probe_error_high_snr"] = float("nan")
        row["probe_error_lift"] = float("nan")
        return

    y_true = labels[test_idx]
    test_snr = snr[test_idx]
    low = test_snr <= float(threshold)
    high = ~low
    low_error = error_rate(y_true, pred, low)
    high_error = error_rate(y_true, pred, high)
    row["probe_error_low_snr"] = low_error
    row["probe_error_high_snr"] = high_error
    row["probe_error_lift"] = float(low_error - high_error) if math.isfinite(low_error) and math.isfinite(high_error) else float("nan")
    for class_id, class_name in enumerate(class_names):
        class_mask = y_true == class_id
        class_low_error = error_rate(y_true, pred, np.logical_and(low, class_mask))
        class_high_error = error_rate(y_true, pred, np.logical_and(high, class_mask))
        row[f"{class_name}_probe_error_low_snr"] = class_low_error
        row[f"{class_name}_probe_error_high_snr"] = class_high_error
        row[f"{class_name}_probe_error_lift"] = (
            float(class_low_error - class_high_error)
            if math.isfinite(class_low_error) and math.isfinite(class_high_error)
            else float("nan")
        )


def evaluate_model_thresholds(
    dataset: DatasetSpec,
    eval_set: EvaluationSet,
    model_key: str,
    embeddings: np.ndarray,
    split: np.ndarray | None,
    thresholds: np.ndarray,
    k: int,
    seed: int,
    probe_max_train_per_class: int,
    silhouette_max_events_per_class: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = eval_set.labels
    snr = eval_set.snr
    class_ids = sorted(int(v) for v in np.unique(labels).tolist())
    x_norm = normalized_embedding_matrix(embeddings)
    neighbor_idx = precompute_neighbor_indices(x_norm, k=max(k, 1))
    probe = probe_predictions(embeddings, labels, split, seed=seed, max_train_per_class=probe_max_train_per_class)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "display_name": MODEL_DISPLAY.get(model_key, model_key),
        "control_note": (
            "Conv1D-GAP is a supervised particle control and should be interpreted as OOD on yeast."
            if dataset.key == "yeast" and model_key == "conv1dgap_same_input_3class"
            else ""
        ),
        "mode": eval_set.mode,
        "class_ids": class_ids,
        "thresholds": {},
    }
    for quantile, threshold in zip(quantile_grid().tolist(), thresholds.tolist()):
        low = snr <= float(threshold)
        high = ~low
        low_matrix, low_top1_same = top1_confusion_from_neighbors(labels, neighbor_idx, low, class_ids)
        high_matrix, high_top1_same = top1_confusion_from_neighbors(labels, neighbor_idx, high, class_ids)
        low_impurity = impurity_from_neighbors(labels, neighbor_idx[:, :k], low)
        high_impurity = impurity_from_neighbors(labels, neighbor_idx[:, :k], high)
        low_sil = thresholded_silhouette(embeddings, labels, low, max_events_per_class=silhouette_max_events_per_class, seed=seed)
        high_sil = thresholded_silhouette(embeddings, labels, high, max_events_per_class=silhouette_max_events_per_class, seed=seed)
        row: dict[str, Any] = {
            "dataset": dataset.key,
            "mode": eval_set.mode,
            "model": model_key,
            "display_name": MODEL_DISPLAY.get(model_key, model_key),
            "quantile": float(quantile),
            "threshold": float(threshold),
            "snr_label": dataset.snr_label,
            "n_total": int(labels.size),
            "n_low_snr": int(low.sum()),
            "n_high_snr": int(high.sum()),
            "low_snr_fraction": float(low.mean()) if labels.size else float("nan"),
            "knn_k": int(k),
            "low_knn_impurity": low_impurity,
            "high_knn_impurity": high_impurity,
            "knn_impurity_delta": (
                float(low_impurity - high_impurity)
                if math.isfinite(low_impurity) and math.isfinite(high_impurity)
                else float("nan")
            ),
            "low_top1_same_class_rate": low_top1_same,
            "high_top1_same_class_rate": high_top1_same,
            "top1_same_class_delta": (
                float(low_top1_same - high_top1_same)
                if math.isfinite(low_top1_same) and math.isfinite(high_top1_same)
                else float("nan")
            ),
            "low_silhouette": low_sil,
            "high_silhouette": high_sil,
            "silhouette_delta": float(low_sil - high_sil) if math.isfinite(low_sil) and math.isfinite(high_sil) else float("nan"),
        }
        row.update(class_enrichment(labels, low, dataset.class_names))
        add_probe_error_columns(row, probe, labels, snr, float(threshold), dataset.class_names)
        rows.append(row)
        threshold_key = f"q{int(round(float(quantile) * 100)):02d}"
        details["thresholds"][threshold_key] = {
            "quantile": float(quantile),
            "threshold": float(threshold),
            "low_top1_retrieval_matrix": low_matrix.tolist(),
            "high_top1_retrieval_matrix": high_matrix.tolist(),
            "low_top1_retrieval_matrix_row_normalized": row_normalize(low_matrix).tolist(),
            "high_top1_retrieval_matrix_row_normalized": row_normalize(high_matrix).tolist(),
        }
    return rows, details


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix_f = np.asarray(matrix, dtype=np.float64)
    denom = matrix_f.sum(axis=1, keepdims=True)
    return np.divide(matrix_f, denom, out=np.full_like(matrix_f, np.nan), where=denom > 0)


def build_evaluation_sets(dataset: DatasetSpec, max_events_per_class: int, seed: int) -> list[EvaluationSet]:
    labels = dataset.metadata["plot_class_id"].to_numpy(dtype=np.int64)
    snr = dataset.metadata["snr_value"].to_numpy(dtype=np.float64)
    event_id = dataset.metadata["event_id"].astype(str).to_numpy()
    full_idx = np.arange(labels.shape[0], dtype=np.int64)
    visual_idx = balanced_indices(labels, max_per_class=max_events_per_class, seed=seed)
    return [
        EvaluationSet("full_dataset", full_idx, labels, snr, event_id),
        EvaluationSet("visual_subset", visual_idx, labels[visual_idx], snr[visual_idx], event_id[visual_idx]),
    ]


def subset_split(split: np.ndarray | None, indices: np.ndarray) -> np.ndarray | None:
    if split is None:
        return None
    return split[np.asarray(indices, dtype=np.int64)]


def plot_metric_curves(dataset_dir: Path, rows: list[dict[str, Any]], dataset: DatasetSpec) -> None:
    if not rows:
        return
    apply_pub_style()
    for mode in sorted({str(row["mode"]) for row in rows}):
        mode_rows = [row for row in rows if str(row["mode"]) == mode]
        models = sorted({str(row["model"]) for row in mode_rows})
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
        for model in models:
            model_rows = sorted([row for row in mode_rows if row["model"] == model], key=lambda row: float(row["quantile"]))
            x = np.asarray([float(row["quantile"]) for row in model_rows]) * 100.0
            impurity_delta = np.asarray([float(row["knn_impurity_delta"]) for row in model_rows], dtype=np.float64)
            error_lift = np.asarray([float(row["probe_error_lift"]) for row in model_rows], dtype=np.float64)
            axes[0].plot(x, impurity_delta, marker="o", linewidth=1.2, label=MODEL_DISPLAY.get(model, model))
            axes[1].plot(x, error_lift, marker="o", linewidth=1.2, label=MODEL_DISPLAY.get(model, model))
        axes[0].axhline(0.0, color="#555555", linewidth=0.8)
        axes[1].axhline(0.0, color="#555555", linewidth=0.8)
        axes[0].set_xlabel("SNR threshold quantile (%)")
        axes[0].set_ylabel("Low - high kNN impurity")
        axes[1].set_xlabel("SNR threshold quantile (%)")
        axes[1].set_ylabel("Low - high probe error")
        for ax in axes:
            ax.legend(frameon=False, fontsize=7)
            set_axis_style(ax)
        fig.suptitle(f"{dataset.display_name} SNR impact metrics ({mode})", fontsize=11)
        fig.savefig(dataset_dir / f"{mode}_snr_impact_curves.pdf")
        fig.savefig(dataset_dir / f"{mode}_snr_impact_curves.png", dpi=220)
        plt.close(fig)


def plot_confusion_heatmaps(dataset_dir: Path, details_by_model: dict[str, Any], dataset: DatasetSpec) -> None:
    apply_pub_style()
    class_names = list(dataset.class_names)
    for mode in sorted({str(model_details["mode"]) for model_details in details_by_model.values()}):
        mode_details = {model: d for model, d in details_by_model.items() if str(d["mode"]) == mode}
        for model, details in mode_details.items():
            display_model = model.removesuffix(f"_{mode}")
            threshold_items = sorted(details["thresholds"].items(), key=lambda item: float(item[1]["quantile"]))
            pair_names: list[str] = []
            heat_values: list[list[float]] = []
            for src_i, src_name in enumerate(class_names):
                for dst_i, dst_name in enumerate(class_names):
                    if src_i == dst_i:
                        continue
                    pair_names.append(f"{src_name}->{dst_name}")
                    heat_values.append([float(item[1]["low_top1_retrieval_matrix_row_normalized"][src_i][dst_i]) for item in threshold_items])
            heat = np.asarray(heat_values, dtype=np.float64)
            fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
            image = ax.imshow(heat, aspect="auto", interpolation="nearest", vmin=0.0, vmax=np.nanmax(heat) if np.isfinite(heat).any() else 1.0)
            ax.set_yticks(np.arange(len(pair_names)))
            ax.set_yticklabels(pair_names, fontsize=7)
            ax.set_xticks(np.arange(len(threshold_items)))
            ax.set_xticklabels([f"{float(item[1]['quantile']) * 100:.0f}" for item in threshold_items], fontsize=7)
            ax.set_xlabel("SNR threshold quantile (%)")
            ax.set_title(f"{MODEL_DISPLAY.get(display_model, display_model)} low-SNR top-1 cross-class retrieval ({mode})", fontsize=9)
            fig.colorbar(image, ax=ax, label="row-normalized rate")
            fig.savefig(dataset_dir / f"{mode}_{display_model}_low_snr_top1_confusion_heatmap.pdf")
            fig.savefig(dataset_dir / f"{mode}_{display_model}_low_snr_top1_confusion_heatmap.png", dpi=220)
            plt.close(fig)


def process_dataset(
    dataset: DatasetSpec,
    model_keys: list[str],
    output_dir: Path,
    max_events_per_class: int,
    k: int,
    seed: int,
    probe_max_train_per_class: int = 500,
    silhouette_max_events_per_class: int = 250,
) -> dict[str, Any]:
    dataset_dir = output_dir / dataset.key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    evaluation_sets = build_evaluation_sets(dataset, max_events_per_class=max_events_per_class, seed=seed)
    thresholds_by_mode = {
        eval_set.mode: quantile_thresholds(eval_set.snr)
        for eval_set in evaluation_sets
    }
    all_rows: list[dict[str, Any]] = []
    json_models: dict[str, Any] = {}
    metadata_event_id = dataset.metadata["event_id"].astype(str).to_numpy()
    for model_key in model_keys:
        bundle = load_impact_bundle(dataset.embedding_root / model_key / "all_embeddings.npz", model_key)
        aligned_embeddings = align_embeddings_to_metadata(bundle, dataset.metadata)
        aligned_split = align_split_to_metadata(bundle, metadata_event_id)
        json_models[model_key] = {}
        for eval_set in evaluation_sets:
            indices = eval_set.indices
            rows, details = evaluate_model_thresholds(
                dataset=dataset,
                eval_set=eval_set,
                model_key=model_key,
                embeddings=aligned_embeddings[indices],
                split=subset_split(aligned_split, indices),
                thresholds=thresholds_by_mode[eval_set.mode],
                k=k,
                seed=seed,
                probe_max_train_per_class=probe_max_train_per_class,
                silhouette_max_events_per_class=silhouette_max_events_per_class,
            )
            all_rows.extend(rows)
            json_models[model_key][eval_set.mode] = details
    write_csv(dataset_dir / "snr_embedding_impact_metrics.csv", all_rows)
    summary = {
        "dataset": dataset.key,
        "display_name": dataset.display_name,
        "embedding_root": str(dataset.embedding_root),
        "n_events_available": int(dataset.metadata.shape[0]),
        "max_events_per_class": int(max_events_per_class),
        "knn_k": int(k),
        "probe_max_train_per_class": int(probe_max_train_per_class),
        "silhouette_max_events_per_class": int(silhouette_max_events_per_class),
        "class_names": list(dataset.class_names),
        "snr_label": dataset.snr_label,
        "quantiles": [float(v) for v in quantile_grid().tolist()],
        "threshold_source": {
            mode: "computed within this evaluation mode"
            for mode in thresholds_by_mode
        },
        "models": json_models,
    }
    with (dataset_dir / "snr_embedding_impact_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    plot_metric_curves(dataset_dir, all_rows, dataset)
    flat_details = {
        f"{model}_{mode}": details
        for model, by_mode in json_models.items()
        for mode, details in by_mode.items()
    }
    plot_confusion_heatmaps(dataset_dir, flat_details, dataset)
    return {
        "dataset": dataset.key,
        "n_rows": len(all_rows),
        "csv": str(dataset_dir / "snr_embedding_impact_metrics.csv"),
        "json": str(dataset_dir / "snr_embedding_impact_metrics.json"),
    }


def run(args: argparse.Namespace) -> None:
    model_keys = parse_csv_strings(args.models)
    missing_model_keys = [key for key in model_keys if key not in MODEL_DISPLAY]
    if missing_model_keys:
        raise ValueError(f"Unsupported model keys: {missing_model_keys}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {"models": model_keys, "datasets": {}}
    for dataset in build_datasets(args):
        summaries["datasets"][dataset.key] = process_dataset(
            dataset=dataset,
            model_keys=model_keys,
            output_dir=args.output_dir,
            max_events_per_class=args.max_events_per_class,
            k=args.knn_k,
            seed=args.seed,
            probe_max_train_per_class=args.probe_max_train_per_class,
            silhouette_max_events_per_class=args.silhouette_max_events_per_class,
        )
    with (args.output_dir / "snr_embedding_impact_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
    print(json.dumps({"output_dir": str(args.output_dir), "datasets": sorted(summaries["datasets"].keys())}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SNR-conditioned embedding overlap and confusion metrics for particles and yeast.")
    parser.add_argument("--particle-root", type=Path, default=ROOT / "outputs" / "pretrained_backbones" / "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap")
    parser.add_argument("--particle-manifest", type=Path, default=REPO_ROOT / "particles2SNR_pipeline" / "output" / "p0_c1_Particles2SNR_F" / "event_classification_dataset" / "event_manifest.csv")
    parser.add_argument("--yeast-event-root", type=Path, default=ROOT / "outputs" / "pretrained_backbones" / "yeast_passage_events_p3_512")
    parser.add_argument("--yeast-embedding-root", type=Path, default=ROOT / "outputs" / "pretrained_backbones" / "particles2snr_f_3class_plus_yeast_moment_patchtst_conv1dgap")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "snr_threshold_metrics")
    parser.add_argument("--models", default="moment_official,patchtst_pretrained,conv1dgap_same_input_3class")
    parser.add_argument("--yeast-classes", default="mix,budding,shmoo2")
    parser.add_argument("--max-events-per-class", type=int, default=500)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--probe-max-train-per-class", type=int, default=500)
    parser.add_argument("--silhouette-max-events-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
