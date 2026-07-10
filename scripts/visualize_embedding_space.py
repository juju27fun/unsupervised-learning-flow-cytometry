#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

from p3_ssl.config import load_config
from p3_ssl.data import ManifestRow, read_manifest
from p3_ssl.decimation import crop_or_pad, decimate_signal, ensure_1d_signal, normalize_signal
from p3_ssl.embedding import (
    CLASS_NAMES,
    EventRecord,
    balanced_event_indices,
    collect_events,
    event_records_to_metadata,
    pool_token_embeddings,
    token_indices_for_interval,
)
from p3_ssl.masking import PatchSpec
from p3_ssl.models import MomentLikeConfig, MomentLikeReconstructor


COLORS = {
    0: "#0072B2",
    1: "#009E73",
    2: "#D55E00",
    3: "#CC79A7",
}
DISPLAY_NAMES = {
    "moment": "MOMENT-like P3 SSL",
    "patchtst": "PatchTST random",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_moment_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> MomentLikeReconstructor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_config = checkpoint.get("config", config)
    data_cfg = ckpt_config["data"]
    patch_cfg = ckpt_config["patching"]
    model_cfg = ckpt_config["model"]
    model = MomentLikeReconstructor(
        MomentLikeConfig(
            input_length=int(data_cfg["input_length_ssl"]),
            patch_size=int(patch_cfg["patch_size"]),
            patch_stride=int(patch_cfg["patch_stride"]),
            d_model=int(model_cfg["d_model"]),
            n_heads=int(model_cfg["n_heads"]),
            n_layers=int(model_cfg["n_layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            activation=str(model_cfg.get("activation", "gelu")),
            max_tokens=int(model_cfg.get("max_tokens", 1024)),
        )
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def make_patchtst_model(
    patch_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    seed: int,
    device: torch.device,
) -> torch.nn.Module:
    from detseg.models.backbones.patchtst1d import PatchTST1DBackbone

    torch.manual_seed(seed)
    model = PatchTST1DBackbone(
        in_channels=1,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        proj_channels=embed_dim,
        dropout=0.0,
        attn_dropout=0.0,
    )
    model.to(device).eval()
    return model


def load_signal(row: ManifestRow, config: dict[str, Any]) -> np.ndarray:
    data_cfg = config["data"]
    signal = ensure_1d_signal(np.load(row.signal_path))
    signal = crop_or_pad(signal, int(data_cfg["input_length_raw"]), mode="center")
    signal = decimate_signal(signal, int(data_cfg["decimation_factor"]), method="mean")
    signal = crop_or_pad(signal, int(data_cfg["input_length_ssl"]), mode="center")
    return normalize_signal(signal, mode=str(data_cfg.get("normalization", "window_zscore")))


def row_key(row: ManifestRow) -> tuple[str, str, str]:
    return (row.split, row.sample_id, str(row.signal_path))


def event_key(event: EventRecord) -> tuple[str, str, str]:
    return (event.split, event.sample_id, str(event.signal_path))


def filter_rows(rows: list[ManifestRow], split: str, max_signals: int | None, seed: int) -> list[ManifestRow]:
    if split != "all":
        rows = [row for row in rows if row.split == split]
    if max_signals is not None and len(rows) > max_signals:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(np.arange(len(rows)), size=max_signals, replace=False))
        rows = [rows[int(i)] for i in idx]
    return rows


def extract_embeddings(
    rows: list[ManifestRow],
    config: dict[str, Any],
    moment: MomentLikeReconstructor,
    patchtst: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[EventRecord]]:
    input_length = int(config["data"]["input_length_ssl"])
    patch_size = int(config["patching"]["patch_size"])
    patch_stride = int(config["patching"]["patch_stride"])
    spec = PatchSpec(input_length=input_length, patch_size=patch_size, patch_stride=patch_stride)

    events = collect_events(rows, input_length=input_length, class_names=CLASS_NAMES)
    events_by_row: dict[tuple[str, str, str], list[EventRecord]] = defaultdict(list)
    for event in events:
        events_by_row[event_key(event)].append(event)
    rows_with_events = [row for row in rows if row_key(row) in events_by_row]
    if not rows_with_events:
        raise RuntimeError("No labeled events found in the selected rows")

    ordered_events: list[EventRecord] = []
    moment_embeddings: list[np.ndarray] = []
    patchtst_embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start_idx in range(0, len(rows_with_events), batch_size):
            batch_rows = rows_with_events[start_idx : start_idx + batch_size]
            signals = [load_signal(row, config) for row in batch_rows]
            signal_tensor = torch.from_numpy(np.stack(signals)).float().unsqueeze(1).to(device)
            moment_tokens = moment.encode(signal_tensor, token_mask=None)
            patch_tokens = patchtst.forward_features(signal_tensor)[0].transpose(1, 2)

            if moment_tokens.shape[1] != spec.n_tokens:
                raise RuntimeError(f"MOMENT token count {moment_tokens.shape[1]} != expected {spec.n_tokens}")
            if patch_tokens.shape[1] != spec.n_tokens:
                raise RuntimeError(f"PatchTST token count {patch_tokens.shape[1]} != expected {spec.n_tokens}")

            for batch_i, row in enumerate(batch_rows):
                for event in events_by_row[row_key(row)]:
                    token_idx = token_indices_for_interval(event.start, event.end, spec)
                    moment_vec = pool_token_embeddings(moment_tokens[batch_i], token_idx)
                    patch_vec = pool_token_embeddings(patch_tokens[batch_i], token_idx)
                    moment_embeddings.append(moment_vec.detach().cpu().numpy().astype(np.float32))
                    patchtst_embeddings.append(patch_vec.detach().cpu().numpy().astype(np.float32))
                    ordered_events.append(event)

    return {
        "moment": np.stack(moment_embeddings),
        "patchtst": np.stack(patchtst_embeddings),
    }, ordered_events


def standardize_embeddings(x: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(x)


def compute_reductions(x: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x_std = standardize_embeddings(x)
    n_samples, n_features = x_std.shape
    if n_samples < 2:
        zeros = np.zeros((n_samples, 2), dtype=np.float32)
        return zeros, zeros, {"trustworthiness": float("nan")}

    pca = PCA(n_components=2, random_state=seed)
    pca_coords = pca.fit_transform(x_std).astype(np.float32)

    if n_samples < 5:
        tsne_coords = pca_coords.copy()
        trust = float("nan")
    else:
        pre_dim = min(50, n_features, n_samples - 1)
        x_pre = PCA(n_components=pre_dim, random_state=seed).fit_transform(x_std) if pre_dim < n_features else x_std
        perplexity = min(30, max(2, (n_samples - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        tsne_coords = tsne.fit_transform(x_pre).astype(np.float32)
        trust_neighbors = min(10, max(1, (n_samples // 2) - 1))
        if trust_neighbors >= 1:
            trust = float(trustworthiness(x_std, tsne_coords, n_neighbors=trust_neighbors))
        else:
            trust = float("nan")

    return pca_coords, tsne_coords, {
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "trustworthiness": trust,
    }


def crossval_scores(x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float]:
    classes, counts = np.unique(y, return_counts=True)
    result = {
        "silhouette": float("nan"),
        "knn_balanced_accuracy": float("nan"),
        "linear_probe_balanced_accuracy": float("nan"),
    }
    if len(classes) < 2 or len(y) <= len(classes):
        return result

    x_std = standardize_embeddings(x)
    try:
        result["silhouette"] = float(silhouette_score(x_std, y))
    except ValueError:
        pass

    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        return result

    knn_scores: list[float] = []
    linear_scores: list[float] = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(x, y):
        scaler = StandardScaler().fit(x[train_idx])
        x_train = scaler.transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        _, train_counts = np.unique(y[train_idx], return_counts=True)
        k = int(min(5, train_counts.min()))
        if k >= 1:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y[train_idx])
            knn_scores.append(float(balanced_accuracy_score(y[test_idx], knn.predict(x_test))))
        linear = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        linear.fit(x_train, y[train_idx])
        linear_scores.append(float(balanced_accuracy_score(y[test_idx], linear.predict(x_test))))

    if knn_scores:
        result["knn_balanced_accuracy"] = float(np.mean(knn_scores))
    if linear_scores:
        result["linear_probe_balanced_accuracy"] = float(np.mean(linear_scores))
    return result


def plot_class_space(
    output_pdf: Path,
    coords: dict[str, dict[str, np.ndarray]],
    class_ids: np.ndarray,
    class_names: np.ndarray,
    metrics: dict[str, dict[str, float]],
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for col, model_key in enumerate(["moment", "patchtst"]):
        for row, reduction in enumerate(["pca", "tsne"]):
            ax = axes[row, col]
            xy = coords[model_key][reduction]
            for class_id in sorted(set(int(c) for c in class_ids.tolist())):
                mask = class_ids == class_id
                label = CLASS_NAMES.get(class_id, str(class_id))
                ax.scatter(
                    xy[mask, 0],
                    xy[mask, 1],
                    s=10,
                    alpha=0.72,
                    c=COLORS.get(class_id, "#666666"),
                    label=label,
                    linewidths=0,
                )
            title = f"{DISPLAY_NAMES[model_key]} - {reduction.upper()}"
            if reduction == "pca":
                title += f"\nSilhouette={metrics[model_key].get('silhouette', float('nan')):.3f}"
            else:
                title += f"\nTrust={metrics[model_key].get('trustworthiness', float('nan')):.3f}"
            ax.set_title(title)
            ax.set_xlabel("dim 1")
            ax.set_ylabel("dim 2")
            if col == 1 and row == 0:
                ax.legend(loc="best", frameon=False, markerscale=1.5)
    fig.suptitle("Event-level latent spaces on Particles2SNR C1 4-class labels", fontsize=14)
    fig.savefig(output_pdf)
    plt.close(fig)


def plot_width_space(
    output_pdf: Path,
    coords: dict[str, dict[str, np.ndarray]],
    width_norm: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for col, model_key in enumerate(["moment", "patchtst"]):
        for row, reduction in enumerate(["pca", "tsne"]):
            ax = axes[row, col]
            xy = coords[model_key][reduction]
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=width_norm, cmap="viridis", s=10, alpha=0.75, linewidths=0)
            ax.set_title(f"{DISPLAY_NAMES[model_key]} - {reduction.upper()} colored by event width")
            ax.set_xlabel("dim 1")
            ax.set_ylabel("dim 2")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="YOLO width norm")
    fig.savefig(output_pdf)
    plt.close(fig)


def write_balanced_csv(
    output_csv: Path,
    events: list[EventRecord],
    selected_idx: np.ndarray,
    coords: dict[str, dict[str, np.ndarray]],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "event_index",
        "event_id",
        "sample_id",
        "split",
        "class_id",
        "class_name",
        "center_norm",
        "width_norm",
        "start",
        "end",
        "moment_pca_x",
        "moment_pca_y",
        "moment_tsne_x",
        "moment_tsne_y",
        "patchtst_pca_x",
        "patchtst_pca_y",
        "patchtst_tsne_x",
        "patchtst_tsne_y",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for local_i, event_i in enumerate(selected_idx):
            event = events[int(event_i)]
            writer.writerow(
                {
                    "event_index": int(event_i),
                    "event_id": event.event_id,
                    "sample_id": event.sample_id,
                    "split": event.split,
                    "class_id": event.class_id,
                    "class_name": event.class_name,
                    "center_norm": event.center_norm,
                    "width_norm": event.width_norm,
                    "start": event.start,
                    "end": event.end,
                    "moment_pca_x": float(coords["moment"]["pca"][local_i, 0]),
                    "moment_pca_y": float(coords["moment"]["pca"][local_i, 1]),
                    "moment_tsne_x": float(coords["moment"]["tsne"][local_i, 0]),
                    "moment_tsne_y": float(coords["moment"]["tsne"][local_i, 1]),
                    "patchtst_pca_x": float(coords["patchtst"]["pca"][local_i, 0]),
                    "patchtst_pca_y": float(coords["patchtst"]["pca"][local_i, 1]),
                    "patchtst_tsne_x": float(coords["patchtst"]["tsne"][local_i, 0]),
                    "patchtst_tsne_y": float(coords["patchtst"]["tsne"][local_i, 1]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize event-level MOMENT-like and PatchTST embedding spaces.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--moment-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--max-events-per-class", type=int, default=500)
    parser.add_argument("--max-signals", type=int, default=None, help="Optional smoke limit before event extraction.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patchtst-embed-dim", type=int, default=128)
    parser.add_argument("--patchtst-depth", type=int, default=4)
    parser.add_argument("--patchtst-heads", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    rows = filter_rows(read_manifest(args.manifest, split=None), split=args.split, max_signals=args.max_signals, seed=args.seed)
    if not rows:
        raise SystemExit("No manifest rows selected")

    device = torch.device(args.device)
    patch_size = int(config["patching"]["patch_size"])
    moment = make_moment_model(config, args.moment_checkpoint, device=device)
    patchtst = make_patchtst_model(
        patch_size=patch_size,
        embed_dim=args.patchtst_embed_dim,
        depth=args.patchtst_depth,
        num_heads=args.patchtst_heads,
        seed=args.seed,
        device=device,
    )

    embeddings, events = extract_embeddings(
        rows=rows,
        config=config,
        moment=moment,
        patchtst=patchtst,
        device=device,
        batch_size=args.batch_size,
    )
    metadata = event_records_to_metadata(events)
    class_ids = metadata["class_id"]
    selected_idx = balanced_event_indices(class_ids, max_per_class=args.max_events_per_class, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "embeddings_all.npz",
        moment_embeddings=embeddings["moment"],
        patchtst_embeddings=embeddings["patchtst"],
        **metadata,
    )

    selected_class_ids = class_ids[selected_idx]
    selected_class_names = metadata["class_name"][selected_idx]
    selected_widths = metadata["width_norm"][selected_idx]
    coords: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for model_key in ["moment", "patchtst"]:
        selected_embeddings = embeddings[model_key][selected_idx]
        pca_coords, tsne_coords, reduction_metrics = compute_reductions(selected_embeddings, seed=args.seed)
        model_metrics = crossval_scores(selected_embeddings, selected_class_ids, seed=args.seed)
        model_metrics.update(reduction_metrics)
        coords[model_key] = {"pca": pca_coords, "tsne": tsne_coords}
        metrics[model_key] = model_metrics

    class_counts = {
        CLASS_NAMES.get(int(class_id), str(class_id)): int(np.sum(class_ids == class_id))
        for class_id in sorted(set(int(c) for c in class_ids.tolist()))
    }
    selected_class_counts = {
        CLASS_NAMES.get(int(class_id), str(class_id)): int(np.sum(selected_class_ids == class_id))
        for class_id in sorted(set(int(c) for c in selected_class_ids.tolist()))
    }
    metrics_payload = {
        "n_events_total": int(len(events)),
        "n_events_plotted": int(len(selected_idx)),
        "class_counts_total": class_counts,
        "class_counts_plotted": selected_class_counts,
        "split": args.split,
        "max_signals": args.max_signals,
        "max_events_per_class": args.max_events_per_class,
        "seed": args.seed,
        "models": metrics,
    }
    with (args.output_dir / "embedding_metrics.json").open("w") as f:
        json.dump(metrics_payload, f, indent=2, sort_keys=True)

    write_balanced_csv(args.output_dir / "embeddings_balanced.csv", events, selected_idx, coords)
    plot_class_space(
        args.output_dir / "embedding_space_pca_tsne.pdf",
        coords=coords,
        class_ids=selected_class_ids,
        class_names=selected_class_names,
        metrics=metrics,
    )
    plot_width_space(
        args.output_dir / "embedding_space_by_width.pdf",
        coords=coords,
        width_norm=selected_widths,
    )
    print(json.dumps(metrics_payload, sort_keys=True))


if __name__ == "__main__":
    main()
