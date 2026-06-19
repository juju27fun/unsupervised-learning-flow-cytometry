#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
P0_ROOT = REPO_ROOT / "P0"
for path_entry in (ROOT, P0_ROOT):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

from p3_ssl.pretrained_backbones import (
    MOMENT_DEFAULT_ID,
    PATCHTST_DEFAULT_ID,
    SWIN_DEFAULT_ID,
    ParticleEvent,
    collect_particle_events,
    encode_batch as encode_pretrained_batch,
    load_moment_official_model,
    load_patchtst_1ch_model,
    load_swin_model,
    save_transfer_report,
    write_event_metadata,
)
from p3_ssl.decimation import crop_or_pad, ensure_1d_signal


CLASS_COLORS = {
    0: "#0072B2",
    1: "#009E73",
    2: "#D55E00",
    3: "#CC79A7",
}
CLASS_LABELS = {
    0: "2um",
    1: "4um",
    2: "10um",
    3: "unclear",
}

MODEL_DISPLAY = {
    "moment_official": "MOMENT official pretrained",
    "patchtst_pretrained": "PatchTST HF pretrained",
    "conv1dgap_4class": "Conv1D-GAP supervised 4-class",
    "swin2d_pretrained": "Swin-2D spectrogram pretrained",
}

COMPARISON_MODEL_ORDER = [
    "moment_official",
    "patchtst_pretrained",
    "conv1dgap_4class",
    "swin2d_pretrained",
]

CONV1DGAP_DEFAULT_CHECKPOINT = P0_ROOT / "outputs" / "training" / "output" / "Conv1DGAP-dataset_4c-zoo-conv1dgap" / "best_model.pth"
MOMENT_OFFICIAL_PATCH_LEN = 8
MOMENT_OFFICIAL_PATCH_STRIDE = 8
PATCHTST_PRETRAIN_CONTEXT_LENGTH = 512
PATCHTST_PRETRAIN_PATCH_LENGTH = 12
PATCHTST_PRETRAIN_PATCH_STRIDE = 12
PATCHTST_FORECASTING_PAPER_PATCH_LENGTH = 16
PATCHTST_FORECASTING_PAPER_PATCH_STRIDE = 8


class EventSignalDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> None:
        self.signals = signals.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.indices = indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[idx])
        return (
            torch.from_numpy(self.signals[sample_idx]).float(),
            torch.tensor(int(self.labels[sample_idx]), dtype=torch.long),
            torch.tensor(sample_idx, dtype=torch.long),
        )


class EncoderClassifier(nn.Module):
    def __init__(self, model_key: str, encoder: nn.Module, feature_dim: int, num_classes: int, device: torch.device) -> None:
        super().__init__()
        self.model_key = model_key
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, num_classes)
        self.device = device

    def features(self, signals: torch.Tensor) -> torch.Tensor:
        return encode_batch(self.model_key, self.encoder, signals, self.device)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(signals))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(events: list[ParticleEvent]) -> dict[str, np.ndarray]:
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, event in enumerate(events):
        if event.split in splits:
            splits[event.split].append(i)
    return {key: np.asarray(value, dtype=np.int64) for key, value in splits.items()}


def parse_csv_strings(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def filter_and_remap_classes(
    events: list[ParticleEvent],
    signals: np.ndarray,
    include_class_names: list[str],
) -> tuple[list[ParticleEvent], np.ndarray]:
    if not include_class_names:
        return events, signals
    class_to_id = {name: idx for idx, name in enumerate(include_class_names)}
    selected_events: list[ParticleEvent] = []
    selected_indices: list[int] = []
    for idx, event in enumerate(events):
        if event.class_name not in class_to_id:
            continue
        selected_indices.append(idx)
        selected_events.append(
            ParticleEvent(
                event_id=event.event_id,
                sample_id=event.sample_id,
                split=event.split,
                signal_path=event.signal_path,
                label_path=event.label_path,
                class_id=class_to_id[event.class_name],
                class_name=event.class_name,
                center_norm=event.center_norm,
                width_norm=event.width_norm,
                center_index=event.center_index,
                crop_start=event.crop_start,
                crop_end=event.crop_end,
            )
        )
    present = {event.class_name for event in selected_events}
    missing = [name for name in include_class_names if name not in present]
    if missing:
        raise ValueError(f"Requested class names were not found: {missing}")
    if not selected_events:
        raise ValueError("Class filtering removed all events")
    selected = np.asarray(selected_indices, dtype=np.int64)
    return selected_events, signals[selected]


def infer_feature_dim(model_key: str, encoder, signals: np.ndarray, device: torch.device) -> int:
    batch = torch.from_numpy(signals[: min(2, len(signals))]).float()
    encoder.eval()
    with torch.no_grad():
        features = encode_batch(model_key, encoder, batch, device)
    return int(features.shape[-1])


def encode_conv1dgap_features(model: nn.Module, signals: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    x = signals.to(device).unsqueeze(1)
    x = torch.relu(model.bn1(model.conv1(x)))
    x = model.drop1(model.pool1(x))
    x = torch.relu(model.bn2(model.conv2(x)))
    x = model.drop2(model.pool2(x))
    x = torch.relu(model.bn3(model.conv3(x)))
    x = model.drop3(model.pool3(x))
    x = model.flatten(model.gap(x))
    return torch.relu(model.fc1(x))


def encode_batch(model_key: str, model, signals: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    if model_key == "conv1dgap_4class":
        return encode_conv1dgap_features(model, signals, device=device)
    return encode_pretrained_batch(model_key, model, signals, device)


def adaptive_bandpass_decimate_np(
    signal: np.ndarray,
    target_length: int,
    native_length: int,
    native_fs_hz: float,
    low_khz: float,
    high_khz_max: float,
) -> np.ndarray:
    if native_length % target_length != 0:
        raise ValueError(
            f"native_length ({native_length}) must be divisible by target_length ({target_length})"
        )
    x = crop_or_pad(ensure_1d_signal(signal), native_length, mode="center").astype(np.float32, copy=False)
    decimate_factor = native_length // target_length
    post_fs_hz = native_fs_hz / decimate_factor
    high_cutoff_hz = min(high_khz_max * 1000.0, 0.9 * (post_fs_hz / 2.0))
    low_cutoff_hz = low_khz * 1000.0
    spectrum = np.fft.fft(x)
    freqs = np.fft.fftfreq(x.size, d=1.0 / native_fs_hz)
    mask = (np.abs(freqs) >= low_cutoff_hz) & (np.abs(freqs) <= high_cutoff_hz)
    filtered = np.fft.ifft(spectrum * mask).real.astype(np.float32)
    return filtered[::decimate_factor].astype(np.float32, copy=False)


def build_conv1dgap_native_signals(args: argparse.Namespace, events: list[ParticleEvent]) -> np.ndarray:
    signals: list[np.ndarray] = []
    cache: dict[str, np.ndarray] = {}
    native_fs_hz = float(args.conv1dgap_sample_rate_mhz) * 1_000_000.0
    for event in events:
        raw = cache.get(event.signal_path)
        if raw is None:
            raw = np.load(event.signal_path).astype(np.float32, copy=False)
            cache[event.signal_path] = raw
        signals.append(
            adaptive_bandpass_decimate_np(
                raw,
                target_length=int(args.conv1dgap_input_length),
                native_length=int(args.conv1dgap_native_length),
                native_fs_hz=native_fs_hz,
                low_khz=float(args.conv1dgap_bandpass_low_khz),
                high_khz_max=float(args.conv1dgap_bandpass_high_khz),
            )
        )
    return np.stack(signals).astype(np.float32, copy=False)


def select_model_signals(args: argparse.Namespace, model_key: str, signals: np.ndarray, events: list[ParticleEvent]) -> np.ndarray:
    if model_key != "conv1dgap_4class" or args.conv1dgap_input_mode == "event_crop_512":
        return signals
    if args.conv1dgap_input_mode == "native_p0":
        return build_conv1dgap_native_signals(args, events)
    raise ValueError(f"Unsupported Conv1D-GAP input mode: {args.conv1dgap_input_mode}")


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(trainable)


def evaluate_classifier(
    classifier: EncoderClassifier,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    classifier.eval()
    labels: list[int] = []
    preds: list[int] = []
    with torch.no_grad():
        for signals, y, _ in loader:
            logits = classifier(signals.to(device))
            pred = logits.argmax(dim=-1).detach().cpu().numpy()
            preds.extend(int(p) for p in pred.tolist())
            labels.extend(int(v) for v in y.numpy().tolist())
    if not labels:
        return {"accuracy": float("nan"), "balanced_accuracy": float("nan"), "macro_f1": float("nan")}
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def train_classifier_stage(
    classifier: EncoderClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    train_encoder: bool,
) -> dict[str, Any]:
    set_encoder_trainable(classifier.encoder, train_encoder)
    classifier.train()
    params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1.0e-4)
    criterion = nn.CrossEntropyLoss()
    best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}
    best_val = -float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        classifier.train()
        losses: list[float] = []
        for signals, y, _ in train_loader:
            signals = signals.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(signals)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val = evaluate_classifier(classifier, val_loader, device)
        score = val["macro_f1"]
        history.append({"epoch": float(epoch), "loss": float(np.mean(losses)), **val})
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()}

    classifier.load_state_dict(best_state)
    return {"best_val_macro_f1": best_val, "history": history}


def encode_all_events(
    model_key: str,
    encoder,
    signals: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    encoder.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, signals.shape[0], batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size]).float()
            features = encode_batch(model_key, encoder, batch, device)
            chunks.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def reduce_embeddings(embeddings: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x = StandardScaler().fit_transform(embeddings)
    n_samples, n_features = x.shape
    pca = PCA(n_components=2, random_state=seed)
    pca_coords = pca.fit_transform(x).astype(np.float32)
    if n_samples < 5:
        return pca_coords, pca_coords.copy(), {
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "trustworthiness": float("nan"),
            "tsne_perplexity": float("nan"),
        }
    pre_dim = min(50, n_features, n_samples - 1)
    x_pre = PCA(n_components=pre_dim, random_state=seed).fit_transform(x) if pre_dim < n_features else x
    perplexity = min(30, max(2, (n_samples - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
    tsne_coords = tsne.fit_transform(x_pre).astype(np.float32)
    trust_neighbors = min(10, max(1, (n_samples // 2) - 1))
    trust = float(trustworthiness(x, tsne_coords, n_neighbors=trust_neighbors)) if trust_neighbors >= 1 else float("nan")
    return pca_coords, tsne_coords, {
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "trustworthiness": trust,
        "tsne_perplexity": float(perplexity),
    }


def apply_fig7_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 9,
            "axes.labelsize": 8,
        }
    )


def set_fig7_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#555555")
    ax.grid(False)


def scatter_classes(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, point_size: float = 9.0) -> list[object]:
    artists: list[object] = []
    for class_id in sorted(set(int(v) for v in labels.tolist())):
        mask = labels == class_id
        artist = ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=0.88,
            linewidths=0,
            c=CLASS_COLORS.get(class_id, "#555555"),
            label=CLASS_LABELS.get(class_id, str(class_id)),
        )
        artists.append(artist)
    return artists


def add_class_legend(fig: plt.Figure, artists: list[object], y: float = 0.045) -> None:
    if not artists:
        return
    labels = [artist.get_label() for artist in artists]
    fig.legend(
        artists,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=len(artists),
        frameon=False,
        fontsize=8,
        markerscale=1.5,
        handletextpad=0.35,
        columnspacing=1.2,
    )


def plot_embedding_space(
    pca_coords: np.ndarray,
    tsne_coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_pdf: Path,
    output_png: Path,
) -> None:
    apply_fig7_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.3), constrained_layout=False)
    legend_artists: list[object] = []
    for ax, coords, name in [(axes[0], pca_coords, "PCA"), (axes[1], tsne_coords, "t-SNE")]:
        artists = scatter_classes(ax, coords, labels)
        if not legend_artists:
            legend_artists = artists
        ax.set_title(name, pad=4)
        set_fig7_axis_style(ax)
    fig.suptitle(title, fontsize=12, y=0.965)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.90, bottom=0.15, hspace=0.32)
    add_class_legend(fig, legend_artists, y=0.035)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def plot_pretrained_model_comparison(
    output_pdf: Path,
    output_png: Path,
    model_output_dirs: dict[str, Path],
) -> None:
    model_keys = [key for key in COMPARISON_MODEL_ORDER if key in model_output_dirs]
    if len(model_keys) < 2:
        return

    apply_fig7_plot_style()
    fig_width = max(10.8, 5.1 * len(model_keys))
    fig, axes = plt.subplots(2, len(model_keys), figsize=(fig_width, 6.3), constrained_layout=False, squeeze=False)
    legend_artists: list[object] = []
    for col, model_key in enumerate(model_keys):
        with np.load(model_output_dirs[model_key] / "embeddings.npz", allow_pickle=True) as data:
            labels = data["labels"].astype(np.int64)
            for row, (reduction_key, reduction_name) in enumerate([("pca", "PCA"), ("tsne", "t-SNE")]):
                ax = axes[row, col]
                artists = scatter_classes(ax, data[reduction_key], labels)
                if not legend_artists:
                    legend_artists = artists
                title = f"{MODEL_DISPLAY[model_key]}\n{reduction_name}" if row == 0 else reduction_name
                ax.set_title(title, pad=4)
                set_fig7_axis_style(ax)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.22, wspace=0.22, hspace=0.36)
    add_class_legend(fig, legend_artists, y=0.095)
    caption = (
        "Event-level latent spaces on Particles2SNR C1 4-class labels. "
        "MOMENT uses its official pretrained 512 / patch 8 / stride 8 setup; "
        "PatchTST uses the HF self-supervised checkpoint setup 512 / patch 12 / stride 12 "
        "(the paper forecasting variant is patch 16 / stride 8). "
        "Conv1D-GAP is a local supervised CNN baseline using its configured P0-style input."
    )
    fig.text(0.055, 0.035, caption, ha="left", va="bottom", fontsize=11, wrap=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)

def save_embedding_outputs(
    output_dir: Path,
    events: list[ParticleEvent],
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int,
    title: str,
    extra_metadata: dict[str, Any],
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pca_coords, tsne_coords, reduction_metrics = reduce_embeddings(embeddings, seed=seed)
    np.savez_compressed(
        output_dir / "embeddings.npz",
        embeddings=embeddings.astype(np.float32),
        labels=labels.astype(np.int64),
        pca=pca_coords.astype(np.float32),
        tsne=tsne_coords.astype(np.float32),
        event_id=np.asarray([e.event_id for e in events]),
        split=np.asarray([e.split for e in events]),
        class_name=np.asarray([e.class_name for e in events]),
    )
    write_event_metadata(output_dir / "metadata.csv", events)
    with (output_dir / "metrics.json").open("w") as f:
        json.dump({"reduction": reduction_metrics, **extra_metadata}, f, indent=2, sort_keys=True)
    plot_embedding_space(
        pca_coords=pca_coords,
        tsne_coords=tsne_coords,
        labels=labels,
        title=title,
        output_pdf=output_dir / "embedding_space_pca_tsne.pdf",
        output_png=output_dir / "embedding_space_pca_tsne.png",
    )
    return reduction_metrics


def patchtst_config_value(config: Any, name: str) -> Any:
    return getattr(config, name, None)


def patchtst_native_metadata(model) -> dict[str, Any]:
    config = model.config
    context_length = patchtst_config_value(config, "context_length")
    patch_length = patchtst_config_value(config, "patch_length")
    patch_stride = patchtst_config_value(config, "patch_stride")
    expected = {
        "context_length": PATCHTST_PRETRAIN_CONTEXT_LENGTH,
        "patch_length": PATCHTST_PRETRAIN_PATCH_LENGTH,
        "patch_stride": PATCHTST_PRETRAIN_PATCH_STRIDE,
    }
    actual = {
        "context_length": int(context_length),
        "patch_length": int(patch_length),
        "patch_stride": int(patch_stride),
    }
    if actual != expected:
        raise ValueError(
            "PatchTST checkpoint config does not match the self-supervised pretrained setup: "
            f"expected {expected}, got {actual}"
        )
    return {
        **actual,
        "num_input_channels": int(config.num_input_channels),
        "paper_forecasting_patch_length": PATCHTST_FORECASTING_PAPER_PATCH_LENGTH,
        "paper_forecasting_patch_stride": PATCHTST_FORECASTING_PAPER_PATCH_STRIDE,
        "used_pretraining_patch_length": PATCHTST_PRETRAIN_PATCH_LENGTH,
        "used_pretraining_patch_stride": PATCHTST_PRETRAIN_PATCH_STRIDE,
        "native_parameter_note": (
            "This checkpoint follows the PatchTST masked-representation/pretraining setup "
            "(512 context, patch 12, stride 12). The paper's forecasting variants use patch 16, stride 8."
        ),
    }


def load_conv1dgap_4class_model(
    checkpoint_path: Path,
    device: torch.device | str,
    model_name: str,
    input_length: int,
):
    from models import create_model

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing Conv1D-GAP checkpoint: {checkpoint_path}")
    model = create_model(model_name, input_length=input_length, num_classes=4)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {key: value for key, value in state.items() if key not in {"total_ops", "total_params"}}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def load_encoder(
    model_key: str,
    model_id: str,
    cache_dir: Path,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
):
    if model_key == "moment_official":
        model = load_moment_official_model(model_id=model_id, cache_dir=cache_dir, device=device, seq_len=args.event_length)
        if int(args.event_length) != 512:
            raise ValueError("Official MOMENT pretrained comparison expects --event-length 512")
        return model, {
            "source_model_id": model_id,
            "input_representation": "512-sample event crop, MOMENT RevIN internal normalization",
            "seq_len": int(args.event_length),
            "patch_len": MOMENT_OFFICIAL_PATCH_LEN,
            "patch_stride_len": MOMENT_OFFICIAL_PATCH_STRIDE,
            "native_parameter_note": "Official MOMENT-1-large pretrained embedding path uses seq_len 512 and patch 8/8.",
        }
    if model_key == "patchtst_pretrained":
        model, report = load_patchtst_1ch_model(model_id=model_id, cache_dir=cache_dir, device=device)
        save_transfer_report(output_dir / "patchtst_weight_transfer_report.json", report)
        metadata = {**asdict(report), **patchtst_native_metadata(model)}
        metadata["input_representation"] = "512-sample event crop, 1-channel PatchTST input, HF pretrained weights transferred from source channels"
        return model, metadata
    if model_key == "conv1dgap_4class":
        input_length = int(args.event_length) if args.conv1dgap_input_mode == "event_crop_512" else int(args.conv1dgap_input_length)
        model = load_conv1dgap_4class_model(
            args.conv1dgap_checkpoint,
            device=device,
            model_name=args.conv1dgap_model_name,
            input_length=input_length,
        )
        metadata = {
            "source_model_id": str(args.conv1dgap_checkpoint),
            "model_name": args.conv1dgap_model_name,
            "input_mode": args.conv1dgap_input_mode,
            "input_length": input_length,
            "supervised_local_checkpoint": True,
        }
        if args.conv1dgap_input_mode == "native_p0":
            metadata.update(
                {
                    "input_representation": "full 16384-sample P0 signal, FFT bandpass 5-100 kHz with anti-alias cap, decimated to 4096",
                    "native_length": int(args.conv1dgap_native_length),
                    "bandpass_low_khz": float(args.conv1dgap_bandpass_low_khz),
                    "bandpass_high_khz": float(args.conv1dgap_bandpass_high_khz),
                    "sample_rate_mhz": float(args.conv1dgap_sample_rate_mhz),
                }
            )
        else:
            metadata["input_representation"] = "512-sample event crop, P3 window_zscore normalization"
        return model, metadata
    if model_key == "swin2d_pretrained":
        model = load_swin_model(model_id=model_id, cache_dir=cache_dir, device=device)
        return model, {
            "source_model_id": model_id,
            "input_representation": "224x224 log-magnitude spectrogram, ImageNet normalization",
        }
    raise ValueError(f"Unsupported model: {model_key}")


def run_model(args: argparse.Namespace, model_key: str, events: list[ParticleEvent], signals: np.ndarray, labels: np.ndarray) -> None:
    device = torch.device(args.device)
    if model_key == "moment_official":
        model_id = args.moment_model_id
    elif model_key == "patchtst_pretrained":
        model_id = args.patchtst_model_id
    elif model_key == "swin2d_pretrained":
        model_id = args.swin_model_id
    else:
        model_id = ""
    model_dir = args.output_dir / model_key
    encoder, model_metadata = load_encoder(model_key, model_id, args.cache_dir, device, model_dir, args)
    model_signals = select_model_signals(args, model_key, signals, events)
    feature_dim = infer_feature_dim(model_key, encoder, model_signals, device)
    model_metadata = {
        **model_metadata,
        "model_key": model_key,
        "display_name": MODEL_DISPLAY[model_key],
        "feature_dim": feature_dim,
        "event_length": int(args.event_length),
        "actual_input_length": int(model_signals.shape[1]),
        "public_pretrained": model_key != "conv1dgap_4class",
    }

    embeddings = encode_all_events(model_key, encoder, model_signals, args.batch_size, device)
    save_embedding_outputs(
        output_dir=model_dir / "zero_shot",
        events=events,
        embeddings=embeddings,
        labels=labels,
        seed=args.seed,
        title=f"{MODEL_DISPLAY[model_key]} zero-shot",
        extra_metadata={"stage": "zero_shot", **model_metadata},
    )

    split_idx = split_indices(events)
    if args.finetune_mode == "zero_shot":
        return
    if model_key == "conv1dgap_4class":
        raise ValueError("conv1dgap_4class is supported only with --finetune-mode zero_shot in this script")
    if model_key == "moment_official" and args.finetune_mode == "full":
        raise ValueError("moment_official supports zero_shot or linear_probe in this script; full fine-tune MOMENT-large separately.")
    if split_idx["train"].size == 0 or split_idx["val"].size == 0:
        raise ValueError("Fine-tuning requires non-empty train and val splits")

    train_loader = DataLoader(
        EventSignalDataset(model_signals, labels, split_idx["train"]),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        EventSignalDataset(model_signals, labels, split_idx["val"]),
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_indices = split_idx["test"] if split_idx["test"].size else split_idx["val"]
    test_loader = DataLoader(
        EventSignalDataset(model_signals, labels, test_indices),
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
    )

    classifier = EncoderClassifier(
        model_key=model_key,
        encoder=encoder,
        feature_dim=feature_dim,
        num_classes=int(labels.max()) + 1,
        device=device,
    ).to(device)

    linear_stats = train_classifier_stage(
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.linear_epochs,
        lr=args.linear_lr,
        train_encoder=False,
    )
    linear_test = evaluate_classifier(classifier, test_loader, device)
    linear_dir = model_dir / "linear_probe"
    linear_dir.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), linear_dir / "classifier.pt")
    with (linear_dir / "classifier_metrics.json").open("w") as f:
        json.dump({"stage": "linear_probe", "val": linear_stats, "test": linear_test, **model_metadata}, f, indent=2, sort_keys=True)

    if args.finetune_mode == "linear_probe":
        return

    full_stats = train_classifier_stage(
        classifier=classifier,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.full_epochs,
        lr=args.full_lr,
        train_encoder=True,
    )
    full_test = evaluate_classifier(classifier, test_loader, device)
    full_dir = model_dir / "full_finetune"
    full_dir.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), full_dir / "classifier.pt")
    embeddings_ft = encode_all_events(model_key, classifier.encoder, model_signals, args.batch_size, device)
    save_embedding_outputs(
        output_dir=full_dir,
        events=events,
        embeddings=embeddings_ft,
        labels=labels,
        seed=args.seed,
        title=f"{MODEL_DISPLAY[model_key]} full fine-tuned",
        extra_metadata={
            "stage": "full_finetune",
            "linear_probe_test": linear_test,
            "full_finetune_val": full_stats,
            "full_finetune_test": full_test,
            **model_metadata,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public-pretrained PatchTST and Swin embedding/fine-tuning pipelines.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "pretrained_backbones")
    parser.add_argument("--models", default="patchtst_pretrained,swin2d_pretrained")
    parser.add_argument("--moment-model-id", default=MOMENT_DEFAULT_ID)
    parser.add_argument("--patchtst-model-id", default=PATCHTST_DEFAULT_ID)
    parser.add_argument("--swin-model-id", default=SWIN_DEFAULT_ID)
    parser.add_argument("--conv1dgap-checkpoint", type=Path, default=CONV1DGAP_DEFAULT_CHECKPOINT)
    parser.add_argument("--conv1dgap-model-name", default="Conv1DGAP")
    parser.add_argument("--conv1dgap-input-mode", choices=["native_p0", "event_crop_512"], default="native_p0")
    parser.add_argument("--conv1dgap-input-length", type=int, default=4096)
    parser.add_argument("--conv1dgap-native-length", type=int, default=16384)
    parser.add_argument("--conv1dgap-bandpass-low-khz", type=float, default=5.0)
    parser.add_argument("--conv1dgap-bandpass-high-khz", type=float, default=100.0)
    parser.add_argument("--conv1dgap-sample-rate-mhz", type=float, default=2.0)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs" / "hf_cache")
    parser.add_argument("--input-length-raw", type=int, default=16384)
    parser.add_argument("--decimation-factor", type=int, default=8)
    parser.add_argument("--input-length-ssl", type=int, default=2048)
    parser.add_argument("--event-length", type=int, default=512)
    parser.add_argument("--normalization", default="window_zscore")
    parser.add_argument("--max-events-per-class", type=int, default=None)
    parser.add_argument("--include-class-names", default=None, help="Optional comma-separated class names to keep and remap in that order, e.g. 2um,4um,10um.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--finetune-mode", choices=["zero_shot", "linear_probe", "full"], default="full")
    parser.add_argument("--linear-epochs", type=int, default=20)
    parser.add_argument("--full-epochs", type=int, default=5)
    parser.add_argument("--linear-lr", type=float, default=1.0e-3)
    parser.add_argument("--full-lr", type=float, default=2.0e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    events, signals = collect_particle_events(
        manifest_csv=args.manifest,
        input_length_raw=args.input_length_raw,
        decimation_factor=args.decimation_factor,
        input_length_ssl=args.input_length_ssl,
        event_length=args.event_length,
        normalization=args.normalization,
        max_events_per_class=args.max_events_per_class,
        seed=args.seed,
    )
    events, signals = filter_and_remap_classes(events, signals, parse_csv_strings(args.include_class_names))
    labels = np.asarray([event.class_id for event in events], dtype=np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_event_metadata(args.output_dir / "events_metadata.csv", events)
    np.savez_compressed(args.output_dir / "event_crops.npz", signals=signals.astype(np.float32), labels=labels)
    with (args.output_dir / "run_config.json").open("w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2, sort_keys=True)

    requested_models = [item.strip() for item in args.models.split(",") if item.strip()]
    for model_key in requested_models:
        if model_key not in MODEL_DISPLAY:
            raise ValueError(f"Unsupported model {model_key}. Expected one of {sorted(MODEL_DISPLAY)}")
        run_model(args, model_key, events, signals, labels)

    zero_shot_dirs = {
        model_key: args.output_dir / model_key / "zero_shot"
        for model_key in COMPARISON_MODEL_ORDER
        if (args.output_dir / model_key / "zero_shot" / "embeddings.npz").is_file()
    }
    plot_pretrained_model_comparison(
        output_pdf=args.output_dir / "moment_patchtst_conv1dgap_native_params_pca_tsne_fig7_style.pdf",
        output_png=args.output_dir / "moment_patchtst_conv1dgap_native_params_pca_tsne_fig7_style.png",
        model_output_dirs=zero_shot_dirs,
    )
    print(f"Wrote pretrained backbone outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
