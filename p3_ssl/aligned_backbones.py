#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
from p3_ssl import backbone_benchmark as backbone
from p3_ssl.decimation import decimate_signal, normalize_signal
from p3_ssl.pretrained_backbones import MOMENT_DEFAULT_ID, PATCHTST_DEFAULT_ID, ParticleEvent

CLASS_NAMES = ("2um", "4um", "10um")
MODEL_KEYS = ("moment_official", "patchtst_pretrained", "conv1dgap_same_input_3class")
CONV_MODEL_KEY = "conv1dgap_same_input_3class"

backbone.MODEL_DISPLAY[CONV_MODEL_KEY] = "Conv1D-GAP-L supervised augmented same-input"
if CONV_MODEL_KEY not in backbone.COMPARISON_MODEL_ORDER:
    insert_at = backbone.COMPARISON_MODEL_ORDER.index("conv1dgap_4class") if "conv1dgap_4class" in backbone.COMPARISON_MODEL_ORDER else 2
    backbone.COMPARISON_MODEL_ORDER.insert(insert_at, CONV_MODEL_KEY)
backbone.MODEL_DISPLAY["moment_official"] = "MOMENT frozen pretrained"
backbone.MODEL_DISPLAY["patchtst_pretrained"] = "PatchTST frozen pretrained"


class SignalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, indices: np.ndarray | None = None) -> None:
        self.signals = signals.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.indices = np.arange(labels.shape[0], dtype=np.int64) if indices is None else indices.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[index])
        return torch.from_numpy(self.signals[sample_idx]).float(), torch.tensor(int(self.labels[sample_idx]), dtype=torch.long)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def crop_around_center(raw: np.ndarray, crop_length: int, center_index: int) -> np.ndarray:
    x = raw.astype(np.float32, copy=False).reshape(-1)
    start = int(center_index) - crop_length // 2
    end = start + crop_length
    crop = np.zeros(crop_length, dtype=np.float32)
    src_start = max(0, start)
    src_end = min(x.shape[0], end)
    if src_end > src_start:
        dst_start = src_start - start
        crop[dst_start : dst_start + (src_end - src_start)] = x[src_start:src_end]
    return crop


def build_aligned_signal(
    raw: np.ndarray,
    raw_crop_length: int = 4096,
    output_length: int = 4096,
    center_offset: int = 0,
    amplitude_scale: float = 1.0,
    noise_snr_db: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if raw_crop_length % output_length != 0:
        raise ValueError("raw_crop_length must be divisible by output_length")
    center = int(raw.shape[-1] // 2) + int(center_offset)
    centered = crop_around_center(raw, raw_crop_length, center)
    decimated = decimate_signal(centered, raw_crop_length // output_length, method="mean")
    decimated = decimated.astype(np.float32, copy=False) * float(amplitude_scale)
    if noise_snr_db is not None:
        if rng is None:
            rng = np.random.default_rng()
        power = float(np.mean(np.square(decimated)))
        if power > 0.0:
            noise_power = power / (10.0 ** (float(noise_snr_db) / 10.0))
            decimated = decimated + rng.normal(0.0, np.sqrt(noise_power), size=decimated.shape).astype(np.float32)
    return normalize_signal(decimated, mode="window_zscore").astype(np.float32, copy=False)


def build_aligned_512_signal(
    raw: np.ndarray,
    raw_crop_length: int = 4096,
    output_length: int = 512,
    center_offset: int = 0,
    amplitude_scale: float = 1.0,
    noise_snr_db: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    return build_aligned_signal(
        raw,
        raw_crop_length=raw_crop_length,
        output_length=output_length,
        center_offset=center_offset,
        amplitude_scale=amplitude_scale,
        noise_snr_db=noise_snr_db,
        rng=rng,
    )


def load_particles2snr_f_events(data_dir: Path, raw_crop_length: int, output_length: int) -> tuple[list[ParticleEvent], np.ndarray, np.ndarray]:
    events: list[ParticleEvent] = []
    signals: list[np.ndarray] = []
    labels: list[int] = []
    for split in ("train", "val", "test"):
        for class_id, class_name in enumerate(CLASS_NAMES):
            class_dir = data_dir / split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for path in sorted(class_dir.glob("*.npy")):
                raw = np.load(path).astype(np.float32, copy=False)
                signals.append(build_aligned_signal(raw, raw_crop_length=raw_crop_length, output_length=output_length))
                labels.append(class_id)
                events.append(
                    ParticleEvent(
                        event_id=f"{split}/{class_name}/{path.stem}",
                        sample_id=path.stem,
                        split=split,
                        signal_path=str(path),
                        label_path="",
                        class_id=class_id,
                        class_name=class_name,
                        center_norm=0.5,
                        width_norm=0.0,
                        center_index=int(raw.shape[-1] // 2),
                        crop_start=int(raw.shape[-1] // 2 - raw_crop_length // 2),
                        crop_end=int(raw.shape[-1] // 2 + raw_crop_length // 2),
                    )
                )
    return events, np.stack(signals).astype(np.float32), np.asarray(labels, dtype=np.int64)


def split_indices(events: list[ParticleEvent]) -> dict[str, np.ndarray]:
    result: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx, event in enumerate(events):
        result[event.split].append(idx)
    return {split: np.asarray(indices, dtype=np.int64) for split, indices in result.items()}


def balanced_visual_indices(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(set(int(v) for v in labels.tolist())):
        idx = np.flatnonzero(labels == class_id)
        if max_per_class > 0 and idx.size > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.extend(int(v) for v in idx.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def balanced_event_indices(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    if max_per_class <= 0:
        return np.arange(labels.shape[0], dtype=np.int64)
    return balanced_visual_indices(labels, max_per_class=max_per_class, seed=seed)


def validate_no_test_leakage(source_splits: np.ndarray) -> None:
    leaks = np.flatnonzero(source_splits.astype(str) == "test")
    if leaks.size:
        raise ValueError(f"Conv1D-GAP training data includes {int(leaks.size)} test-derived views")


def materialize_conv_train_views(
    events: list[ParticleEvent],
    labels: np.ndarray,
    train_indices: np.ndarray,
    views_per_event: int,
    raw_crop_length: int,
    output_length: int,
    jitter_frac: float,
    aug_snr_db: float | None,
    aug_scale_min: float,
    aug_scale_max: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    max_offset = int(round(float(raw_crop_length) * float(jitter_frac)))
    signals: list[np.ndarray] = []
    view_labels: list[int] = []
    source_event_index: list[int] = []
    view_id: list[int] = []
    source_split: list[str] = []
    source_path: list[str] = []
    for event_idx in train_indices.tolist():
        event = events[int(event_idx)]
        raw = np.load(event.signal_path).astype(np.float32, copy=False)
        for view in range(int(views_per_event)):
            offset = int(rng.integers(-max_offset, max_offset + 1)) if max_offset > 0 else 0
            scale = float(rng.uniform(aug_scale_min, aug_scale_max))
            signals.append(
                build_aligned_signal(
                    raw,
                    raw_crop_length=raw_crop_length,
                    output_length=output_length,
                    center_offset=offset,
                    amplitude_scale=scale,
                    noise_snr_db=aug_snr_db,
                    rng=rng,
                )
            )
            view_labels.append(int(labels[int(event_idx)]))
            source_event_index.append(int(event_idx))
            view_id.append(int(view))
            source_split.append(event.split)
            source_path.append(event.signal_path)
    source_split_arr = np.asarray(source_split)
    validate_no_test_leakage(source_split_arr)
    return {
        "signals": np.stack(signals).astype(np.float32),
        "labels": np.asarray(view_labels, dtype=np.int64),
        "source_event_index": np.asarray(source_event_index, dtype=np.int64),
        "view_id": np.asarray(view_id, dtype=np.int64),
        "source_split": source_split_arr,
        "source_path": np.asarray(source_path),
    }


def materialize_eval_split(signals: np.ndarray, labels: np.ndarray, split_indices_arr: np.ndarray, events: list[ParticleEvent]) -> dict[str, np.ndarray]:
    return {
        "signals": signals[split_indices_arr].astype(np.float32, copy=False),
        "labels": labels[split_indices_arr].astype(np.int64, copy=False),
        "source_event_index": split_indices_arr.astype(np.int64, copy=False),
        "source_split": np.asarray([events[int(i)].split for i in split_indices_arr]),
        "source_path": np.asarray([events[int(i)].signal_path for i in split_indices_arr]),
    }


def write_conv_dataset_npz(output_dir: Path, name: str, payload: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"conv_dataset_{name}.npz", **payload)


def train_conv1dgap_same_input(
    train_payload: dict[str, np.ndarray],
    val_payload: dict[str, np.ndarray],
    device: torch.device,
    output_dir: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
) -> nn.Module:
    from p0.models import create_model

    train_signals = train_payload["signals"]
    train_labels = train_payload["labels"]
    val_signals = val_payload["signals"]
    val_labels = val_payload["labels"]
    model = create_model(model_name, input_length=train_signals.shape[1], num_classes=len(CLASS_NAMES)).to(device)
    train_loader = DataLoader(SignalDataset(train_signals, train_labels), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SignalDataset(val_signals, val_labels), batch_size=batch_size, shuffle=False, num_workers=0)
    counts = np.bincount(train_labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device).unsqueeze(1)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate_torch_classifier(model, val_loader, device)
        row = {"epoch": float(epoch), "loss": float(np.mean(losses)), **val_metrics}
        history.append(row)
        if val_metrics["macro_f1"] > best_macro_f1 + float(min_delta):
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "class_names": CLASS_NAMES,
            "input_length": int(train_signals.shape[1]),
            "model_name": model_name,
            "best_epoch": int(best_epoch),
            "best_val_macro_f1": float(best_macro_f1),
        },
        output_dir / "best_model.pt",
    )
    with (output_dir / "training_history.json").open("w") as f:
        json.dump(
            {
                "history": history,
                "best_val_macro_f1": best_macro_f1,
                "best_epoch": best_epoch,
                "class_weights": weights.tolist(),
                "model_name": model_name,
                "train_views": int(train_signals.shape[0]),
            },
            f,
            indent=2,
            sort_keys=True,
        )
    model.eval()
    return model


def evaluate_torch_classifier(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device).unsqueeze(1))
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            y_pred.extend(int(v) for v in pred.tolist())
            y_true.extend(int(v) for v in y.numpy().tolist())
    return classification_metrics(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64), compact=True)


def evaluate_torch_classifier_payload(model: nn.Module, payload: dict[str, np.ndarray], batch_size: int, device: torch.device) -> dict[str, Any]:
    loader = DataLoader(SignalDataset(payload["signals"], payload["labels"]), batch_size=batch_size, shuffle=False, num_workers=0)
    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device).unsqueeze(1))
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            y_pred.extend(int(v) for v in pred.tolist())
            y_true.extend(int(v) for v in y.numpy().tolist())
    return {"n": int(len(y_true)), **classification_metrics(np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64))}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, compact: bool = False) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if compact:
        return metrics
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).astype(int).tolist()
    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    return metrics


def encode_conv_features_all(model: nn.Module, signals: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, signals.shape[0], batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size]).float()
            features = backbone.encode_conv1dgap_features(model, batch, device=device)
            chunks.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def linear_probe_on_embeddings(embeddings: np.ndarray, labels: np.ndarray, splits: dict[str, np.ndarray]) -> dict[str, Any]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    clf.fit(embeddings[splits["train"]], labels[splits["train"]])
    result: dict[str, Any] = {"classifier": "StandardScaler + LogisticRegression(class_weight=balanced)"}
    for split, idx in splits.items():
        pred = clf.predict(embeddings[idx])
        result[split] = {"n": int(idx.size), **classification_metrics(labels[idx], pred)}
    return result


def write_rows(path: Path, events: list[ParticleEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(events[0]).keys()))
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    args.event_length = int(args.input_length)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events, signals, labels = load_particles2snr_f_events(args.data_dir, args.raw_crop_length, args.input_length)
    keep_idx = balanced_event_indices(labels, args.max_events_per_class, args.seed)
    events = [events[int(i)] for i in keep_idx]
    signals = signals[keep_idx]
    labels = labels[keep_idx]
    splits = split_indices(events)
    visual_idx = balanced_visual_indices(labels, args.max_plot_per_class, args.seed)
    visual_events = [events[int(i)] for i in visual_idx]
    visual_labels = labels[visual_idx]
    write_rows(args.output_dir / "events_metadata.csv", events)
    write_rows(args.output_dir / "visual_events_metadata.csv", visual_events)
    np.savez_compressed(args.output_dir / "aligned_inputs.npz", signals=signals, labels=labels, split=np.asarray([e.split for e in events]))

    conv_dir = args.output_dir / CONV_MODEL_KEY
    conv_train = materialize_conv_train_views(
        events=events,
        labels=labels,
        train_indices=splits["train"],
        views_per_event=args.views_per_train_event,
        raw_crop_length=args.raw_crop_length,
        output_length=args.input_length,
        jitter_frac=args.jitter_frac,
        aug_snr_db=None if args.aug_snr_db <= 0 else args.aug_snr_db,
        aug_scale_min=args.aug_scale_min,
        aug_scale_max=args.aug_scale_max,
        seed=args.seed,
    )
    conv_val = materialize_eval_split(signals, labels, splits["val"], events)
    conv_test = materialize_eval_split(signals, labels, splits["test"], events)
    if args.materialize_conv_dataset:
        write_conv_dataset_npz(conv_dir, "train", conv_train)
        write_conv_dataset_npz(conv_dir, "val", conv_val)
        write_conv_dataset_npz(conv_dir, "test", conv_test)

    with (args.output_dir / "run_config.json").open("w") as f:
        json.dump(
            {
                "dataset": str(args.data_dir),
                "classes": list(CLASS_NAMES),
                "n_events": len(events),
                "visual_n_events": int(visual_idx.size),
                "input_representation_all_models": f"center crop raw {int(args.raw_crop_length)} -> {int(args.input_length)} -> window_zscore",
                "conv1dgap_role": "supervised same-input CNN control, not a public pretrained zero-shot model",
                "conv_model_name": args.conv_model_name,
                "views_per_train_event": args.views_per_train_event,
                "conv_train_views": int(conv_train["signals"].shape[0]),
                "augmentation": {
                    "jitter_frac": args.jitter_frac,
                    "aug_snr_db": args.aug_snr_db,
                    "aug_scale_min": args.aug_scale_min,
                    "aug_scale_max": args.aug_scale_max,
                },
                "seed": args.seed,
                "device": str(device),
            },
            f,
            indent=2,
            sort_keys=True,
        )

    conv_model = train_conv1dgap_same_input(
        train_payload=conv_train,
        val_payload=conv_val,
        device=device,
        output_dir=conv_dir,
        model_name=args.conv_model_name,
        epochs=args.conv_epochs,
        batch_size=args.conv_batch_size,
        lr=args.conv_lr,
        weight_decay=args.conv_weight_decay,
        patience=args.conv_patience,
        min_delta=args.conv_min_delta,
    )
    direct_classifier_metrics = {
        "train_deterministic": evaluate_torch_classifier_payload(conv_model, materialize_eval_split(signals, labels, splits["train"], events), args.conv_batch_size, device),
        "val": evaluate_torch_classifier_payload(conv_model, conv_val, args.conv_batch_size, device),
        "test": evaluate_torch_classifier_payload(conv_model, conv_test, args.conv_batch_size, device),
    }
    with (conv_dir / "direct_classifier_metrics.json").open("w") as f:
        json.dump(direct_classifier_metrics, f, indent=2, sort_keys=True)

    model_dirs: dict[str, Path] = {}
    all_metrics: dict[str, Any] = {}
    requested_models = [item.strip() for item in args.models.split(",") if item.strip()]
    for model_key in requested_models:
        if model_key not in MODEL_KEYS:
            raise ValueError(f"Unsupported model {model_key}. Expected one of {sorted(MODEL_KEYS)}")
        model_dir = args.output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        if model_key == "moment_official":
            encoder, metadata = backbone.load_encoder(model_key, args.moment_model_id, args.cache_dir, device, model_dir, args)
            embeddings = backbone.encode_all_events(model_key, encoder, signals, args.batch_size, device)
        elif model_key == "patchtst_pretrained":
            encoder, metadata = backbone.load_encoder(model_key, args.patchtst_model_id, args.cache_dir, device, model_dir, args)
            embeddings = backbone.encode_all_events(model_key, encoder, signals, args.batch_size, device)
        else:
            metadata = {
                "source_model_id": str(conv_dir / "best_model.pt"),
                "model_name": args.conv_model_name,
                "input_representation": f"same {int(signals.shape[1])}-sample aligned tensor as MOMENT/PatchTST",
                "input_length": int(signals.shape[1]),
                "supervised_same_input_checkpoint": True,
                "views_per_train_event": int(args.views_per_train_event),
                "conv_train_views": int(conv_train["signals"].shape[0]),
                "public_pretrained": False,
            }
            embeddings = encode_conv_features_all(conv_model, signals, args.batch_size, device)

        np.savez_compressed(
            model_dir / "all_embeddings.npz",
            embeddings=embeddings.astype(np.float32),
            labels=labels.astype(np.int64),
            split=np.asarray([event.split for event in events]),
            event_id=np.asarray([event.event_id for event in events]),
        )
        feature_dim = int(embeddings.shape[1])
        probe_metrics = linear_probe_on_embeddings(embeddings, labels, splits)
        all_metrics[model_key] = {
            "feature_dim": feature_dim,
            "linear_probe": probe_metrics,
            "metadata": metadata,
        }
        if model_key == CONV_MODEL_KEY:
            all_metrics[model_key]["direct_classifier"] = direct_classifier_metrics
        with (model_dir / "linear_probe_metrics.json").open("w") as f:
            json.dump(all_metrics[model_key], f, indent=2, sort_keys=True)

        visual_embeddings = embeddings[visual_idx]
        visual_metadata = {
            **metadata,
            "model_key": model_key,
            "display_name": backbone.MODEL_DISPLAY[model_key],
            "feature_dim": feature_dim,
            "n_classes": len(CLASS_NAMES),
            "class_labels": {str(i): name for i, name in enumerate(CLASS_NAMES)},
            "actual_input_length": int(signals.shape[1]),
            "stage": "aligned_same_input_features_3class_particles2snr_f",
            "linear_probe_test": probe_metrics["test"],
        }
        if model_key == CONV_MODEL_KEY:
            visual_metadata["direct_classifier_test"] = direct_classifier_metrics["test"]
        backbone.save_embedding_outputs(
            output_dir=model_dir / "zero_shot",
            events=visual_events,
            embeddings=visual_embeddings,
            labels=visual_labels,
            seed=args.seed,
            title=f"{backbone.MODEL_DISPLAY[model_key]} - same input 3-class",
            extra_metadata=visual_metadata,
        )
        model_dirs[model_key] = model_dir / "zero_shot"

    with (args.output_dir / "classification_summary.json").open("w") as f:
        json.dump(all_metrics, f, indent=2, sort_keys=True)

    backbone.plot_pretrained_model_comparison(
        output_pdf=args.output_dir / "particles2snr_f_3class_same_input_augmented_conv1dgap_pca_tsne_fig7_style.pdf",
        output_png=args.output_dir / "particles2snr_f_3class_same_input_augmented_conv1dgap_pca_tsne_fig7_style.png",
        model_output_dirs=model_dirs,
    )
    print(f"Wrote aligned 3-class backbone comparison to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aligned same-input MOMENT/PatchTST/Conv1D-GAP comparison on Particles2SNR_F 3-class events.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "datasets" / "processed" / "particles2snr-f-c1-events" / "v1",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones" / "particles2snr_f_3class_same_input_conv1dgap_augmented_train")
    parser.add_argument("--cache-dir", type=Path, default=ROOT.parent / ".cache" / "huggingface")
    parser.add_argument("--moment-model-id", default=MOMENT_DEFAULT_ID)
    parser.add_argument("--patchtst-model-id", default=PATCHTST_DEFAULT_ID)
    parser.add_argument("--input-length", type=int, default=4096)
    parser.add_argument("--raw-crop-length", type=int, default=4096)
    parser.add_argument("--max-events-per-class", type=int, default=0)
    parser.add_argument("--max-plot-per-class", type=int, default=500)
    parser.add_argument("--models", default="moment_official,patchtst_pretrained,conv1dgap_same_input_3class")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--conv-model-name", default="Conv1DGAP-L")
    parser.add_argument("--views-per-train-event", type=int, default=12)
    parser.add_argument("--jitter-frac", type=float, default=0.25)
    parser.add_argument("--aug-snr-db", type=float, default=25.0)
    parser.add_argument("--aug-scale-min", type=float, default=0.8)
    parser.add_argument("--aug-scale-max", type=float, default=1.25)
    parser.add_argument("--no-materialize-conv-dataset", dest="materialize_conv_dataset", action="store_false")
    parser.set_defaults(materialize_conv_dataset=True)
    parser.add_argument("--conv-batch-size", type=int, default=64)
    parser.add_argument("--conv-epochs", type=int, default=50)
    parser.add_argument("--conv-patience", type=int, default=12)
    parser.add_argument("--conv-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--conv-lr", type=float, default=1.0e-3)
    parser.add_argument("--conv-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
