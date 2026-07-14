from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy import signal as scipy_signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .pretrained_backbones import (
    MOMENT_DEFAULT_ID,
    PATCHTST_DEFAULT_ID,
    encode_moment_official_batch,
    encode_patchtst_batch,
    load_moment_official_model,
    load_patchtst_1ch_model,
)
from .study_model import YeastStudyModel, YeastStudyModelConfig


BASELINE_METHODS = {
    "rms",
    "raw",
    "handcrafted",
    "random",
    "moment",
    "patchtst",
    "conv1d",
}


@dataclass(frozen=True)
class BaselineData:
    signals: np.ndarray
    rows: list[dict[str, str]]
    labels: np.ndarray
    class_names: list[str]
    train_indices: np.ndarray
    validation_indices: np.ndarray


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stratified_limit(
    indices: np.ndarray,
    labels: np.ndarray,
    max_per_class: int | None,
    seed: int,
) -> np.ndarray:
    if max_per_class is None:
        return indices
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(set(labels[indices].tolist())):
        candidates = indices[labels[indices] == class_id]
        if candidates.size > max_per_class:
            candidates = rng.choice(candidates, size=max_per_class, replace=False)
        selected.extend(int(value) for value in candidates)
    return np.asarray(sorted(selected), dtype=np.int64)


def load_baseline_data(
    root: Path,
    *,
    max_per_class: int | None = None,
    seed: int = 42,
) -> BaselineData:
    rows = _read_rows(root / "events.csv")
    signals = np.load(root / "signals.npy", mmap_mode="r")
    if len(rows) != len(signals):
        raise ValueError("events.csv and signals.npy have different row counts")
    class_names = sorted({row["source_group"] for row in rows})
    class_to_id = {name: index for index, name in enumerate(class_names)}
    labels = np.asarray([class_to_id[row["source_group"]] for row in rows], dtype=np.int64)
    split = np.asarray([row["development_split"] for row in rows])
    train = np.flatnonzero(split == "development_train")
    validation = np.flatnonzero(split == "development_validation")
    if not train.size or not validation.size:
        raise ValueError("Development train and validation must both be non-empty")
    train = _stratified_limit(train, labels, max_per_class, seed)
    validation = _stratified_limit(validation, labels, max_per_class, seed + 1)
    expected = set(range(len(class_names)))
    if set(labels[train]) != expected or set(labels[validation]) != expected:
        raise ValueError("Every source-group proxy must occur in train and validation")
    return BaselineData(signals, rows, labels, class_names, train, validation)


def sample_record_groups(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    candidate_indices: np.ndarray,
    fraction: float,
    seed: int,
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    selected_records: set[str] = set()
    for class_id in sorted(set(labels[candidate_indices].tolist())):
        class_indices = candidate_indices[labels[candidate_indices] == class_id]
        records = np.asarray(sorted({rows[int(index)]["record_id"] for index in class_indices}))
        n_selected = max(1, int(np.ceil(fraction * records.size)))
        if n_selected < records.size:
            records = rng.choice(records, size=n_selected, replace=False)
        selected_records.update(str(value) for value in records)
    selected = [
        int(index)
        for index in candidate_indices
        if rows[int(index)]["record_id"] in selected_records
    ]
    return np.asarray(selected, dtype=np.int64)


def rms_features(signals: np.ndarray) -> np.ndarray:
    values = np.sqrt(np.mean(np.square(np.asarray(signals, dtype=np.float64)), axis=1))
    return values.astype(np.float32).reshape(-1, 1)


def handcrafted_features(signals: np.ndarray, sampling_frequency_hz: float = 1_000_000.0) -> np.ndarray:
    x = np.asarray(signals, dtype=np.float64)
    abs_x = np.abs(x)
    quantiles = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], axis=1).T
    rms = np.sqrt(np.mean(np.square(x), axis=1))
    time_features = np.column_stack(
        [
            np.mean(x, axis=1),
            np.std(x, axis=1),
            rms,
            np.mean(abs_x, axis=1),
            np.max(abs_x, axis=1),
            np.ptp(x, axis=1),
            np.max(abs_x, axis=1) / np.maximum(rms, 1.0e-12),
            np.mean(np.diff(np.signbit(x), axis=1), axis=1),
            quantiles,
        ]
    )
    spectrum = np.abs(np.fft.rfft(x, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(x.shape[1], d=1.0 / sampling_frequency_hz)
    total = np.maximum(spectrum.sum(axis=1), 1.0e-12)
    centroid = (spectrum * frequencies).sum(axis=1) / total
    bandwidth = np.sqrt(
        (spectrum * np.square(frequencies[None, :] - centroid[:, None])).sum(axis=1) / total
    )
    peak_frequency = frequencies[np.argmax(spectrum, axis=1)]
    edges = np.asarray([0, 5, 10, 20, 40, 60, 80, 100, 200, 500], dtype=float) * 1000.0
    band_features = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        band_features.append(spectrum[:, mask].sum(axis=1) / total)
    envelope = np.abs(scipy_signal.hilbert(x, axis=1))
    frequency_features = np.column_stack(
        [centroid, bandwidth, peak_frequency, *band_features]
    )
    envelope_features = np.column_stack(
        [np.mean(envelope, axis=1), np.std(envelope, axis=1), np.max(envelope, axis=1)]
    )
    result = np.column_stack([time_features, frequency_features, envelope_features])
    if not np.isfinite(result).all():
        raise ValueError("Handcrafted features contain non-finite values")
    return result.astype(np.float32)


def _iter_batches(signals: np.ndarray, batch_size: int) -> Iterable[torch.Tensor]:
    for start in range(0, len(signals), batch_size):
        yield torch.from_numpy(np.asarray(signals[start : start + batch_size], dtype=np.float32))


@torch.no_grad()
def random_encoder_features(
    signals: np.ndarray,
    *,
    config: YeastStudyModelConfig,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    model = YeastStudyModel(config).to(device).eval()
    outputs = []
    for batch in _iter_batches(signals, batch_size):
        outputs.append(model(batch.unsqueeze(1).to(device))["embedding"].cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


@torch.no_grad()
def public_encoder_features(
    method: str,
    signals: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    cache_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "moment":
        model = load_moment_official_model(
            MOMENT_DEFAULT_ID, cache_dir=cache_dir, device=device, seq_len=signals.shape[1]
        )
        outputs = [
            encode_moment_official_batch(model, batch, device).cpu().numpy()
            for batch in _iter_batches(signals, batch_size)
        ]
        metadata = {"model_id": MOMENT_DEFAULT_ID, "input_policy": "full-4096"}
    elif method == "patchtst":
        model, report = load_patchtst_1ch_model(
            PATCHTST_DEFAULT_ID, cache_dir=cache_dir, device=device
        )
        context_length = int(model.config.context_length)
        if signals.shape[1] % context_length:
            raise ValueError("PatchTST context length must divide the frozen input length")
        outputs = []
        for batch in _iter_batches(signals, batch_size):
            batch = batch.to(device)
            chunks = batch.reshape(batch.shape[0], -1, context_length)
            encoded = [encode_patchtst_batch(model, chunks[:, index]) for index in range(chunks.shape[1])]
            outputs.append(torch.stack(encoded, dim=1).mean(dim=1).cpu().numpy())
        metadata = {
            "model_id": PATCHTST_DEFAULT_ID,
            "input_policy": f"nonoverlapping-{context_length}-sample-chunks-mean-pooled",
            "transfer": {
                "loaded_keys": report.loaded_keys,
                "skipped_keys": report.skipped_keys,
                "missing_keys": report.missing_keys,
            },
        }
    else:
        raise ValueError(f"Unsupported public encoder: {method}")
    return np.concatenate(outputs).astype(np.float32), metadata


def prediction_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    class_ids = np.arange(len(class_names))
    recalls = recall_score(labels, predictions, labels=class_ids, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "per_class_recall": {
            name: float(recalls[index]) for index, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_ids).tolist(),
        "n_evaluation": int(labels.size),
    }


def linear_probe(
    features: np.ndarray,
    data: BaselineData,
    *,
    fraction: float,
    seed: int,
) -> dict[str, Any]:
    train = sample_record_groups(data.rows, data.labels, data.train_indices, fraction, seed)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=500,
            random_state=seed,
        ),
    )
    model.fit(features[train], data.labels[train])
    predictions = model.predict(features[data.validation_indices])
    return {
        **prediction_metrics(data.labels[data.validation_indices], predictions, data.class_names),
        "n_probe_events": int(train.size),
        "n_probe_records": len({data.rows[int(index)]["record_id"] for index in train}),
    }


class Conv1DGapClassifier(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=2, padding=4),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(signals).squeeze(-1))


class _IndexedSignals(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: BaselineData, indices: np.ndarray) -> None:
        self.data = data
        self.indices = indices

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.indices[index])
        signal = np.array(self.data.signals[row], dtype=np.float32, copy=True)
        return torch.from_numpy(signal).unsqueeze(0), torch.tensor(self.data.labels[row])


def supervised_conv1d(
    data: BaselineData,
    *,
    fraction: float,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train = sample_record_groups(data.rows, data.labels, data.train_indices, fraction, seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _IndexedSignals(data, train), batch_size=batch_size, shuffle=True, generator=generator
    )
    validation_loader = DataLoader(
        _IndexedSignals(data, data.validation_indices), batch_size=batch_size, shuffle=False
    )
    model = Conv1DGapClassifier(len(data.class_names)).to(device)
    counts = np.bincount(data.labels[train], minlength=len(data.class_names))
    weights = counts.sum() / np.maximum(counts, 1) / len(counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    for _ in range(epochs):
        model.train()
        for signals, labels in train_loader:
            loss = criterion(model(signals.to(device)), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for signals, target in validation_loader:
            predictions.extend(model(signals.to(device)).argmax(dim=1).cpu().tolist())
            labels.extend(target.tolist())
    return {
        **prediction_metrics(np.asarray(labels), np.asarray(predictions), data.class_names),
        "n_probe_events": int(train.size),
        "n_probe_records": len({data.rows[int(index)]["record_id"] for index in train}),
        "epochs": epochs,
    }
