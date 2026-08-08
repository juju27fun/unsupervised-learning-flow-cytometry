from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from .bead_ssl import train_bead_ssl
from .config import load_config
from .decimation import normalize_signal


SYNTHETIC_DATASET_ID = (
    "particles2snr-fbase-z8-cholesky-physicalcorr-effective-snr-"
    "synthetic-events@v5"
)
REAL_EVENT_DATASET_ID = (
    "particles2snr-fbase-dual-clean-z8-events-3class-plus-unclear-"
    "development@v2"
)
REAL_SIGNAL_DATASET_ID = "particles2snr-f-dual-clean-c1-yolo-4class@v2"
PHYSICAL_CLASSES = ("2um", "4um", "10um")


def fixed_class_proportional_monitor_indices(
    dataset: Z8AsymmetricSyntheticDataset,
    *,
    max_samples: int,
    seed: int,
    split_tag: str,
) -> tuple[list[int], dict[str, Any]]:
    """Select an immutable class-proportional monitor from dataset metadata."""
    if max_samples < 1:
        raise ValueError("monitor size must be positive")
    target = min(int(max_samples), len(dataset))
    grouped: dict[str, list[tuple[int, str]]] = {name: [] for name in PHYSICAL_CLASSES}
    for dataset_index, (_, row) in enumerate(dataset.rows):
        class_name = str(row["class_name"])
        if class_name not in grouped:
            raise ValueError(f"unexpected monitor class: {class_name}")
        grouped[class_name].append((dataset_index, str(row["sample_id"])))
    exact = {
        name: target * len(rows) / max(len(dataset), 1)
        for name, rows in grouped.items()
    }
    quotas = {name: int(math.floor(value)) for name, value in exact.items()}
    remaining = target - sum(quotas.values())
    remainder_order = sorted(
        PHYSICAL_CLASSES,
        key=lambda name: (-(exact[name] - quotas[name]), name),
    )
    for name in remainder_order[:remaining]:
        quotas[name] += 1

    selected: list[tuple[int, str]] = []
    for class_name in PHYSICAL_CLASSES:
        ranked = sorted(
            grouped[class_name],
            key=lambda item: hashlib.sha256(
                f"{seed}|{split_tag}|{item[1]}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ranked[: quotas[class_name]])
    if len(selected) != target:
        raise ValueError("class-proportional monitor selection is incomplete")
    indices = sorted(index for index, _ in selected)
    selected_ids = sorted(sample_id for _, sample_id in selected)
    ids_sha256 = hashlib.sha256("\n".join(selected_ids).encode("utf-8")).hexdigest()
    return indices, {
        "protocol": "class-proportional-sample-id-sha256-v1",
        "split": split_tag,
        "seed": int(seed),
        "count": len(indices),
        "class_counts": quotas,
        "selected_ids_sha256": ids_sha256,
    }


def load_bead_ssl_v2_config(path: Path) -> dict[str, Any]:
    config = load_config(path)
    study = config["study"]
    if study["protocol"] != "bead-ssl-comparison-v2":
        raise ValueError("bead SSL v2 requires protocol bead-ssl-comparison-v2")
    expected = {
        "simulation_dataset": SYNTHETIC_DATASET_ID,
        "real_event_dataset": REAL_EVENT_DATASET_ID,
        "real_signal_dataset": REAL_SIGNAL_DATASET_ID,
    }
    for key, value in expected.items():
        if study.get(key) != value:
            raise ValueError(f"bead SSL v2 requires {key}={value}")
    if study["training_stage"] != "synthetic_only":
        raise ValueError("bead SSL v2 training must remain synthetic-only")
    if "test" not in study["forbidden_splits"]:
        raise ValueError("sealed test must remain forbidden")
    if config["model"]["mask_encoding"] != "sample_visibility_v1":
        raise ValueError("bead SSL v2 requires sample_visibility_v1")
    if config["training"].get("checkpoint_selection") != "fixed_final":
        raise ValueError("bead SSL v2 requires the frozen final checkpoint")
    return config


class Z8AsymmetricSyntheticDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: Path,
        *,
        split: str,
        normalization: str,
        sampling_frequency_hz: float = 1_000_000.0,
        max_samples: int | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"unsupported v5 split: {split}")
        self.signals = np.load(
            root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False
        )
        with (root / "events.csv").open(newline="", encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
        if len(all_rows) != len(self.signals):
            raise ValueError("v5 event rows and raw signals are not aligned")
        self.rows = [
            (index, row)
            for index, row in enumerate(all_rows)
            if row["noise_source_split"] == split
        ]
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"no v5 rows for split={split}")
        self.normalization = normalization
        self.sampling_frequency_hz = float(sampling_frequency_hz)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        signal_index, row = self.rows[index]
        signal = np.asarray(self.signals[signal_index], dtype=np.float32).copy()
        signal = normalize_signal(signal, mode=self.normalization)
        center = float(row["t0_fraction"]) * (signal.size - 1)
        tau_samples = (
            float(row["tau_ms"]) * 1.0e-3 * self.sampling_frequency_hz
        )
        asymmetry = float(row["waveform_asymmetry"])
        left = 3.0 * tau_samples * np.exp(-asymmetry)
        right = 3.0 * tau_samples * np.exp(asymmetry)
        event_start = max(0, int(np.floor(center - left)))
        event_end = min(signal.size, int(np.ceil(center + right)) + 1)
        if event_end <= event_start:
            raise ValueError(f"empty v5 event support: {row['sample_id']}")
        event_mask = np.zeros(signal.size, dtype=bool)
        event_mask[event_start:event_end] = True
        return {
            "signal": torch.from_numpy(signal).unsqueeze(0),
            "event_mask": torch.from_numpy(event_mask),
            "sample_index": signal_index,
            "sample_id": row["sample_id"],
            "class_name": row["class_name"],
        }


class Z8RealValidationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        event_root: Path,
        signal_root: Path,
        *,
        split: str,
        normalization: str,
        max_per_class: int | None = None,
        crop_length: int = 4096,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"unsupported Z8 split: {split}")
        with (event_root / "events.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["split"] == split and row["class_name"] in PHYSICAL_CLASSES
            ]
        self.rows: list[dict[str, str]] = []
        for class_name in PHYSICAL_CLASSES:
            selected = [row for row in rows if row["class_name"] == class_name]
            if max_per_class is not None:
                selected = selected[:max_per_class]
            self.rows.extend(selected)
        if not self.rows:
            raise ValueError(f"no physical Z8 rows for split={split}")
        self.signal_root = signal_root
        self.normalization = normalization
        self.crop_length = int(crop_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source = np.load(
            self.signal_root / row["source_signal_relative_path"],
            allow_pickle=False,
        )
        if source.ndim != 1 or source.size < self.crop_length:
            raise ValueError(f"invalid source signal: {row['source_filename']}")
        center = float(row["center_sample"])
        crop_start = int(round(center)) - self.crop_length // 2
        crop_start = min(max(crop_start, 0), source.size - self.crop_length)
        crop_end = crop_start + self.crop_length
        signal = np.asarray(source[crop_start:crop_end], dtype=np.float32).copy()
        signal = normalize_signal(signal, mode=self.normalization)
        event_start = max(0, int(np.floor(float(row["start_sample"]))) - crop_start)
        event_end = min(
            self.crop_length,
            int(np.ceil(float(row["end_sample"]))) - crop_start,
        )
        if event_end <= event_start:
            raise ValueError(f"event outside crop: {row['event_id']}")
        event_mask = np.zeros(self.crop_length, dtype=bool)
        event_mask[event_start:event_end] = True
        return {
            "signal": torch.from_numpy(signal).unsqueeze(0),
            "event_mask": torch.from_numpy(event_mask),
            "sample_index": index,
            "sample_id": row["event_id"],
            "class_name": row["class_name"],
            "source_filename": row["source_filename"],
        }


def train_bead_ssl_v2(
    config: dict[str, Any],
    *,
    simulation_root: Path,
    real_event_root: Path,
    real_signal_root: Path,
    output_dir: Path,
    profile_name: str,
    device_name: str,
) -> dict[str, Any]:
    profile = config["training"]["profiles"][profile_name]
    train = Z8AsymmetricSyntheticDataset(
        simulation_root,
        split=config["data"]["simulation_train_split"],
        normalization=config["data"]["normalization"],
        sampling_frequency_hz=float(config["data"]["sampling_frequency_hz"]),
        max_samples=profile["max_simulation_train"],
    )
    validation = Z8AsymmetricSyntheticDataset(
        simulation_root,
        split=config["data"]["simulation_validation_split"],
        normalization=config["data"]["normalization"],
        sampling_frequency_hz=float(config["data"]["sampling_frequency_hz"]),
        max_samples=profile["max_simulation_validation"],
    )
    real = Z8RealValidationDataset(
        real_event_root,
        real_signal_root,
        split=config["data"]["real_validation_split"],
        normalization=config["data"]["normalization"],
        max_per_class=profile["max_real_validation_per_class"],
    )
    monitoring = config["training"]["matched_monitoring"]
    train_indices, train_monitor = fixed_class_proportional_monitor_indices(
        train,
        max_samples=int(monitoring["samples_per_split"]),
        seed=int(config["training"]["seed"]),
        split_tag="train",
    )
    validation_indices, validation_monitor = fixed_class_proportional_monitor_indices(
        validation,
        max_samples=int(monitoring["samples_per_split"]),
        seed=int(config["training"]["seed"]),
        split_tag="val",
    )
    return train_bead_ssl(
        config,
        simulation_root=None,
        real_root=None,
        output_dir=output_dir,
        profile_name=profile_name,
        device_name=device_name,
        prepared_datasets=(train, validation, real),
        monitoring_datasets=(Subset(train, train_indices), Subset(validation, validation_indices)),
        monitoring_metadata={
            "protocol": "matched-policy-fixed-monitor-v1",
            "evaluation_policy": config["masking"]["training_policy"],
            "cyclic_schedule": "all_unique_passes_per_sample",
            "train": train_monitor,
            "validation": validation_monitor,
        },
    )
