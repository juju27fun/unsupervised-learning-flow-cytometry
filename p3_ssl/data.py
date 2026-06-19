from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .decimation import crop_or_pad, decimate_signal, ensure_1d_signal, normalize_signal
from .masking import PatchSpec, build_ssl_masks


@dataclass(frozen=True)
class ManifestRow:
    split: str
    sample_id: str
    signal_path: Path
    label_path: Path | None
    source_kind: str = "unknown"


def _resolve_manifest_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    candidate = manifest_path.parent / path
    if candidate.exists():
        return candidate
    return path


def read_manifest(path: str | Path, split: str | None = None) -> list[ManifestRow]:
    manifest_path = Path(path)
    rows: list[ManifestRow] = []
    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row_split = raw.get("split", "")
            if split is not None and row_split != split:
                continue
            signal = raw.get("signal_path") or raw.get("source_path") or raw.get("feature_path")
            if not signal:
                raise ValueError(f"Missing signal path in manifest row: {raw}")
            label = raw.get("label_path") or ""
            rows.append(
                ManifestRow(
                    split=row_split,
                    sample_id=raw.get("id") or raw.get("sample_id") or Path(signal).stem,
                    signal_path=_resolve_manifest_path(signal, manifest_path),
                    label_path=_resolve_manifest_path(label, manifest_path) if label else None,
                    source_kind=raw.get("source_kind", "unknown"),
                )
            )
    return rows


def parse_yolo_1d_labels(path: str | Path | None) -> np.ndarray:
    """Parse labels as rows of class, center, width normalized to [0, 1]."""
    if path is None:
        return np.zeros((0, 3), dtype=np.float32)
    p = Path(path)
    if not p.exists():
        return np.zeros((0, 3), dtype=np.float32)
    rows: list[list[float]] = []
    with p.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not rows:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def labels_to_event_mask(labels: np.ndarray, input_length: int) -> np.ndarray:
    mask = np.zeros(input_length, dtype=bool)
    for _, center, width in labels:
        start = int(round((float(center) - float(width) / 2.0) * input_length))
        end = int(round((float(center) + float(width) / 2.0) * input_length))
        start = max(0, min(input_length, start))
        end = max(0, min(input_length, end))
        if end > start:
            mask[start:end] = True
    return mask


class SSLManifestDataset(Dataset[dict[str, Any]]):
    """Dataset for masked reconstruction over independently listed `.npy` files."""

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str = "train",
        input_length_raw: int = 16384,
        decimation_factor: int = 8,
        input_length_ssl: int = 2048,
        normalization: str = "window_zscore",
        patch_size: int = 4,
        patch_stride: int = 4,
        guard_points: int = 8,
        mask_ratio: float = 0.25,
        min_block_length: int = 24,
        max_block_length: int = 128,
        high_derivative_probability: float = 0.25,
        decimation_method: str = "mean",
        seed: int = 42,
    ) -> None:
        self.rows = read_manifest(manifest_csv, split=split)
        if not self.rows:
            raise ValueError(f"No rows found for split={split} in {manifest_csv}")
        self.input_length_raw = input_length_raw
        self.decimation_factor = decimation_factor
        self.input_length_ssl = input_length_ssl
        self.normalization = normalization
        self.spec = PatchSpec(input_length_ssl, patch_size, patch_stride)
        self.guard_points = guard_points
        self.mask_ratio = mask_ratio
        self.min_block_length = min_block_length
        self.max_block_length = max_block_length
        self.high_derivative_probability = high_derivative_probability
        self.decimation_method = decimation_method
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def _load_signal(self, path: Path) -> np.ndarray:
        arr = np.load(path)
        signal = ensure_1d_signal(arr)
        signal = crop_or_pad(signal, self.input_length_raw, mode="center")
        signal = decimate_signal(signal, self.decimation_factor, method=self.decimation_method)
        signal = crop_or_pad(signal, self.input_length_ssl, mode="center")
        return normalize_signal(signal, mode=self.normalization)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        signal = self._load_signal(row.signal_path)
        rng = np.random.default_rng(self.seed + idx + int(torch.randint(0, 2**16, ()).item()))
        masks = build_ssl_masks(
            signal=signal,
            spec=self.spec,
            rng=rng,
            mask_ratio=self.mask_ratio,
            min_block_length=self.min_block_length,
            max_block_length=self.max_block_length,
            guard_points=self.guard_points,
            high_derivative_probability=self.high_derivative_probability,
        )
        labels = parse_yolo_1d_labels(row.label_path)
        event_mask = labels_to_event_mask(labels, self.input_length_ssl)
        return {
            "signal": torch.from_numpy(signal).float().unsqueeze(0),
            "target": torch.from_numpy(signal).float().unsqueeze(0),
            "target_time_mask": torch.from_numpy(masks["target_time_mask"]).bool(),
            "hidden_time_mask": torch.from_numpy(masks["hidden_time_mask"]).bool(),
            "token_mask": torch.from_numpy(masks["token_mask"]).bool(),
            "event_mask": torch.from_numpy(event_mask).bool(),
            "sample_id": row.sample_id,
            "source_kind": row.source_kind,
        }

