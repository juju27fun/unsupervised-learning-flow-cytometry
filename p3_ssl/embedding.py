from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .data import ManifestRow, parse_yolo_1d_labels
from .masking import PatchSpec


CLASS_NAMES = {
    0: "2um",
    1: "4um",
    2: "10um",
    3: "unclear",
}


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    sample_id: str
    split: str
    signal_path: Path
    class_id: int
    class_name: str
    center_norm: float
    width_norm: float
    start: int
    end: int
    source_kind: str


def label_to_interval(center_norm: float, width_norm: float, input_length: int) -> tuple[int, int]:
    """Convert normalized YOLO 1D center/width to a clipped sample interval."""
    start = int(round((float(center_norm) - float(width_norm) / 2.0) * input_length))
    end = int(round((float(center_norm) + float(width_norm) / 2.0) * input_length))
    start = max(0, min(input_length, start))
    end = max(0, min(input_length, end))
    if end <= start:
        end = min(input_length, start + 1)
    return start, end


def token_indices_for_interval(start: int, end: int, spec: PatchSpec) -> np.ndarray:
    """Return token indices whose patch intersects [start, end)."""
    if end <= start:
        raise ValueError("event interval must be non-empty")
    spans = spec.spans
    hits = np.flatnonzero((spans[:, 0] < end) & (spans[:, 1] > start))
    if hits.size == 0:
        nearest = int(np.clip(start // spec.patch_stride, 0, spec.n_tokens - 1))
        hits = np.asarray([nearest], dtype=np.int64)
    return hits.astype(np.int64, copy=False)


def pool_token_embeddings(tokens: torch.Tensor, token_indices: Iterable[int]) -> torch.Tensor:
    """Average selected token embeddings from a `(T, D)` tensor."""
    idx = torch.as_tensor(list(token_indices), dtype=torch.long, device=tokens.device)
    if idx.numel() == 0:
        raise ValueError("token_indices cannot be empty")
    return tokens.index_select(0, idx).mean(dim=0)


def collect_events(
    rows: list[ManifestRow],
    input_length: int,
    class_names: dict[int, str] | None = None,
) -> list[EventRecord]:
    """Create one event record per YOLO label row."""
    names = class_names or CLASS_NAMES
    events: list[EventRecord] = []
    per_sample_count: dict[str, int] = defaultdict(int)
    for row in rows:
        labels = parse_yolo_1d_labels(row.label_path)
        for class_float, center, width in labels:
            class_id = int(class_float)
            if class_id not in names:
                continue
            start, end = label_to_interval(float(center), float(width), input_length)
            local_idx = per_sample_count[row.sample_id]
            per_sample_count[row.sample_id] += 1
            events.append(
                EventRecord(
                    event_id=f"{row.sample_id}::{local_idx}",
                    sample_id=row.sample_id,
                    split=row.split,
                    signal_path=row.signal_path,
                    class_id=class_id,
                    class_name=names[class_id],
                    center_norm=float(center),
                    width_norm=float(width),
                    start=start,
                    end=end,
                    source_kind=row.source_kind,
                )
            )
    return events


def balanced_event_indices(
    class_ids: np.ndarray,
    max_per_class: int,
    seed: int = 42,
) -> np.ndarray:
    """Return deterministic class-balanced indices for plotting."""
    if max_per_class <= 0:
        raise ValueError("max_per_class must be positive")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in sorted(set(int(c) for c in class_ids.tolist())):
        idx = np.flatnonzero(class_ids == class_id)
        if idx.size > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.extend(int(i) for i in idx)
    selected_arr = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_arr)
    return selected_arr


def event_records_to_metadata(events: list[EventRecord]) -> dict[str, np.ndarray]:
    """Convert event metadata to numpy arrays suitable for `.npz` storage."""
    return {
        "event_id": np.asarray([e.event_id for e in events]),
        "sample_id": np.asarray([e.sample_id for e in events]),
        "split": np.asarray([e.split for e in events]),
        "signal_path": np.asarray([str(e.signal_path) for e in events]),
        "class_id": np.asarray([e.class_id for e in events], dtype=np.int64),
        "class_name": np.asarray([e.class_name for e in events]),
        "center_norm": np.asarray([e.center_norm for e in events], dtype=np.float32),
        "width_norm": np.asarray([e.width_norm for e in events], dtype=np.float32),
        "start": np.asarray([e.start for e in events], dtype=np.int64),
        "end": np.asarray([e.end for e in events], dtype=np.int64),
        "source_kind": np.asarray([e.source_kind for e in events]),
    }
