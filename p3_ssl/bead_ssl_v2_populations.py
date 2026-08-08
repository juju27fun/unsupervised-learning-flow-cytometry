from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .bead_representation_benchmark import BenchmarkPopulation
from .bead_ssl_v2 import PHYSICAL_CLASSES, Z8RealValidationDataset
from .decimation import normalize_signal


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_v5_population(
    root: Path,
    *,
    split: str,
    normalization: str = "window_zscore",
) -> BenchmarkPopulation:
    if split not in {"train", "val"}:
        raise PermissionError(f"v5 population split is development-only: {split}")
    all_rows = _rows(root / "events.csv")
    metadata = [row for row in all_rows if row["noise_source_split"] == split]
    if not metadata:
        raise ValueError(f"no v5 rows for split={split}")
    source = np.load(root / "signals_raw_4096.npy", mmap_mode="r", allow_pickle=False)
    index_by_id = {row["sample_id"]: index for index, row in enumerate(all_rows)}
    signals = np.stack(
        [
            normalize_signal(
                np.asarray(source[index_by_id[row["sample_id"]]], dtype=np.float32),
                mode=normalization,
            )
            for row in metadata
        ]
    ).astype(np.float32)
    labels = np.asarray(
        [[float(row["tau_ms"]), float(row["frequency_khz"])] for row in metadata],
        dtype=np.float64,
    )
    return BenchmarkPopulation(
        signals=signals,
        ids=np.asarray([row["sample_id"] for row in metadata], dtype=str),
        groups=np.asarray([row["noise_source_relative_path"] for row in metadata], dtype=str),
        labels=labels,
        metadata=tuple(metadata),
    )


def load_z8_v2_population(
    event_root: Path,
    signal_root: Path,
    *,
    split: str,
    normalization: str = "window_zscore",
) -> BenchmarkPopulation:
    dataset = Z8RealValidationDataset(
        event_root,
        signal_root,
        split=split,
        normalization=normalization,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    class_to_id = {name: index for index, name in enumerate(PHYSICAL_CLASSES)}
    return BenchmarkPopulation(
        signals=np.stack([sample["signal"].numpy()[0] for sample in samples]).astype(
            np.float32
        ),
        ids=np.asarray([sample["sample_id"] for sample in samples], dtype=str),
        groups=np.asarray([sample["source_filename"] for sample in samples], dtype=str),
        labels=np.asarray(
            [class_to_id[str(sample["class_name"])] for sample in samples],
            dtype=np.int64,
        ),
        metadata=tuple(dataset.rows),
    )
