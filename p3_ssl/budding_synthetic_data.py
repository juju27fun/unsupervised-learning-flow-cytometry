from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


GEOMETRY_FACTORS = (
    "mother_radius_relative",
    "bud_radius_ratio",
    "orientation_cosine",
)
SUPPORTED_DATASET_IDS = frozenset(
    {
        "yeast-budding-simulations-data@v1",
        "yeast-budding-simulations-biophysics@v1",
    }
)


@dataclass(frozen=True)
class BuddingSimulationContract:
    dataset_id: str
    generator_id: str
    input_contract: str
    signal_shape: tuple[int, int]
    views_per_latent: int
    split_signal_counts: dict[str, int]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _asymmetric_envelope(
    time_ms: np.ndarray,
    *,
    center_ms: float,
    sigma_left_ms: float,
    sigma_right_ms: float,
    shape: float,
) -> np.ndarray:
    scale = np.where(time_ms < center_ms, sigma_left_ms, sigma_right_ms)
    distance = np.abs((time_ms - center_ms) / np.maximum(scale, 1.0e-9))
    return np.exp(-0.5 * np.power(distance, shape))


def budding_event_support_mask(
    row: dict[str, str],
    *,
    length: int = 4096,
    sampling_frequency_hz: float = 1_000_000.0,
    relative_threshold: float = 0.25,
) -> np.ndarray:
    """Derive the generator's two-component envelope support at output rate."""
    if length <= 0 or sampling_frequency_hz <= 0.0:
        raise ValueError("length and sampling_frequency_hz must be positive")
    if not 0.0 < relative_threshold < 1.0:
        raise ValueError("relative_threshold must lie strictly between zero and one")
    time_ms = np.arange(length, dtype=np.float64) / sampling_frequency_hz * 1000.0
    first = _asymmetric_envelope(
        time_ms,
        center_ms=float(row["component1_center_ms"]),
        sigma_left_ms=float(row["sigma1_left_ms"]),
        sigma_right_ms=float(row["sigma1_right_ms"]),
        shape=float(row["shape1"]),
    )
    second = _asymmetric_envelope(
        time_ms,
        center_ms=float(row["component2_center_ms"]),
        sigma_left_ms=float(row["sigma2_left_ms"]),
        sigma_right_ms=float(row["sigma2_right_ms"]),
        shape=float(row["shape2"]),
    )
    combined = first + float(row["relative_amplitude"]) * second
    peak = float(np.max(combined))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("generator envelope must have a finite positive peak")
    return np.asarray(combined >= relative_threshold * peak, dtype=bool)


def validate_budding_simulation_dataset(
    root: Path,
    *,
    expected_dataset_id: str | None = None,
) -> BuddingSimulationContract:
    summary = json.loads((root / "dataset_summary.json").read_text(encoding="utf-8"))
    dataset_id = str(summary.get("dataset_id", ""))
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise ValueError(
            f"Dataset identity mismatch: summary={dataset_id!r}, expected={expected_dataset_id!r}"
        )
    if dataset_id not in SUPPORTED_DATASET_IDS:
        raise ValueError(f"Unsupported budding simulation dataset: {dataset_id!r}")
    signals = np.load(root / "signals.npy", mmap_mode="r")
    if signals.shape != (14_000, 4096) or signals.dtype != np.float32:
        raise ValueError(
            f"Expected float32 signals with shape (14000, 4096), got {signals.dtype} {signals.shape}"
        )
    if not np.isfinite(np.asarray(signals[: min(32, len(signals))])).all():
        raise ValueError("Dataset smoke sample contains NaN or Inf")
    if summary.get("input_contract") != "yeast-event-8192to4096-bandpass-global-v1":
        raise ValueError("Unexpected input contract")
    if int(summary.get("views_per_latent", -1)) != 2:
        raise ValueError("Budding SSL datasets require exactly two views per latent")
    split_counts = {
        str(key): int(value)
        for key, value in dict(summary.get("split_signal_counts", {})).items()
    }
    if split_counts != {"train": 10_000, "validation": 2_000, "test": 2_000}:
        raise ValueError(f"Unexpected split counts: {split_counts}")
    return BuddingSimulationContract(
        dataset_id=dataset_id,
        generator_id=str(summary["generator_id"]),
        input_contract=str(summary["input_contract"]),
        signal_shape=(int(signals.shape[0]), int(signals.shape[1])),
        views_per_latent=2,
        split_signal_counts=split_counts,
    )


class BuddingSimulationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: Path,
        split: str,
        *,
        expected_dataset_id: str | None = None,
        max_latents: int | None = None,
        support_threshold: float = 0.25,
    ) -> None:
        self.root = root
        self.contract = validate_budding_simulation_dataset(
            root,
            expected_dataset_id=expected_dataset_id,
        )
        self.signals = np.load(root / "signals.npy", mmap_mode="r")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in _read_csv(root / "simulation_metadata.csv"):
            if row["split"] == split:
                grouped[row["latent_id"]].append(row)
        latent_rows: list[list[dict[str, str]]] = []
        for latent_id, rows in sorted(grouped.items()):
            ordered = sorted(rows, key=lambda item: int(item["view_index"]))
            if [int(row["view_index"]) for row in ordered] != [0, 1]:
                raise ValueError(f"Latent {latent_id} must contain views 0 and 1 exactly once")
            if len({row["generator_model"] for row in ordered}) != 1:
                raise ValueError(f"Latent {latent_id} changes generator model across views")
            latent_rows.append(ordered)
        if max_latents is not None:
            latent_rows = latent_rows[:max_latents]
        if not latent_rows:
            raise ValueError(f"No budding simulation latents for split={split!r}")
        self.latent_rows = latent_rows
        self.support_threshold = support_threshold

    def __len__(self) -> int:
        return len(self.latent_rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rows = self.latent_rows[index]
        signals = np.stack(
            [
                np.array(
                    self.signals[int(row["signal_row"])],
                    dtype=np.float32,
                    copy=True,
                )
                for row in rows
            ]
        )
        supports = np.stack(
            [
                budding_event_support_mask(
                    row,
                    length=signals.shape[1],
                    relative_threshold=self.support_threshold,
                )
                for row in rows
            ]
        )
        geometry_values = []
        geometry_valid = []
        for name in GEOMETRY_FACTORS:
            raw = rows[0].get(name, "")
            valid = raw not in {"", None}
            geometry_values.append(float(raw) if valid else 0.0)
            geometry_valid.append(valid)
        return {
            "signals": torch.from_numpy(signals).unsqueeze(1),
            "event_masks": torch.from_numpy(supports),
            "geometry_targets": torch.tensor(geometry_values, dtype=torch.float32),
            "geometry_valid": torch.tensor(geometry_valid, dtype=torch.bool),
            "latent_id": rows[0]["latent_id"],
            "dataset_id": self.contract.dataset_id,
            "generator_model": rows[0]["generator_model"],
            "resolved": rows[0]["resolved"].lower() == "true",
        }
