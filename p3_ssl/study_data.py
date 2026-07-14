from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


CONTINUOUS_FACTORS = (
    "duration_ms",
    "doppler_khz",
    "component_separation_ms",
    "relative_component_amplitude",
    "frequency_separation_khz",
)
FACTOR_RANGES = {
    "duration_ms": (0.464, 1.424),
    "doppler_khz": (7.8125, 23.4375),
    "component_separation_ms": (0.08, 0.70),
    "relative_component_amplitude": (0.40, 1.00),
    "frequency_separation_khz": (0.0, 8.0),
}
SEALED_REAL_SPLITS = frozenset({"sealed_acquisition_test"})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_study_dataset_contracts(real_root: Path, simulation_root: Path) -> dict[str, Any]:
    real_contract = json.loads((real_root / "input_contract.json").read_text(encoding="utf-8"))
    simulation_summary = json.loads(
        (simulation_root / "dataset_summary.json").read_text(encoding="utf-8")
    )
    real_signals = np.load(real_root / "signals.npy", mmap_mode="r")
    simulation_signals = np.load(simulation_root / "signals.npy", mmap_mode="r")
    errors: list[str] = []
    if real_contract.get("contract_id") != "yeast-event-8192to4096-bandpass-global-v1":
        errors.append("unexpected real input contract")
    if int(real_contract.get("output_length", -1)) != 4096:
        errors.append("real output length is not 4096")
    if list(real_signals.shape)[1:] != [4096]:
        errors.append(f"real signal shape is incompatible: {real_signals.shape}")
    if list(simulation_signals.shape)[1:] != [4096]:
        errors.append(f"simulation signal shape is incompatible: {simulation_signals.shape}")
    if "4096" not in str(simulation_summary.get("input_contract", "")):
        errors.append("simulation summary does not declare the 4096 contract")
    return {
        "valid": not errors,
        "errors": errors,
        "real_shape": list(real_signals.shape),
        "simulation_shape": list(simulation_signals.shape),
        "real_contract": real_contract["contract_id"],
        "simulation_generator": simulation_summary["generator_id"],
    }


class RealEventDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: Path,
        split: str,
        max_events: int | None = None,
        *,
        allow_sealed_split: bool = False,
    ) -> None:
        if split in SEALED_REAL_SPLITS and not allow_sealed_split:
            raise PermissionError(
                f"Split {split} is sealed; open it explicitly only for the frozen final evaluation"
            )
        self.root = root
        self.signals = np.load(root / "signals.npy", mmap_mode="r")
        rows = [row for row in _read_csv(root / "events.csv") if row["development_split"] == split]
        if max_events is not None:
            rows = rows[:max_events]
        if not rows:
            raise ValueError(f"No real events for split={split}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        signal = np.array(self.signals[int(row["signal_row"])], dtype=np.float32, copy=True)
        event_mask = np.zeros(signal.size, dtype=bool)
        start = max(0, min(signal.size, int(round(float(row["event_start_input_index"])))))
        end = max(0, min(signal.size, int(round(float(row["event_end_input_index"])))))
        event_mask[start:end] = True
        return {
            "signal": torch.from_numpy(signal).unsqueeze(0),
            "event_mask": torch.from_numpy(event_mask),
            "event_id": row["event_id"],
            "record_id": row["record_id"],
            "source_group": row["source_group"],
            "condition_id": row["condition_id"],
            "acquisition_id": row["acquisition_id"],
            "acquisition_role": row.get("acquisition_role", ""),
            "quality": row["quality"],
        }


def _normalized_factors(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    component_count = int(row["component_count"])
    values: list[float] = []
    valid: list[bool] = []
    for name in CONTINUOUS_FACTORS:
        low, high = FACTOR_RANGES[name]
        value = float(row[name])
        conditional = name in {
            "component_separation_ms",
            "relative_component_amplitude",
            "frequency_separation_khz",
        }
        is_valid = not conditional or component_count == 2
        values.append((value - low) / (high - low) if is_valid else 0.0)
        valid.append(is_valid)
    return np.asarray(values, dtype=np.float32), np.asarray(valid, dtype=bool)


class SimulatedLatentDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: Path, split: str, max_latents: int | None = None) -> None:
        self.root = root
        self.signals = np.load(root / "signals.npy", mmap_mode="r")
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in _read_csv(root / "simulation_metadata.csv"):
            if row["split"] == split:
                grouped[row["latent_id"]].append(row)
        latent_rows = []
        for latent_id, views in sorted(grouped.items()):
            ordered = sorted(views, key=lambda row: int(row["view_index"]))
            if len(ordered) < 2:
                raise ValueError(f"Latent {latent_id} has fewer than two nuisance views")
            latent_rows.append(ordered)
        if max_latents is not None:
            latent_rows = latent_rows[:max_latents]
        if not latent_rows:
            raise ValueError(f"No simulation latents for split={split}")
        self.latent_rows = latent_rows

    def __len__(self) -> int:
        return len(self.latent_rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rows = self.latent_rows[index]
        signals = np.stack(
            [np.array(self.signals[int(row["signal_row"])], dtype=np.float32, copy=True) for row in rows]
        )
        targets, valid = _normalized_factors(rows[0])
        event_masks = np.zeros_like(signals, dtype=bool)
        for view_index, row in enumerate(rows):
            center = float(row["event_position_fraction"]) * (signals.shape[1] - 1)
            half_width = float(row["duration_ms"]) / 1000.0 * 1_000_000.0 / 2.0
            start = max(0, int(round(center - half_width)))
            end = min(signals.shape[1], int(round(center + half_width)))
            event_masks[view_index, start:end] = True
        return {
            "signals": torch.from_numpy(signals).unsqueeze(1),
            "event_masks": torch.from_numpy(event_masks),
            "continuous_targets": torch.from_numpy(targets),
            "continuous_valid": torch.from_numpy(valid),
            "component_target": torch.as_tensor(int(rows[0]["component_count"]) - 1, dtype=torch.long),
            "latent_id": rows[0]["latent_id"],
            "generator_variant": rows[0]["generator_variant"],
        }
