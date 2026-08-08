from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PARAMETERS = (
    "log_A_A",
    "fD_A_khz",
    "log_tau_A_ms",
    "snr_db",
    "log_B_over_A",
    "delta_t0_ms",
    "delta_fD_khz",
    "delta_phi_rad",
    "log_tau_B_over_tau_A",
)
QUANTILE_PROBABILITIES = np.linspace(0.01, 0.99, 31)


@dataclass(frozen=True)
class DomainInputs:
    primary_dataset_id: str
    gold_dataset_id: str
    fit_run_id: str
    fit_normalization_std: float
    target_normalization_std: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def wrap_phase(value: float) -> float:
    return float(np.angle(np.exp(1j * float(value))))


def circular_center(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Circular center requires finite values")
    vector = np.mean(np.exp(1j * array))
    if abs(vector) <= 1.0e-12:
        return 0.0
    return float(np.angle(vector))


def unwrap_around(values: np.ndarray, center: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return center + np.angle(np.exp(1j * (values - center)))


def fit_is_resolved(row: dict[str, Any]) -> bool:
    try:
        values = [
            float(row["delta_bic_m1_minus_m2"]),
            float(row["resolvability_score"]),
            *(
                float(row[f"m2_c{component}_{field}"])
                for component in (1, 2)
                for field in (
                    "amplitude",
                    "center_ms",
                    "sigma_left_ms",
                    "sigma_right_ms",
                    "frequency_khz",
                    "phase_rad",
                )
            ),
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        np.isfinite(values).all()
        and values[0] >= 10.0
        and values[1] >= 0.1
    )


def _component(row: dict[str, Any], index: int) -> dict[str, float]:
    prefix = f"m2_c{index}_"
    return {
        "amplitude": float(row[prefix + "amplitude"]),
        "center_ms": float(row[prefix + "center_ms"]),
        "sigma_left_ms": float(row[prefix + "sigma_left_ms"]),
        "sigma_right_ms": float(row[prefix + "sigma_right_ms"]),
        "shape": float(row[prefix + "shape"]),
        "frequency_khz": float(row[prefix + "frequency_khz"]),
        "chirp_khz_per_ms": float(row[prefix + "chirp_khz_per_ms"]),
        "phase_rad": float(row[prefix + "phase_rad"]),
    }


def snr_db_from_signal(
    signal: np.ndarray,
    *,
    event_start_index: float,
    event_end_index: float,
) -> float:
    values = np.asarray(signal, dtype=np.float64)
    start = max(0, int(math.floor(event_start_index)))
    end = min(values.size, int(math.ceil(event_end_index)))
    if end <= start:
        raise ValueError("Invalid event bounds")
    outside = np.ones(values.size, dtype=bool)
    outside[start:end] = False
    if not np.any(outside):
        raise ValueError("Event occupies the complete signal")
    event_rms = float(np.sqrt(np.mean(np.square(values[start:end]))))
    noise_rms = float(np.sqrt(np.mean(np.square(values[outside]))))
    return float(20.0 * np.log10(max(event_rms, 1.0e-12) / max(noise_rms, 1.0e-12)))


def canonical_parameter_row(
    fit: dict[str, Any],
    *,
    snr_db: float,
    amplitude_scale: float,
    population: str,
) -> dict[str, Any]:
    first, second = sorted(
        (_component(fit, 1), _component(fit, 2)),
        key=lambda component: (component["center_ms"], -component["amplitude"]),
    )
    tau_a = 0.5 * (first["sigma_left_ms"] + first["sigma_right_ms"])
    tau_b = 0.5 * (second["sigma_left_ms"] + second["sigma_right_ms"])
    amplitude_a = first["amplitude"] * float(amplitude_scale)
    amplitude_b = second["amplitude"] * float(amplitude_scale)
    if min(tau_a, tau_b, amplitude_a, amplitude_b) <= 0.0:
        raise ValueError("Positive amplitudes and widths are required")
    return {
        "event_id": str(fit["event_id"]),
        "population": population,
        "fit_valid": fit_is_resolved(fit),
        "delta_bic_m1_minus_m2": float(fit["delta_bic_m1_minus_m2"]),
        "resolvability_score": float(fit["resolvability_score"]),
        "log_A_A": float(np.log(amplitude_a)),
        "fD_A_khz": first["frequency_khz"],
        "log_tau_A_ms": float(np.log(tau_a)),
        "snr_db": float(snr_db),
        "log_B_over_A": float(np.log(amplitude_b / amplitude_a)),
        "delta_t0_ms": second["center_ms"] - first["center_ms"],
        "delta_fD_khz": second["frequency_khz"] - first["frequency_khz"],
        "delta_phi_rad": wrap_phase(second["phase_rad"] - first["phase_rad"]),
        "log_tau_B_over_tau_A": float(np.log(tau_b / tau_a)),
        "anchor_shape_A": first["shape"],
        "anchor_shape_B": second["shape"],
        "anchor_sigma_left_over_tau_A": first["sigma_left_ms"] / tau_a,
        "anchor_sigma_right_over_tau_A": first["sigma_right_ms"] / tau_a,
        "anchor_sigma_left_over_tau_B": second["sigma_left_ms"] / tau_b,
        "anchor_sigma_right_over_tau_B": second["sigma_right_ms"] / tau_b,
        "anchor_chirp_A_khz_per_ms": first["chirp_khz_per_ms"],
        "anchor_chirp_B_khz_per_ms": second["chirp_khz_per_ms"],
    }


def finalize_phase(
    rows: list[dict[str, Any]],
    *,
    reference_center: float | None = None,
) -> float:
    selected = [row for row in rows if bool(row["fit_valid"])]
    center = (
        circular_center(row["delta_phi_rad"] for row in selected)
        if reference_center is None
        else float(reference_center)
    )
    unwrapped = unwrap_around(
        np.asarray([row["delta_phi_rad"] for row in selected], dtype=np.float64),
        center,
    )
    for row, value in zip(selected, unwrapped, strict=True):
        row["delta_phi_rad"] = float(value)
    return center


def domain_statistics(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = [row for row in rows if bool(row["fit_valid"])]
    if len(selected) < 8:
        raise ValueError("At least eight valid fits are required")
    statistics: list[dict[str, Any]] = []
    grids: list[dict[str, Any]] = []
    for parameter in PARAMETERS:
        values = np.asarray([row[parameter] for row in selected], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite parameter: {parameter}")
        quantiles = np.quantile(values, QUANTILE_PROBABILITIES)
        statistics.append(
            {
                "parameter": parameter,
                "n": int(values.size),
                "q01": float(quantiles[0]),
                "q50": float(np.quantile(values, 0.5)),
                "q99": float(quantiles[-1]),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )
        grids.extend(
            {
                "parameter": parameter,
                "quantile_index": index,
                "probability": float(probability),
                "value": float(value),
            }
            for index, (probability, value) in enumerate(
                zip(QUANTILE_PROBABILITIES, quantiles, strict=True)
            )
        )
    return statistics, grids


def robust_medoid(rows: list[dict[str, Any]]) -> str:
    selected = [row for row in rows if bool(row["fit_valid"])]
    matrix = np.asarray(
        [[float(row[parameter]) for parameter in PARAMETERS] for row in selected],
        dtype=np.float64,
    )
    center = np.median(matrix, axis=0)
    mad = 1.4826 * np.median(np.abs(matrix - center), axis=0)
    q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
    scale = np.where(mad > 1.0e-9, mad, np.where(q75 - q25 > 1.0e-9, q75 - q25, 1.0))
    distances = np.sum(np.abs((matrix - center) / scale), axis=1)
    order = sorted(
        range(len(selected)),
        key=lambda index: (float(distances[index]), str(selected[index]["event_id"])),
    )
    return str(selected[order[0]]["event_id"])


def sensitivity_rows(
    primary_stats: list[dict[str, Any]],
    gold_stats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary = {row["parameter"]: row for row in primary_stats}
    gold = {row["parameter"]: row for row in gold_stats}
    return [
        {
            "parameter": parameter,
            "primary_n": primary[parameter]["n"],
            "gold_n": gold[parameter]["n"],
            "primary_q01": primary[parameter]["q01"],
            "primary_q99": primary[parameter]["q99"],
            "gold_q01": gold[parameter]["q01"],
            "gold_q99": gold[parameter]["q99"],
            "q01_shift": gold[parameter]["q01"] - primary[parameter]["q01"],
            "q99_shift": gold[parameter]["q99"] - primary[parameter]["q99"],
        }
        for parameter in PARAMETERS
    ]
