from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import hilbert
from scipy.stats import rankdata

from particles2snr.yeast_representation_dataset import preprocess_crop


INPUT_LENGTH = 4096
RAW_LENGTH = 8192
RAW_FS_HZ = 2_000_000.0
OUTPUT_FS_HZ = 1_000_000.0
WINDOW_DURATION_MS = 4.096
METHOD_EVIDENCE_ID = "yeast-budding-m2-resnet-stft-latent-sweep-method-r1"
DENSE_METHOD_EVIDENCE_ID = "yeast-budding-m2-dense-atlas-method-r1"
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
CLASS_NAMES = ("background", "budding", "mix", "shmoo")
COMMON_PHASE_OFFSETS = (0.0, math.pi / 2.0)
POSITION_FRACTIONS = (0.40, 0.60)


@dataclass(frozen=True)
class Carrier:
    carrier_id: str
    record_id: str
    signal_row: int
    energy_quantile: float
    values: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wrap_phase(value: float | np.ndarray) -> float | np.ndarray:
    wrapped = (np.asarray(value) + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped) if np.ndim(value) == 0 else wrapped


def load_parameter_grid(path: Path) -> dict[str, np.ndarray]:
    rows = read_csv(path)
    grids: dict[str, list[tuple[int, float]]] = {parameter: [] for parameter in PARAMETERS}
    for row in rows:
        parameter = row["parameter"]
        if parameter not in grids:
            raise ValueError(f"Unknown sweep parameter: {parameter}")
        grids[parameter].append((int(row["quantile_index"]), float(row["value"])))
    output: dict[str, np.ndarray] = {}
    for parameter, values in grids.items():
        ordered = sorted(values)
        if [index for index, _ in ordered] != list(range(31)):
            raise ValueError(f"{parameter} must contain exactly quantile indices 0..30")
        array = np.asarray([value for _, value in ordered], dtype=np.float64)
        if not np.all(np.isfinite(array)) or np.any(np.diff(array) < 0.0):
            raise ValueError(f"Invalid empirical grid: {parameter}")
        output[parameter] = array
    return output


def interpolate_parameter_grids(
    grids: dict[str, np.ndarray],
    *,
    count: int = 225,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if count < 2:
        raise ValueError("Dense quantile count must be at least two")
    source_probabilities = np.linspace(0.01, 0.99, 31, dtype=np.float64)
    target_probabilities = np.linspace(0.01, 0.99, count, dtype=np.float64)
    dense: dict[str, np.ndarray] = {}
    for parameter in PARAMETERS:
        source = np.asarray(grids[parameter], dtype=np.float64)
        if source.shape != (31,) or np.any(np.diff(source) < 0.0):
            raise ValueError(f"Invalid source quantile grid: {parameter}")
        values = np.interp(target_probabilities, source_probabilities, source)
        if values.shape != (count,) or not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0.0):
            raise ValueError(f"Invalid interpolated quantile grid: {parameter}")
        dense[parameter] = values
    return dense, target_probabilities


def select_real_carriers(
    rows: Iterable[dict[str, str]],
    signals: np.ndarray,
) -> list[Carrier]:
    eligible = [
        row
        for row in rows
        if row["class_name"] == "background"
        and row["source_group_original"] == "budding"
        and row["development_split"] == "development_train"
    ]
    if len(eligible) < 4:
        raise ValueError("Insufficient budding development_train background carriers")
    energies = np.asarray([float(row["background_energy"]) for row in eligible], dtype=np.float64)
    selected: list[Carrier] = []
    used_records: set[str] = set()
    for quantile in (0.50, 0.75):
        target = float(np.quantile(energies, quantile))
        candidates = sorted(
            eligible,
            key=lambda row: (
                abs(float(row["background_energy"]) - target),
                row["record_id"],
                row["sample_id"],
            ),
        )
        row = next((candidate for candidate in candidates if candidate["record_id"] not in used_records), None)
        if row is None:
            raise ValueError("Could not select source-disjoint carriers")
        used_records.add(row["record_id"])
        values = np.asarray(signals[int(row["signal_row"])], dtype=np.float64)
        centered = values - float(np.mean(values))
        rms = float(np.sqrt(np.mean(np.square(centered))))
        if values.shape != (INPUT_LENGTH,) or rms <= 1.0e-12:
            raise ValueError("Invalid carrier signal")
        selected.append(
            Carrier(
                carrier_id=row["sample_id"],
                record_id=row["record_id"],
                signal_row=int(row["signal_row"]),
                energy_quantile=quantile,
                values=(centered / rms).astype(np.float32),
            )
        )
    return selected


def _component_raw(
    *,
    amplitude: float,
    center_ms: float,
    sigma_left_ms: float,
    sigma_right_ms: float,
    shape: float,
    frequency_khz: float,
    chirp_khz_per_ms: float,
    phase_rad: float,
) -> np.ndarray:
    time_ms = np.arange(RAW_LENGTH, dtype=np.float64) / RAW_FS_HZ * 1000.0
    relative = time_ms - center_ms
    scale = np.where(relative < 0.0, sigma_left_ms, sigma_right_ms)
    envelope = np.exp(-0.5 * np.power(np.abs(relative) / np.maximum(scale, 1.0e-12), shape))
    phase = 2.0 * np.pi * (
        frequency_khz * relative + 0.5 * chirp_khz_per_ms * np.square(relative)
    ) + phase_rad
    return (amplitude * envelope * np.cos(phase)).astype(np.float32)


def _filtered_component(component: np.ndarray, target_amplitude_v1: float) -> np.ndarray:
    filtered = preprocess_crop(component)
    peak = float(np.max(np.abs(hilbert(filtered.astype(np.float64)))))
    if peak <= 1.0e-12:
        raise ValueError("Degenerate filtered M2 component")
    return (filtered * (target_amplitude_v1 / peak)).astype(np.float32)


def physical_parameters(
    transformed: dict[str, float],
    anchor: dict[str, Any],
    *,
    common_phase: float,
    position_fraction: float,
    amplitude_v2_to_v1: float,
) -> dict[str, float]:
    amplitude_a = math.exp(transformed["log_A_A"]) * amplitude_v2_to_v1
    tau_a = math.exp(transformed["log_tau_A_ms"])
    amplitude_b = amplitude_a * math.exp(transformed["log_B_over_A"])
    tau_b = tau_a * math.exp(transformed["log_tau_B_over_tau_A"])
    separation = transformed["delta_t0_ms"]
    center = position_fraction * WINDOW_DURATION_MS
    return {
        "amplitude_a": amplitude_a,
        "amplitude_b": amplitude_b,
        "center_a_ms": center - separation / 2.0,
        "center_b_ms": center + separation / 2.0,
        "sigma_left_a_ms": tau_a * float(anchor["anchor_sigma_left_over_tau_A"]),
        "sigma_right_a_ms": tau_a * float(anchor["anchor_sigma_right_over_tau_A"]),
        "sigma_left_b_ms": tau_b * float(anchor["anchor_sigma_left_over_tau_B"]),
        "sigma_right_b_ms": tau_b * float(anchor["anchor_sigma_right_over_tau_B"]),
        "shape_a": float(anchor["anchor_shape_A"]),
        "shape_b": float(anchor["anchor_shape_B"]),
        "frequency_a_khz": transformed["fD_A_khz"],
        "frequency_b_khz": transformed["fD_A_khz"] + transformed["delta_fD_khz"],
        "chirp_a_khz_per_ms": float(anchor["anchor_chirp_A_khz_per_ms"]),
        "chirp_b_khz_per_ms": float(anchor["anchor_chirp_B_khz_per_ms"]),
        "phase_a_rad": wrap_phase(common_phase),
        "phase_b_rad": wrap_phase(common_phase + transformed["delta_phi_rad"]),
        "tau_a_ms": tau_a,
        "tau_b_ms": tau_b,
        "snr_db": transformed["snr_db"],
    }


def generate_signal(
    transformed: dict[str, float],
    anchor: dict[str, Any],
    carrier: Carrier,
    *,
    common_phase: float,
    position_fraction: float,
    amplitude_v2_to_v1: float,
) -> tuple[np.ndarray, dict[str, float]]:
    params = physical_parameters(
        transformed,
        anchor,
        common_phase=common_phase,
        position_fraction=position_fraction,
        amplitude_v2_to_v1=amplitude_v2_to_v1,
    )
    raw_a = _component_raw(
        amplitude=1.0,
        center_ms=params["center_a_ms"],
        sigma_left_ms=params["sigma_left_a_ms"],
        sigma_right_ms=params["sigma_right_a_ms"],
        shape=params["shape_a"],
        frequency_khz=params["frequency_a_khz"],
        chirp_khz_per_ms=params["chirp_a_khz_per_ms"],
        phase_rad=params["phase_a_rad"],
    )
    raw_b = _component_raw(
        amplitude=1.0,
        center_ms=params["center_b_ms"],
        sigma_left_ms=params["sigma_left_b_ms"],
        sigma_right_ms=params["sigma_right_b_ms"],
        shape=params["shape_b"],
        frequency_khz=params["frequency_b_khz"],
        chirp_khz_per_ms=params["chirp_b_khz_per_ms"],
        phase_rad=params["phase_b_rad"],
    )
    component_a = _filtered_component(raw_a, params["amplitude_a"])
    component_b = _filtered_component(raw_b, params["amplitude_b"])
    clean = component_a + component_b
    left_ms = min(
        params["center_a_ms"] - 4.0 * params["sigma_left_a_ms"],
        params["center_b_ms"] - 4.0 * params["sigma_left_b_ms"],
    )
    right_ms = max(
        params["center_a_ms"] + 4.0 * params["sigma_right_a_ms"],
        params["center_b_ms"] + 4.0 * params["sigma_right_b_ms"],
    )
    event_start = max(0, int(math.floor(left_ms * OUTPUT_FS_HZ / 1000.0)))
    event_end = min(INPUT_LENGTH, int(math.ceil(right_ms * OUTPUT_FS_HZ / 1000.0)))
    if event_end - event_start < 16:
        raise ValueError("Generated event support is too short")
    clean_rms = float(np.sqrt(np.mean(np.square(clean[event_start:event_end]))))
    carrier_rms = float(
        np.sqrt(np.mean(np.square(carrier.values[event_start:event_end])))
    )
    target_ratio = 10.0 ** (params["snr_db"] / 20.0)
    noise_scale = clean_rms / max(target_ratio * carrier_rms, 1.0e-12)
    noise = carrier.values * noise_scale
    realized = 20.0 * math.log10(
        max(clean_rms, 1.0e-12)
        / max(
            float(np.sqrt(np.mean(np.square(noise[event_start:event_end])))),
            1.0e-12,
        )
    )
    signal = clean + noise
    if signal.shape != (INPUT_LENGTH,) or not np.all(np.isfinite(signal)):
        raise ValueError("Invalid generated M2 signal")
    return signal.astype(np.float32), {
        **params,
        "event_start_index": event_start,
        "event_end_index": event_end,
        "noise_scale": noise_scale,
        "realized_construction_snr_db": realized,
    }


def build_sweep_bank(
    *,
    anchor: dict[str, Any],
    grids: dict[str, np.ndarray],
    carriers: list[Carrier],
    amplitude_v2_to_v1: float,
    parameter_subset: tuple[str, ...] = PARAMETERS,
    quantile_indices: tuple[int, ...] = tuple(range(31)),
    quantile_probabilities: tuple[float, ...] | None = None,
    phases: tuple[float, ...] = COMMON_PHASE_OFFSETS,
    positions: tuple[float, ...] = POSITION_FRACTIONS,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(carriers) != 2:
        raise ValueError("The full method requires exactly two carriers")
    transformed_anchor = {parameter: float(anchor[parameter]) for parameter in PARAMETERS}
    if quantile_probabilities is None:
        quantile_probabilities = tuple(0.01 + index * (0.98 / 30.0) for index in quantile_indices)
    if len(quantile_probabilities) != len(quantile_indices):
        raise ValueError("Quantile indices and probabilities must have equal length")
    if any(probability < 0.0 or probability > 1.0 for probability in quantile_probabilities):
        raise ValueError("Quantile probabilities must lie in [0, 1]")
    probability_by_index = dict(zip(quantile_indices, quantile_probabilities, strict=True))
    sample_id_width = max(2, len(str(max(quantile_indices, default=0))))
    contexts = [
        (phase_index, phase, position_index, position, carrier_index, carrier)
        for phase_index, phase in enumerate(phases)
        for position_index, position in enumerate(positions)
        for carrier_index, carrier in enumerate(carriers)
    ]
    signals: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for phase_index, phase, position_index, position, carrier_index, carrier in contexts:
        context_id = f"p{phase_index}-x{position_index}-n{carrier_index}"
        signal, physical = generate_signal(
            transformed_anchor,
            anchor,
            carrier,
            common_phase=phase,
            position_fraction=position,
            amplitude_v2_to_v1=amplitude_v2_to_v1,
        )
        signals.append(signal)
        metadata.append(
            {
                "sample_id": f"anchor:{context_id}",
                "sample_kind": "anchor",
                "sweep_parameter": "anchor",
                "quantile_index": -1,
                "quantile_probability": 0.5,
                "context_id": context_id,
                "phase_index": phase_index,
                "common_phase_rad": phase,
                "position_index": position_index,
                "position_fraction": position,
                "carrier_index": carrier_index,
                "carrier_id": carrier.carrier_id,
                "carrier_record_id": carrier.record_id,
                **transformed_anchor,
                **physical,
            }
        )
        for parameter in parameter_subset:
            for quantile_index in quantile_indices:
                transformed = dict(transformed_anchor)
                transformed[parameter] = float(grids[parameter][quantile_index])
                signal, physical = generate_signal(
                    transformed,
                    anchor,
                    carrier,
                    common_phase=phase,
                    position_fraction=position,
                    amplitude_v2_to_v1=amplitude_v2_to_v1,
                )
                signals.append(signal)
                metadata.append(
                    {
                        "sample_id": f"{parameter}:q{quantile_index:0{sample_id_width}d}:{context_id}",
                        "sample_kind": "sweep",
                        "sweep_parameter": parameter,
                        "quantile_index": quantile_index,
                        "quantile_probability": probability_by_index[quantile_index],
                        "context_id": context_id,
                        "phase_index": phase_index,
                        "common_phase_rad": phase,
                        "position_index": position_index,
                        "position_fraction": position,
                        "carrier_index": carrier_index,
                        "carrier_id": carrier.carrier_id,
                        "carrier_record_id": carrier.record_id,
                        **transformed,
                        **physical,
                    }
                )
    bank = np.stack(signals).astype(np.float32)
    if len({row["sample_id"] for row in metadata}) != len(metadata):
        raise ValueError("Sweep sample IDs are not unique")
    expected = len(contexts) * (1 + len(parameter_subset) * len(quantile_indices))
    if bank.shape != (expected, INPUT_LENGTH):
        raise ValueError("Unexpected sweep bank shape")
    return bank, metadata


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.sum(a * b, axis=-1)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if np.ptp(y) <= 1.0e-12:
        return 0.0
    xr = rankdata(x)
    yr = rankdata(y)
    return float(np.corrcoef(xr, yr)[0, 1])


def analyze_embeddings(
    *,
    model_name: str,
    metadata: list[dict[str, Any]],
    embeddings_l2: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if embeddings_l2.shape != (len(metadata), 512):
        raise ValueError("Expected one 512-D embedding per generated signal")
    if probabilities.shape != (len(metadata), len(CLASS_NAMES)):
        raise ValueError("Probability shape mismatch")
    by_context_anchor = {
        row["context_id"]: index
        for index, row in enumerate(metadata)
        if row["sample_kind"] == "anchor"
    }
    per_point: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(metadata):
        anchor_index = by_context_anchor[row["context_id"]]
        point = {
            "model": model_name,
            "sample_id": row["sample_id"],
            "sample_kind": row["sample_kind"],
            "sweep_parameter": row["sweep_parameter"],
            "context_id": row["context_id"],
            "quantile_index": row["quantile_index"],
            "quantile_probability": row["quantile_probability"],
            "cosine_from_context_anchor": float(
                _cosine_distance(embeddings_l2[index], embeddings_l2[anchor_index])
            ),
            **{
                f"probability_{class_name}": float(probabilities[index, class_index])
                for class_index, class_name in enumerate(CLASS_NAMES)
            },
        }
        per_point.append(point)
        if row["sample_kind"] == "sweep":
            grouped.setdefault((row["sweep_parameter"], row["context_id"]), []).append(index)

    active_parameters = tuple(
        parameter
        for parameter in PARAMETERS
        if any(
            row["sample_kind"] == "sweep" and row["sweep_parameter"] == parameter
            for row in metadata
        )
    )
    context_metrics: dict[str, list[dict[str, float]]] = {
        parameter: [] for parameter in active_parameters
    }
    for (parameter, context_id), indices in grouped.items():
        ordered = sorted(indices, key=lambda index: int(metadata[index]["quantile_index"]))
        curve = embeddings_l2[ordered]
        steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        path_length = float(np.sum(steps))
        endpoint_vector = curve[-1] - curve[0]
        endpoint_distance = float(np.linalg.norm(endpoint_vector))
        direction = endpoint_vector / max(endpoint_distance, 1.0e-12)
        projection = (curve - curve[0]) @ direction
        anchor = embeddings_l2[by_context_anchor[context_id]]
        context_metrics[parameter].append(
            {
                "excursion": float(np.max(_cosine_distance(curve, anchor[None, :]))),
                "path_efficiency": endpoint_distance / max(path_length, 1.0e-12),
                "jump_ratio": float(np.max(steps) / max(float(np.median(steps)), 1.0e-12)),
                "monotonicity": abs(_spearman(np.arange(len(curve), dtype=np.float64), projection)),
                **{
                    f"probability_excursion_{class_name}": float(
                        np.ptp(probabilities[ordered, class_index])
                    )
                    for class_index, class_name in enumerate(CLASS_NAMES)
                },
            }
        )

    per_parameter: list[dict[str, Any]] = []
    for parameter in active_parameters:
        sweep_indices = [
            index
            for index, row in enumerate(metadata)
            if row["sample_kind"] == "sweep" and row["sweep_parameter"] == parameter
        ]
        by_quantile: dict[int, list[int]] = {}
        for index in sweep_indices:
            by_quantile.setdefault(int(metadata[index]["quantile_index"]), []).append(index)
        dispersions = []
        for indices in by_quantile.values():
            values = embeddings_l2[indices]
            centroid = np.mean(values, axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1.0e-12)
            dispersions.extend(_cosine_distance(values, centroid[None, :]).tolist())
        rows = context_metrics[parameter]
        output: dict[str, Any] = {
            "model": model_name,
            "sweep_parameter": parameter,
            "contexts": len(rows),
            "nuisance_dispersion_median": float(np.median(dispersions)),
            "nuisance_dispersion_q90": float(np.quantile(dispersions, 0.90)),
        }
        for metric in (
            "excursion",
            "path_efficiency",
            "jump_ratio",
            "monotonicity",
            *(f"probability_excursion_{name}" for name in CLASS_NAMES),
        ):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            output[f"{metric}_median"] = float(np.median(values))
            output[f"{metric}_q10"] = float(np.quantile(values, 0.10))
            output[f"{metric}_q90"] = float(np.quantile(values, 0.90))
        per_parameter.append(output)
    return per_point, per_parameter


def computation_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
