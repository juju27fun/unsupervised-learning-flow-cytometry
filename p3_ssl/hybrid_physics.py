from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .decimation import normalize_signal
from .physics import PHYSICS_PARAM_NAMES


HYBRID_COLUMNS = (
    "split",
    "id",
    "signal_path",
    "label_path",
    "source_kind",
    "source",
    "scenario",
    "particle_count",
    "physics_param_source",
    *PHYSICS_PARAM_NAMES,
)


def _load_particle_equation_module() -> Any:
    from . import particle_equation_sweeps

    return particle_equation_sweeps


def _write_yolo_labels(path: Path, labels: list[tuple[float, float]], class_id: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for center, width in labels:
        center = float(np.clip(center, 0.0, 1.0))
        width = float(np.clip(width, 1.0e-4, 1.0))
        lines.append(f"{class_id} {center:.8f} {width:.8f}")
    path.write_text("\n".join(lines) + "\n")


def _split_for_index(index: int, n: int) -> str:
    train_cut = int(round(0.70 * n))
    val_cut = int(round(0.85 * n))
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def _snr_noise(rng: np.random.Generator, clean: np.ndarray, snr_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(clean))))
    if rms <= 0.0:
        return np.zeros_like(clean, dtype=np.float32)
    noise_std = rms / (10.0 ** (float(snr_db) / 20.0))
    return rng.normal(0.0, noise_std, size=clean.shape).astype(np.float32)


def _parse_float(raw: str | None, default: float = np.nan) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _sample_key_from_path(raw: str) -> str:
    return Path(raw).stem.split("__ann")[0]


def load_particles2snr_event_estimates(event_manifest: str | Path | None) -> dict[str, dict[str, str]]:
    """Load reliable sample-level physics estimates from particles2SNR event manifests."""
    if event_manifest is None:
        return {}
    path = Path(event_manifest)
    if not path.is_file():
        return {}
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            key = row.get("source_filename") or row.get("source_path") or row.get("output_filename") or row.get("output_path") or ""
            if not key:
                continue
            grouped.setdefault(_sample_key_from_path(key), []).append(dict(row))
    estimates: dict[str, dict[str, str]] = {}
    for sample_id, rows in grouped.items():
        best = max(rows, key=lambda row: _parse_float(row.get("snr_db"), default=-np.inf))
        estimates[sample_id] = {
            "particle_count": str(len(rows)),
            "physics_param_source": "particles2snr_event_manifest",
            "fD_khz": f"{_parse_float(best.get('frequency')) / 1000.0:.8g}",
            "t0_fraction": f"{_parse_float(best.get('center')):.8g}",
            "tau_ms": f"{_parse_float(best.get('passage_time_ms') or best.get('width_ms')):.8g}",
            "snr_db": f"{_parse_float(best.get('snr_db')):.8g}",
        }
    return estimates


def generate_synthetic_manifest(
    output_dir: str | Path,
    n_samples: int = 100,
    input_length: int = 16384,
    seed: int = 42,
    normalization: str = "none",
    include_two_particle: bool = True,
) -> Path:
    """Generate standardized P3 synthetic signals and a physics manifest."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if input_length <= 0:
        raise ValueError("input_length must be positive")
    helpers = _load_particle_equation_module()
    rng = np.random.default_rng(seed)
    output = Path(output_dir)
    signal_dir = output / "synthetic_signals"
    label_dir = output / "synthetic_labels"
    signal_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "synthetic_manifest.csv"

    rows: list[dict[str, str]] = []
    t = np.linspace(0.0, 1.0, input_length, dtype=np.float32)
    duration_ms = float(helpers.WINDOW_DURATION_MS)
    fwhm_to_sigma = float(helpers.GAUSSIAN_FWHM_TO_SIGMA)
    for index in range(n_samples):
        two_particle = include_two_particle and (index % 5 == 4)
        scenario = "two_particles" if two_particle else "single_particle"
        amplitude = float(rng.uniform(0.10, 3.56))
        fD_khz = float(rng.uniform(8.00, 37.60))
        phi = float(rng.uniform(0.0, 2.0 * np.pi))
        t0 = float(rng.uniform(0.20, 0.80))
        tau_ms = float(rng.uniform(0.33, 1.10))
        snr_db = float(rng.uniform(5.0, 30.0))
        tau_fraction = (tau_ms / fwhm_to_sigma) / duration_ms
        fD_cycles = fD_khz * duration_ms

        clean = helpers.particle_wave(
            t,
            np.asarray([amplitude], dtype=np.float32),
            np.asarray([fD_cycles], dtype=np.float32),
            np.asarray([phi], dtype=np.float32),
            np.asarray([t0], dtype=np.float32),
            np.asarray([tau_fraction], dtype=np.float32),
        )[0]
        particle_count = 1
        labels = [(t0, tau_ms / duration_ms)]
        if two_particle:
            particle_count = 2
            second_amp = amplitude * float(rng.uniform(0.45, 1.35))
            second_fD = fD_cycles + float(rng.uniform(-4.0, 4.0)) * duration_ms
            second_phi = phi + float(rng.uniform(-np.pi / 3.0, np.pi / 3.0))
            second_t0 = float(np.clip(t0 + rng.uniform(-1.5, 1.5) * tau_fraction, 0.10, 0.90))
            second_tau = tau_fraction * float(rng.uniform(0.75, 1.35))
            labels.append((second_t0, second_tau * fwhm_to_sigma))
            clean = clean + helpers.particle_wave(
                t,
                np.asarray([second_amp], dtype=np.float32),
                np.asarray([second_fD], dtype=np.float32),
                np.asarray([second_phi], dtype=np.float32),
                np.asarray([second_t0], dtype=np.float32),
                np.asarray([second_tau], dtype=np.float32),
            )[0]

        signal = clean + _snr_noise(rng, clean, snr_db)
        signal = normalize_signal(signal, mode=normalization)
        sample_id = f"synthetic_{scenario}_{index:06d}"
        signal_path = signal_dir / f"{sample_id}.npy"
        label_path = label_dir / f"{sample_id}.txt"
        np.save(signal_path, signal.astype(np.float32))
        _write_yolo_labels(label_path, labels, class_id=0)
        rows.append(
            {
                "split": _split_for_index(index, n_samples),
                "id": sample_id,
                "signal_path": str(signal_path),
                "label_path": str(label_path),
                "source_kind": "synthetic",
                "source": "synthetic",
                "scenario": scenario,
                "particle_count": str(particle_count),
                "physics_param_source": "synthetic_internal",
                "A": f"{amplitude:.8g}",
                "fD_khz": f"{fD_khz:.8g}",
                "phi_rad": f"{phi:.8g}",
                "t0_fraction": f"{t0:.8g}",
                "tau_ms": f"{tau_ms:.8g}",
                "snr_db": f"{snr_db:.8g}",
            }
        )

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(HYBRID_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def build_hybrid_manifest(
    synthetic_manifest: str | Path,
    real_manifest: str | Path | None,
    output_path: str | Path,
    max_real_rows: int | None = None,
    particles2snr_event_manifest: str | Path | None = None,
) -> Path:
    """Merge synthetic physics rows with real SSL rows, preserving extra fields."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    with Path(synthetic_manifest).open("r", newline="") as f:
        rows.extend(dict(row) for row in csv.DictReader(f))

    real_estimates = load_particles2snr_event_estimates(particles2snr_event_manifest)

    if real_manifest is not None and Path(real_manifest).exists():
        raw_real_rows: list[dict[str, str]] = []
        with Path(real_manifest).open("r", newline="") as f:
            for row in csv.DictReader(f):
                raw_real_rows.append(dict(row))
        if max_real_rows is not None and len(raw_real_rows) > max_real_rows:
            by_split: dict[str, list[dict[str, str]]] = {}
            for row in raw_real_rows:
                by_split.setdefault(row.get("split", ""), []).append(row)
            splits = sorted(by_split)
            quota = max(1, max_real_rows // max(len(splits), 1))
            selected: list[dict[str, str]] = []
            for split in splits:
                selected.extend(by_split[split][:quota])
            remainder = max_real_rows - len(selected)
            if remainder > 0:
                already = {id(row) for row in selected}
                for row in raw_real_rows:
                    if id(row) not in already:
                        selected.append(row)
                        if len(selected) >= max_real_rows:
                            break
            raw_real_rows = selected[:max_real_rows]
        for row in raw_real_rows:
            sample_id = row.get("id") or row.get("sample_id", "")
            signal_path = row.get("signal_path") or row.get("source_path") or row.get("feature_path") or ""
            estimate = real_estimates.get(sample_id) or real_estimates.get(_sample_key_from_path(signal_path)) or {}
            merged = {key: "" for key in HYBRID_COLUMNS}
            merged.update(
                {
                    "split": row.get("split", ""),
                    "id": sample_id,
                    "signal_path": signal_path,
                    "label_path": row.get("label_path", ""),
                    "source_kind": row.get("source_kind", "real"),
                    "source": "real",
                    "scenario": row.get("source_kind", "real"),
                    "particle_count": row.get("n_labels", ""),
                }
            )
            merged.update({key: value for key, value in estimate.items() if key in HYBRID_COLUMNS})
            rows.append(merged)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(HYBRID_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in HYBRID_COLUMNS})
    return output


def load_physics_params_from_manifest(path: str | Path) -> np.ndarray:
    values: list[list[float]] = []
    with Path(path).open("r", newline="") as f:
        for row in csv.DictReader(f):
            parsed: list[float] = []
            for key in PHYSICS_PARAM_NAMES:
                try:
                    parsed.append(float(row.get(key, "")))
                except (TypeError, ValueError):
                    parsed.append(np.nan)
            values.append(parsed)
    return np.asarray(values, dtype=np.float32)


def summarize_hybrid_manifest(path: str | Path) -> dict[str, Any]:
    """Summarize source mix and finite physics-parameter coverage for a hybrid manifest."""
    summary: dict[str, Any] = {
        "total_rows": 0,
        "by_split": {},
        "by_source": {},
        "by_source_kind": {},
        "by_physics_param_source": {},
        "rows_with_any_physics_param": 0,
        "rows_with_all_physics_params": 0,
        "physics_param_coverage": {name: 0 for name in PHYSICS_PARAM_NAMES},
    }
    with Path(path).open("r", newline="") as f:
        for row in csv.DictReader(f):
            summary["total_rows"] += 1
            for field, key in (
                ("split", "by_split"),
                ("source", "by_source"),
                ("source_kind", "by_source_kind"),
                ("physics_param_source", "by_physics_param_source"),
            ):
                value = row.get(field) or ""
                summary[key][value] = summary[key].get(value, 0) + 1
            finite_flags: list[bool] = []
            for name in PHYSICS_PARAM_NAMES:
                try:
                    value = float(row.get(name, ""))
                except (TypeError, ValueError):
                    value = np.nan
                finite = bool(np.isfinite(value))
                finite_flags.append(finite)
                if finite:
                    summary["physics_param_coverage"][name] += 1
            if any(finite_flags):
                summary["rows_with_any_physics_param"] += 1
            if all(finite_flags):
                summary["rows_with_all_physics_params"] += 1
    return summary
