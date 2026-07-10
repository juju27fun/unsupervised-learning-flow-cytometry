from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .decimation import crop_or_pad
from .physics import PHYSICS_PARAM_NAMES, evaluate_physical_latent_space
from .serialization import json_safe


WINDOW_DURATION_MS = 2.048


def _parse_float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _metadata_row_to_physics(row: dict[str, str], window_duration_ms: float) -> np.ndarray:
    if "fD" in row:
        values = [
            _parse_float(row, "A"),
            _parse_float(row, "fD") / window_duration_ms,
            _parse_float(row, "phi"),
            _parse_float(row, "t0"),
            _parse_float(row, "tau") * window_duration_ms,
            _parse_float(row, "snr_db"),
        ]
    else:
        values = [
            _parse_float(row, "A"),
            _parse_float(row, "fDA") / window_duration_ms,
            _parse_float(row, "phiA"),
            _parse_float(row, "t0A"),
            _parse_float(row, "tauA") * window_duration_ms,
            _parse_float(row, "snr_db"),
        ]
    return np.asarray(values, dtype=np.float32)


def load_sweep_physics_by_panel(
    metadata_csv: str | Path,
    window_duration_ms: float = WINDOW_DURATION_MS,
) -> dict[str, np.ndarray]:
    """Load `run_particle_equation_latent_sweeps.py` metadata as P3 physics columns."""
    by_panel: dict[str, list[np.ndarray]] = {}
    with Path(metadata_csv).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            panel = row.get("panel", "")
            if not panel:
                raise ValueError(f"Missing panel in metadata row: {row}")
            by_panel.setdefault(panel, []).append(_metadata_row_to_physics(row, window_duration_ms))
    return {panel: np.stack(values).astype(np.float32) for panel, values in by_panel.items()}


def load_sweep_embeddings_by_panel(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    panels: dict[str, np.ndarray] = {}
    suffix = "_embeddings"
    for key in data.files:
        if key.endswith(suffix):
            panels[key[: -len(suffix)]] = np.asarray(data[key], dtype=np.float32)
    if not panels:
        raise ValueError(f"No '*_embeddings' arrays found in {path}")
    return panels


def load_sweep_signals_by_panel(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    panels: dict[str, np.ndarray] = {}
    suffix = "_signals"
    for key in data.files:
        if key.endswith(suffix):
            signals = np.asarray(data[key], dtype=np.float32)
            panels[key[: -len(suffix)]] = signals.reshape(signals.shape[0], -1)
    if not panels:
        raise ValueError(f"No '*_signals' arrays found in {path}")
    return panels


def _shared_panels(embeddings: dict[str, np.ndarray], physics: dict[str, np.ndarray]) -> list[str]:
    panels = sorted(set(embeddings) & set(physics))
    if not panels:
        raise ValueError("No shared panels between embeddings and physics metadata")
    for panel in panels:
        if embeddings[panel].shape[0] != physics[panel].shape[0]:
            raise ValueError(
                f"Panel {panel} row mismatch: {embeddings[panel].shape[0]} embeddings vs {physics[panel].shape[0]} metadata rows"
            )
    return panels


def evaluate_embedding_panels(
    embeddings_by_panel: dict[str, np.ndarray],
    physics_by_panel: dict[str, np.ndarray],
    k_neighbors: int = 5,
    max_combined_samples: int | None = None,
    seed: int = 123,
    pass_threshold: float = 0.05,
) -> dict[str, Any]:
    panels = _shared_panels(embeddings_by_panel, physics_by_panel)
    per_panel: dict[str, Any] = {}
    scores: list[float] = []
    all_embeddings: list[np.ndarray] = []
    all_physics: list[np.ndarray] = []
    for panel in panels:
        metrics = evaluate_physical_latent_space(
            embeddings_by_panel[panel],
            physics_by_panel[panel],
            k_neighbors=k_neighbors,
            pass_threshold=pass_threshold,
        )
        per_panel[panel] = metrics
        score = metrics.get("physical_score")
        if isinstance(score, (float, int)) and np.isfinite(score):
            scores.append(float(score))
        all_embeddings.append(embeddings_by_panel[panel])
        all_physics.append(physics_by_panel[panel])
    combined_embeddings = np.concatenate(all_embeddings, axis=0)
    combined_physics = np.concatenate(all_physics, axis=0)
    if max_combined_samples is not None and combined_embeddings.shape[0] > max_combined_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(combined_embeddings.shape[0], size=max_combined_samples, replace=False))
        combined_embeddings = combined_embeddings[indices]
        combined_physics = combined_physics[indices]
    combined = evaluate_physical_latent_space(
        combined_embeddings,
        combined_physics,
        k_neighbors=k_neighbors,
        pass_threshold=pass_threshold,
    )
    panel_mean_score = float(np.mean(scores)) if scores else 0.0
    return {
        "panels": panels,
        "combined_samples": int(combined_embeddings.shape[0]),
        "panel_mean_physical_score": panel_mean_score,
        "combined": combined,
        "per_panel": per_panel,
    }


def build_random_embeddings_like(
    embeddings_by_panel: dict[str, np.ndarray],
    dim: int = 128,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        panel: rng.normal(0.0, 1.0, size=(values.shape[0], dim)).astype(np.float32)
        for panel, values in embeddings_by_panel.items()
    }


def evaluate_encoder_on_sweep_directory(
    sweep_dir: str | Path,
    encode_panel,
    model_name: str,
    input_length: int,
    k_neighbors: int = 5,
    max_combined_samples: int | None = 2000,
    seed: int = 123,
    pass_threshold: float = 0.05,
) -> dict[str, Any]:
    """Evaluate a caller-provided encoder on existing sweep signals."""
    root = Path(sweep_dir)
    physics = load_sweep_physics_by_panel(root / "synthetic_metadata.csv")
    signals = load_sweep_signals_by_panel(root / "synthetic_signals_encoded.npz")
    embeddings: dict[str, np.ndarray] = {}
    for panel, values in signals.items():
        prepared = np.stack([crop_or_pad(row, input_length, mode="center") for row in values]).astype(np.float32)
        embeddings[panel] = np.asarray(encode_panel(prepared), dtype=np.float32)
    metrics = evaluate_embedding_panels(
        embeddings,
        physics,
        k_neighbors=k_neighbors,
        max_combined_samples=max_combined_samples,
        seed=seed,
        pass_threshold=pass_threshold,
    )
    return {
        "model": model_name,
        "sweep_dir": str(root),
        "input_length": int(input_length),
        "metrics": metrics,
        "ranking_row": {
            "model": model_name,
            "combined_physical_score": float(metrics["combined"].get("physical_score", 0.0)),
            "panel_mean_physical_score": float(metrics.get("panel_mean_physical_score", 0.0)),
            "physical_validation_pass": bool(metrics["combined"].get("physical_validation_pass", False)),
        },
    }


def evaluate_sweep_directory(
    sweep_dir: str | Path,
    model_names: list[str] | None = None,
    include_raw: bool = True,
    include_random: bool = True,
    k_neighbors: int = 5,
    random_seed: int = 123,
    max_combined_samples: int | None = 2000,
    pass_threshold: float = 0.05,
) -> dict[str, Any]:
    root = Path(sweep_dir)
    physics = load_sweep_physics_by_panel(root / "synthetic_metadata.csv")
    models = model_names or sorted(
        child.name for child in root.iterdir() if child.is_dir() and (child / "embeddings.npz").is_file()
    )
    model_metrics: dict[str, Any] = {}
    reference_embeddings: dict[str, np.ndarray] | None = None

    if include_raw and (root / "synthetic_signals_encoded.npz").is_file():
        raw = load_sweep_signals_by_panel(root / "synthetic_signals_encoded.npz")
        model_metrics["raw_signal"] = evaluate_embedding_panels(
            raw,
            physics,
            k_neighbors=k_neighbors,
            max_combined_samples=max_combined_samples,
            seed=random_seed,
            pass_threshold=pass_threshold,
        )
        reference_embeddings = raw

    for model in models:
        embeddings = load_sweep_embeddings_by_panel(root / model / "embeddings.npz")
        model_metrics[model] = evaluate_embedding_panels(
            embeddings,
            physics,
            k_neighbors=k_neighbors,
            max_combined_samples=max_combined_samples,
            seed=random_seed,
            pass_threshold=pass_threshold,
        )
        if reference_embeddings is None:
            reference_embeddings = embeddings

    if include_random and reference_embeddings is not None:
        random_embeddings = build_random_embeddings_like(reference_embeddings, seed=random_seed)
        model_metrics["random_embedding"] = evaluate_embedding_panels(
            random_embeddings,
            physics,
            k_neighbors=k_neighbors,
            max_combined_samples=max_combined_samples,
            seed=random_seed,
            pass_threshold=pass_threshold,
        )

    ranking = sorted(
        (
            {
                "model": model,
                "combined_physical_score": float(metrics["combined"].get("physical_score", 0.0)),
                "panel_mean_physical_score": float(metrics.get("panel_mean_physical_score", 0.0)),
                "physical_validation_pass": bool(metrics["combined"].get("physical_validation_pass", False)),
            }
            for model, metrics in model_metrics.items()
        ),
        key=lambda row: (row["combined_physical_score"], row["panel_mean_physical_score"]),
        reverse=True,
    )
    return {
        "sweep_dir": str(root),
        "max_combined_samples": max_combined_samples,
        "physics_columns": list(PHYSICS_PARAM_NAMES),
        "models": model_metrics,
        "ranking": ranking,
    }


def merge_reference_and_candidate_rankings(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    model_metrics = dict(reference.get("models", {}))
    model_metrics[str(candidate["model"])] = candidate["metrics"]
    ranking = sorted(
        [*reference.get("ranking", []), candidate["ranking_row"]],
        key=lambda row: (float(row["combined_physical_score"]), float(row["panel_mean_physical_score"])),
        reverse=True,
    )
    return {
        **reference,
        "models": model_metrics,
        "candidate_model": candidate["model"],
        "ranking": ranking,
    }


def write_physical_evaluation_report(metrics: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "physical_metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, sort_keys=True, allow_nan=False))
    lines = ["# Physical Latent Space Evaluation", ""]
    lines.append("| rank | model | combined physical_score | panel mean | pass |")
    lines.append("|---:|---|---:|---:|---|")
    for rank, row in enumerate(metrics["ranking"], start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['combined_physical_score']:.6f} | "
            f"{row['panel_mean_physical_score']:.6f} | {row['physical_validation_pass']} |"
        )
    lines.append("")
    lines.append("Scores use physical parameters from the synthetic sweep metadata; class retrieval is not used as pass/fail evidence.")
    (output / "physical_ranking.md").write_text("\n".join(lines) + "\n")
