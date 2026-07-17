from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_local_spectral_result(
    s1_metrics: dict[str, Any], c1_metrics: dict[str, Any]
) -> dict[str, Any]:
    history = s1_metrics["history"]
    if not history:
        raise ValueError("S1 history is empty")
    controls = s1_metrics["validation_local_spectral_controls"]
    regions = controls["regions"]
    model_mse = float(controls["model_masked_feature_mse"])
    interpolation_mse = float(controls["feature_of_interpolation_masked_feature_mse"])
    c1_health = c1_metrics["validation_embedding_health"]["real"]
    s1_health = s1_metrics["validation_embedding_health"]
    first_loss = float(history[0]["local_spectral_prediction"])
    final_loss = float(history[-1]["local_spectral_prediction"])

    region_rows = {}
    for region in ("all", "event", "boundary", "background"):
        s1_mse = float(regions["model"][region]["mse"])
        baseline_mse = float(regions["feature_of_interpolation"][region]["mse"])
        region_rows[region] = {
            "s1_mse": s1_mse,
            "interpolation_mse": baseline_mse,
            "s1_to_interpolation_ratio": s1_mse / baseline_mse,
            "s1_relative_change_vs_interpolation": s1_mse / baseline_mse - 1.0,
        }

    return {
        "decision": s1_metrics["decision"],
        "gates": dict(s1_metrics["gates"]),
        "training": {
            "first_epoch_prediction_loss": first_loss,
            "final_epoch_prediction_loss": final_loss,
            "relative_reduction": 1.0 - final_loss / first_loss,
            "epochs": len(history),
        },
        "controls": {
            "s1_mse": model_mse,
            "zero_mse": float(controls["zero_masked_feature_mse"]),
            "train_constant_mse": float(controls["train_constant_masked_feature_mse"]),
            "interpolation_mse": interpolation_mse,
            "s1_to_interpolation_ratio": model_mse / interpolation_mse,
            "output_rms_fraction_of_target": float(
                controls["model_output_rms_fraction_of_target"]
            ),
        },
        "regions": region_rows,
        "geometry": {
            "C1": {
                "effective_rank": float(c1_health["effective_rank"]),
                "mean_pairwise_cosine": float(
                    c1_health["mean_off_diagonal_cosine_similarity"]
                ),
            },
            "S1": {
                "effective_rank": float(s1_health["effective_rank"]),
                "mean_pairwise_cosine": float(
                    s1_health["mean_off_diagonal_cosine_similarity"]
                ),
            },
        },
        "interpretation": {
            "optimization_worked": final_loss < first_loss,
            "beats_zero_and_constant": bool(
                s1_metrics["gates"]["beats_zero"]
                and s1_metrics["gates"]["beats_train_constant"]
            ),
            "beats_interpolation": bool(
                s1_metrics["gates"]["beats_feature_of_interpolation"]
            ),
            "geometry_passed": bool(
                s1_metrics["gates"]["effective_rank"]
                and s1_metrics["gates"]["mean_pairwise_cosine"]
            ),
            "utility_authorized": s1_metrics["decision"]
            == "run_development_utility_with_handcrafted_controls",
        },
    }


def audit_registered_dataset(dataset_root: Path, manifest_path: Path) -> dict[str, Any]:
    entries = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path: {relative}")
        path = dataset_root / relative
        if path.stat().st_size != int(entry["size"]) or sha256(path) != entry["sha256"]:
            raise ValueError(f"Dataset manifest mismatch: {relative}")

    with (dataset_root / "events.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    split_key = "development_split"
    split_names = sorted({row[split_key] for row in rows})
    split_counts = {
        split: sum(row[split_key] == split for row in rows) for split in split_names
    }
    unique_counts = {
        field: {
            split: len({row[field] for row in rows if row[split_key] == split})
            for split in split_names
        }
        for field in ("record_id", "capture_block_id")
    }
    overlaps = {}
    for field in ("record_id", "capture_block_id"):
        sets = {
            split: {row[field] for row in rows if row[split_key] == split}
            for split in split_names
        }
        overlaps[field] = {
            f"{left}__{right}": len(sets[left] & sets[right])
            for index, left in enumerate(split_names)
            for right in split_names[index + 1 :]
        }
    if any(count for values in overlaps.values() for count in values.values()):
        raise ValueError("Dataset records or capture blocks overlap across splits")
    return {
        "manifest_sha256": sha256(manifest_path),
        "manifest_entries_verified": len(entries),
        "event_count": len(rows),
        "split_counts": split_counts,
        "unique_counts": unique_counts,
        "cross_split_overlaps": overlaps,
    }
