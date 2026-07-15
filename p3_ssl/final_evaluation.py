from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


FINAL_SPLIT = "in_session_test"
LABEL_FRACTION = 0.10
MINIMUM_EFFECT = 0.03


def prior_final_open(run_root: Path) -> Path | None:
    for manifest in sorted(run_root.glob("*/run.json")):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "complete" and FINAL_SPLIT in payload.get(
            "sealed_splits_used", []
        ):
            return manifest
    return None


def paired_comparison(
    rows: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    differences = []
    direct = []
    for left_row in [row for row in rows if row["method"] == left]:
        candidates = [
            row
            for row in rows
            if row["method"] == right
            and int(row["probe_seed"]) == int(left_row["probe_seed"])
            and (
                right == "handcrafted"
                or int(row["representation_seed"])
                == int(left_row["representation_seed"])
            )
        ]
        if len(candidates) != 1:
            raise ValueError(f"Expected one paired {right} row, found {len(candidates)}")
        right_row = candidates[0]
        left_values = np.asarray(
            left_row["grouped_bootstrap"]["metrics"]["macro_f1"]["replicates"]
        )
        right_values = np.asarray(
            right_row["grouped_bootstrap"]["metrics"]["macro_f1"]["replicates"]
        )
        differences.append(left_values - right_values)
        direct.append(float(left_row["macro_f1"] - right_row["macro_f1"]))
    rng = np.random.default_rng(20260715)
    repeats = len(differences[0])
    hierarchical = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        sampled = rng.integers(0, len(differences), size=len(differences))
        hierarchical[repeat] = np.mean(
            [
                values[rng.integers(0, len(values))]
                for values in (differences[index] for index in sampled)
            ]
        )
    interval = np.quantile(hierarchical, [0.025, 0.975])
    return {
        "comparison": f"{left} - {right}",
        "mean_difference": float(np.mean(direct)),
        "ci_95_low": float(interval[0]),
        "ci_95_high": float(interval[1]),
        "bootstrap_probability_gt_zero": float(np.mean(hierarchical > 0.0)),
        "n_paired_runs": len(direct),
        "n_repeats": repeats,
        "uncertainty_method": (
            "hierarchical paired bootstrap over representation/probe runs and "
            "in-session capture-block bootstrap replicates"
        ),
    }
