from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_RECONSTRUCTION_KEYS = (
    "masked_mse",
    "event_region_masked_mse",
    "background_region_masked_mse",
    "derivative_mse",
)


def load_reference_reconstruction_metrics(reference_root: str | Path, split: str) -> dict[str, Any]:
    path = Path(reference_root) / f"eval_{split}" / "metrics.json"
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    return {"status": "ok", "path": str(path), "metrics": json.loads(path.read_text())}


def compare_reconstruction_metrics(
    current: dict[str, Any],
    reference: dict[str, Any],
    max_regression_fraction: float = 0.25,
    keys: tuple[str, ...] = DEFAULT_RECONSTRUCTION_KEYS,
) -> dict[str, Any]:
    if reference.get("status") != "ok":
        return {"status": "not_run", "reason": "missing_reference", "reference": reference}
    ref_metrics = reference.get("metrics", {})
    rows: dict[str, Any] = {}
    pass_flags: list[bool] = []
    for key in keys:
        current_value = current.get(key)
        reference_value = ref_metrics.get(key)
        if current_value is None or reference_value is None:
            rows[key] = {"status": "missing", "current": current_value, "reference": reference_value}
            continue
        current_f = float(current_value)
        reference_f = float(reference_value)
        delta = current_f - reference_f
        relative_delta = delta / reference_f if reference_f != 0.0 else None
        passed = relative_delta is None or relative_delta <= max_regression_fraction
        rows[key] = {
            "status": "ok",
            "current": current_f,
            "reference": reference_f,
            "delta": delta,
            "relative_delta": relative_delta,
            "pass": bool(passed),
        }
        pass_flags.append(bool(passed))
    return {
        "status": "ok",
        "max_regression_fraction": float(max_regression_fraction),
        "reference_path": reference.get("path"),
        "reconstruction_regression_pass": bool(pass_flags and all(pass_flags)),
        "metrics": rows,
    }
