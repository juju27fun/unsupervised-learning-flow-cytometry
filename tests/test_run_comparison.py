from __future__ import annotations

import json

from p3_ssl.run_comparison import compare_reconstruction_metrics, load_reference_reconstruction_metrics


def test_compare_reconstruction_metrics_flags_regression() -> None:
    reference = {
        "status": "ok",
        "path": "reference.json",
        "metrics": {
            "masked_mse": 1.0,
            "event_region_masked_mse": 2.0,
            "background_region_masked_mse": 1.5,
            "derivative_mse": 0.5,
        },
    }
    current = {
        "masked_mse": 1.1,
        "event_region_masked_mse": 2.7,
        "background_region_masked_mse": 1.6,
        "derivative_mse": 0.55,
    }
    result = compare_reconstruction_metrics(current, reference, max_regression_fraction=0.25)
    assert result["status"] == "ok"
    assert not result["reconstruction_regression_pass"]
    assert result["metrics"]["event_region_masked_mse"]["pass"] is False
    assert result["metrics"]["masked_mse"]["pass"] is True


def test_load_reference_reconstruction_metrics(tmp_path) -> None:
    path = tmp_path / "eval_val"
    path.mkdir()
    (path / "metrics.json").write_text(json.dumps({"masked_mse": 1.0}))
    loaded = load_reference_reconstruction_metrics(tmp_path, "val")
    assert loaded["status"] == "ok"
    assert loaded["metrics"]["masked_mse"] == 1.0
    missing = load_reference_reconstruction_metrics(tmp_path, "test")
    assert missing["status"] == "missing"
