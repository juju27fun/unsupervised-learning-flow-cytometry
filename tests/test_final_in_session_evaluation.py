from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_yeast_final_in_session import _paired_comparison, _prior_final_open


def _metric(
    method: str,
    probe_seed: int,
    value: float,
    representation_seed: int | None = None,
) -> dict:
    return {
        "method": method,
        "representation_seed": representation_seed,
        "probe_seed": probe_seed,
        "macro_f1": value,
        "grouped_bootstrap": {
            "metrics": {
                "macro_f1": {"replicates": [value - 0.01, value + 0.01]}
            }
        },
    }


def test_final_paired_comparison_preserves_probe_and_representation_pairing() -> None:
    rows = []
    for representation_seed in (42, 43):
        for probe_seed in (42, 43):
            rows.append(_metric("A4", probe_seed, 0.5, representation_seed))
            rows.append(_metric("A3", probe_seed, 0.4, representation_seed))
    for probe_seed in (42, 43):
        rows.append(_metric("handcrafted", probe_seed, 0.45))
    primary = _paired_comparison(rows, "A4", "handcrafted")
    ablation = _paired_comparison(rows, "A4", "A3")
    assert primary["n_paired_runs"] == 4
    assert primary["mean_difference"] == pytest.approx(0.05)
    assert ablation["mean_difference"] == pytest.approx(0.1)


def test_prior_final_open_detects_only_completed_in_session_use(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "run.json").write_text(
        json.dumps({"status": "failed", "sealed_splits_used": ["in_session_test"]}),
        encoding="utf-8",
    )
    assert _prior_final_open(tmp_path) is None
    complete = tmp_path / "complete"
    complete.mkdir()
    manifest = complete / "run.json"
    manifest.write_text(
        json.dumps({"status": "complete", "sealed_splits_used": ["in_session_test"]}),
        encoding="utf-8",
    )
    assert _prior_final_open(tmp_path) == manifest
