from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_bead_ssl_v2_cross_masks.py"
SPEC = importlib.util.spec_from_file_location("evaluate_bead_ssl_v2_cross_masks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
load_sources = MODULE.load_sources
write_metrics_manifest = MODULE._write_metrics_manifest


def _source(root: Path, run_id: str, policy: str, seed: int, profile: str = "full") -> None:
    path = root / run_id
    (path / "checkpoints").mkdir(parents=True)
    (path / "checkpoints/latest.pt").write_bytes(b"checkpoint")
    (path / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "training_mask_policy": policy,
                "seed": seed,
                "profile": profile,
            }
        ),
        encoding="utf-8",
    )


def test_sources_require_paired_policy_seeds(tmp_path: Path) -> None:
    _source(tmp_path, "p25", "P25", 42)
    _source(tmp_path, "cyclic", "CYCLIC25", 42)
    sources = load_sources(tmp_path, ["p25", "cyclic"])
    assert len(sources) == 2


def test_sources_reject_unpaired_seeds(tmp_path: Path) -> None:
    _source(tmp_path, "p25", "P25", 42)
    _source(tmp_path, "cyclic", "CYCLIC25", 43)
    with pytest.raises(ValueError, match="paired"):
        load_sources(tmp_path, ["p25", "cyclic"])


def test_sources_reject_pilot_runs(tmp_path: Path) -> None:
    _source(tmp_path, "p25", "P25", 42, profile="pilot")
    _source(tmp_path, "cyclic", "CYCLIC25", 42)
    with pytest.raises(ValueError, match="full runs"):
        load_sources(tmp_path, ["p25", "cyclic"])


def test_metrics_manifest_binds_metrics_and_computation(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps({"counts": {"rows": 20}}), encoding="utf-8"
    )
    run = {
        "run_id": "cross-mask",
        "source_runs": ["p25", "cyclic"],
        "datasets": {"simulation": {"id": "synthetic@v5"}},
        "evaluation_policies": ["P25", "CYCLIC25"],
        "checkpoint_policy": "fixed_final",
        "cyclic_evaluation_schedule": "all_unique_passes_per_sample",
        "sealed_splits_used": [],
        "source_sha256": {"entrypoint": "a" * 64},
        "repositories": {"workspace": "revision"},
    }
    result = {"counts": {"rows": 20}}
    write_metrics_manifest(tmp_path, run, result)
    manifest = json.loads(
        (tmp_path / "metrics_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["analysis_run_id"] == "cross-mask"
    assert len(manifest["computation_fingerprint"]) == 64
    assert manifest["metrics"][0]["path"] == "metrics.json"
