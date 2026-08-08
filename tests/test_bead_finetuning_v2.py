from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_bead_finetuning_v2.py"
SPEC = importlib.util.spec_from_file_location("run_bead_finetuning_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_source_run_requires_matching_policy_seed_and_full_profile(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "checkpoints").mkdir(parents=True)
    (root / "checkpoints/latest.pt").write_bytes(b"checkpoint")
    (root / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "profile": "full",
                "training_mask_policy": "P25",
                "seed": 42,
                "epochs": 25,
            }
        ),
        encoding="utf-8",
    )
    checkpoint, run = MODULE._source_run(tmp_path, "source", "P25", 42)
    assert checkpoint.name == "latest.pt"
    assert run["epochs"] == 25
    with pytest.raises(ValueError, match="mismatch"):
        MODULE._source_run(tmp_path, "source", "CYCLIC25", 42)
