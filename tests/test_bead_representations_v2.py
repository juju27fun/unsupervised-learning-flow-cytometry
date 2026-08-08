from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_bead_representations_v2.py"
SPEC = importlib.util.spec_from_file_location("evaluate_bead_representations_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source(root: Path, policy: str, seed: int, epoch: int = 25) -> str:
    run_id = f"{policy}-{seed}"
    path = root / run_id
    (path / "checkpoints").mkdir(parents=True)
    (path / "checkpoints/latest.pt").write_bytes(b"checkpoint")
    (path / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "profile": "full",
                "training_mask_policy": policy,
                "seed": seed,
                "epochs": epoch,
            }
        ),
        encoding="utf-8",
    )
    return run_id


def test_checkpoint_map_requires_complete_seed_matrix(tmp_path: Path) -> None:
    ids = [
        _source(tmp_path, policy, seed)
        for policy in MODULE.POLICIES
        for seed in MODULE.SEEDS
    ]
    mapping, epoch = MODULE.checkpoint_map(tmp_path, ids)
    assert len(mapping) == 10
    assert epoch == 25


def test_checkpoint_map_rejects_missing_seed(tmp_path: Path) -> None:
    ids = [
        _source(tmp_path, policy, seed)
        for policy in MODULE.POLICIES
        for seed in MODULE.SEEDS
        if not (policy == "CYCLIC25" and seed == 46)
    ]
    with pytest.raises(ValueError, match="complete paired"):
        MODULE.checkpoint_map(tmp_path, ids)
