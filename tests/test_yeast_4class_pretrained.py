from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_yeast_4class_pretrained.py"
SPEC = importlib.util.spec_from_file_location("benchmark_yeast_4class_pretrained", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_stratified_screening_indices_are_deterministic_and_capped() -> None:
    labels = np.asarray([0] * 9 + [1] * 8 + [2] * 7 + [3] * 6, dtype=np.int64)
    indices = np.arange(labels.size)
    first = MODULE.stratified_screening_indices(labels, indices, max_per_class=4, seed=42)
    second = MODULE.stratified_screening_indices(labels, indices, max_per_class=4, seed=42)
    np.testing.assert_array_equal(first, second)
    assert first.size == 16
    assert np.bincount(labels[first], minlength=4).tolist() == [4, 4, 4, 4]


def test_script_declares_frozen_probe_and_no_sealed_split() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'choices=("moment", "patchtst")' in text
    assert '"transfer": "frozen_encoder_linear_probe"' in text
    assert '"sealed_holdout_accessed": False' in text
    assert "in_session_test" not in text
