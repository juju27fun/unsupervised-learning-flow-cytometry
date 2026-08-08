from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_yeast_budding_m2_dense_atlas.py"


def _module():
    spec = importlib.util.spec_from_file_location("dense_atlas_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dense_runner_contract() -> None:
    module = _module()
    assert module.DENSE_QUANTILES == 225
    assert module.DENSE_METHOD_EVIDENCE_ID == "yeast-budding-m2-dense-atlas-method-r1"
    assert module.DEFAULT_V1_DATASET.endswith("@v1")
    assert module.DEFAULT_V2_DATASET.endswith("@v2")
