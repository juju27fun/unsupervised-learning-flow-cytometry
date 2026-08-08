from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_yeast_budding_m2_latent_sweep.py"


def _module():
    spec = importlib.util.spec_from_file_location("yeast_budding_m2_sweep_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_uses_approved_method_and_registered_contracts() -> None:
    module = _module()
    assert module.DEFAULT_V1_DATASET.endswith("@v1")
    assert module.DEFAULT_V2_DATASET.endswith("@v2")
    assert module.METHOD_EVIDENCE_ID == "yeast-budding-m2-resnet-stft-latent-sweep-method-r1"
