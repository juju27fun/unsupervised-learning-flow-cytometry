from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/encode_yeast_two_particle_sweep.py"


def load_module():
    spec = importlib.util.spec_from_file_location("yeast_two_particle", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase4_metadata_contract_preserves_physics_and_nuisances(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "metadata.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(module.REQUIRED_METADATA))
        writer.writeheader()
        for index in range(2):
            writer.writerow({name: (f"sample-{index}" if name == "sample_id" else "budding" if name == "carrier_class" else f"noise-{index}" if name == "noise_id" else 0.1) for name in module.REQUIRED_METADATA})
    rows = module.read_metadata(path, 2)
    assert [row["sample_id"] for row in rows] == ["sample-0", "sample-1"]
    assert set(module.ABSOLUTE_PARAMETERS) == {"log_A_A", "fD_A", "log_tau_A", "snr_db"}
    assert "delta_phi" in module.RELATIVE_PARAMETERS
