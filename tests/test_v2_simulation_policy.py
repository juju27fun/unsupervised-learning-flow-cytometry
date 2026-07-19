from __future__ import annotations

from pathlib import Path

import pytest

from p3_ssl.config import (
    ACTIVE_SIMULATION_DATASET,
    load_config,
    validate_active_simulation_dataset,
)


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "name",
    (
        "bead_ssl_p25_v2.yaml",
        "yeast_ssl_rebuild_v2.yaml",
        "yeast_ssl_mask_ablation_v2.yaml",
        "yeast_ssl_followup_week2_v2.yaml",
    ),
)
def test_v2_configs_resolve_to_active_simulation(name: str) -> None:
    config = load_config(CONFIG_ROOT / name)
    validate_active_simulation_dataset(config)
    assert config["study"]["simulation_dataset"] == ACTIVE_SIMULATION_DATASET


def test_historical_v1_config_is_rejected_for_active_training() -> None:
    config = load_config(CONFIG_ROOT / "yeast_ssl_rebuild_v1.yaml")
    with pytest.raises(ValueError, match="historical/reference only"):
        validate_active_simulation_dataset(config)
