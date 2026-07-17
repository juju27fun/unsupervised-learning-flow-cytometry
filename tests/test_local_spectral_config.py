from copy import deepcopy

import pytest

from p3_ssl.config import load_config, validate_local_spectral_study_config


def test_local_spectral_config_is_single_target_and_c1_matched() -> None:
    config = load_config("configs/yeast_ssl_local_spectral_v1.yaml")
    validate_local_spectral_study_config(config)
    assert config["target"]["feature_count"] == 24
    assert config["training"]["control_cell"] == "C1"
    assert config["training"]["vicreg_global_weight"] == 1.0


def test_local_spectral_config_rejects_target_or_vicreg_sweep() -> None:
    config = load_config("configs/yeast_ssl_local_spectral_v1.yaml")
    changed = deepcopy(config)
    changed["target"]["window_samples"] = 128
    with pytest.raises(ValueError, match="target differs"):
        validate_local_spectral_study_config(changed)
    changed = deepcopy(config)
    changed["training"]["vicreg_global_weight"] = 0.1
    with pytest.raises(ValueError, match="VICReg weight"):
        validate_local_spectral_study_config(changed)
