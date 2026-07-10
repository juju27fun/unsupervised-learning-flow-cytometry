from __future__ import annotations

import copy

import pytest

from p3_ssl.config import validate_ssl_config


def _base_config() -> dict:
    return {
        "data": {
            "input_length_raw": 16384,
            "decimation_factor": 4,
            "input_length_ssl": 4096,
        },
        "patching": {
            "patch_size": 4,
            "patch_stride": 4,
        },
        "model": {
            "max_tokens": 1024,
        },
    }


def test_validate_ssl_config_accepts_full_window_4096() -> None:
    validate_ssl_config(_base_config())


def test_validate_ssl_config_rejects_mismatched_decimation() -> None:
    config = copy.deepcopy(_base_config())
    config["data"]["decimation_factor"] = 8

    with pytest.raises(ValueError, match="decimation_factor"):
        validate_ssl_config(config)


def test_validate_ssl_config_rejects_excess_tokens() -> None:
    config = copy.deepcopy(_base_config())
    config["patching"]["patch_stride"] = 2

    with pytest.raises(ValueError, match="token count"):
        validate_ssl_config(config)
