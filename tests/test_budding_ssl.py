from __future__ import annotations

from pathlib import Path

import pytest
import torch

from p3_ssl.budding_ssl import (
    balanced_reconstruction_loss,
    balanced_validation_loss,
    load_budding_ssl_config,
    validation_improved,
)


def test_balanced_reconstruction_loss_reports_raw_regions() -> None:
    target = torch.zeros((1, 1, 8))
    prediction = torch.zeros_like(target)
    prediction[..., :2] = 2.0
    prediction[..., 4:6] = 1.0
    event = torch.tensor([[True, True, False, False, False, False, False, False]])
    background = torch.tensor(
        [[False, False, False, False, True, True, False, False]]
    )

    loss, components = balanced_reconstruction_loss(
        prediction,
        target,
        event,
        background,
        event_weight=0.8,
        background_weight=0.2,
    )

    assert components["event_mse"].item() == 4.0
    assert components["background_mse"].item() == 1.0
    assert components["raw_masked_mse"].item() == 2.5
    assert loss.item() == pytest.approx(3.4)


def test_validation_loss_matches_training_objective_weights() -> None:
    loss = balanced_validation_loss(
        event_mse=4.0,
        background_mse=1.0,
        event_weight=0.8,
        background_weight=0.2,
    )
    assert loss == pytest.approx(3.4)


def test_validation_improvement_requires_strict_minimum_delta() -> None:
    assert validation_improved(0.998, 1.0, min_delta=0.001)
    assert not validation_improved(0.999, 1.0, min_delta=0.001)
    assert not validation_improved(1.001, 1.0, min_delta=0.001)


def test_full_profile_is_bounded_by_frozen_early_stopping_contract() -> None:
    config = load_budding_ssl_config(
        Path(__file__).resolve().parents[1]
        / "configs/yeast_budding_ssl_b0_v1.yaml"
    )
    assert config["training"]["profiles"]["full"]["coverage_cycles"] == 150
    assert config["training"]["early_stopping"] == {
        "metric": "simulation_validation.model.balanced_loss",
        "patience": 10,
        "min_delta": 0.001,
    }
