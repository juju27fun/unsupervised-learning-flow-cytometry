from __future__ import annotations

from pathlib import Path

import numpy as np

from p3_ssl.collapse_utility import (
    apply_utility_gate,
    fit_train_only_pca,
    paired_group_macro_f1_difference,
)
from p3_ssl.config import load_config, validate_collapse_utility_config


CONFIG_PATH = Path(__file__).parents[1] / "configs/yeast_ssl_collapse_utility_v1.yaml"


def test_collapse_utility_config_is_frozen() -> None:
    validate_collapse_utility_config(load_config(CONFIG_PATH))


def test_pca_is_fitted_on_train_only_and_has_frozen_width() -> None:
    train = np.arange(80, dtype=np.float32).reshape(10, 8)
    validation = np.full((4, 8), 1000.0, dtype=np.float32)
    transformed, metadata = fit_train_only_pca(
        train,
        np.concatenate([train, validation]),
        {
            "components": 4,
            "fit_split": "development_train",
            "svd_solver": "randomized",
            "random_state": 42,
            "whiten": False,
        },
    )
    assert transformed.shape == (14, 4)
    assert metadata["components"] == 4
    assert np.allclose(train.mean(axis=0), np.arange(36, 44, dtype=np.float32))


def test_paired_group_difference_and_gate_require_absolute_gain_and_interval() -> None:
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    groups = np.asarray(["a", "b", "a", "b", "c", "d", "c", "d"])
    candidate = [labels.copy(), labels.copy()]
    baseline = [np.zeros_like(labels), np.zeros_like(labels)]
    comparison = paired_group_macro_f1_difference(
        labels,
        groups,
        candidate,
        baseline,
        class_count=2,
        repeats=100,
        seed=42,
        interval_level=0.95,
    )
    result = apply_utility_gate(
        {"raw": comparison},
        {
            "primary_gate": {
                "minimum_gain": 0.03,
                "success_action": "promote",
                "failure_action": "reject",
            }
        },
        scientific_decision_allowed=True,
    )
    assert comparison["gain"] > 0.03
    assert result["passed"] is True
    assert result["decision"] == "promote"


def test_smoke_never_emits_scientific_pass() -> None:
    comparison = {
        "candidate_mean_macro_f1": 1.0,
        "baseline_mean_macro_f1": 0.0,
        "gain": 1.0,
        "paired_interval": [0.5, 1.0],
    }
    result = apply_utility_gate(
        {"raw": comparison},
        {
            "primary_gate": {
                "minimum_gain": 0.03,
                "success_action": "promote",
                "failure_action": "reject",
            }
        },
        scientific_decision_allowed=False,
    )
    assert result["passed"] is False
    assert result["decision"] == "smoke_only_no_scientific_decision"
