from __future__ import annotations

import numpy as np
import pytest
import torch

from p3_ssl.hybrid_training import (
    build_training_stages,
    fixed_ratio_hybrid_batch_sampler,
    hybrid_sampling_weights,
    synthetic_only_physics_params,
)


def test_build_training_stages_pretrain_then_real_adaptation() -> None:
    config = {
        "training": {"epochs": {"smoke": 1, "full": 20}},
        "hybrid_sampling": {
            "synthetic_fraction_pretrain": 0.70,
            "synthetic_fraction_adaptation": 0.30,
        },
        "real_adaptation": {"enabled": True, "epochs": {"smoke": 1, "full": 5}},
    }
    stages = build_training_stages(config, "full")
    assert [stage.name for stage in stages] == ["hybrid_pretrain", "real_adaptation"]
    assert [stage.epochs for stage in stages] == [20, 5]
    assert [stage.synthetic_fraction for stage in stages] == [0.70, 0.30]


def test_hybrid_sampling_weights_match_requested_fraction() -> None:
    kinds = ["synthetic"] * 7 + ["particle"] * 3
    weights = hybrid_sampling_weights(kinds, synthetic_fraction=0.70)
    assert weights is not None
    synthetic_mass = float(np.sum(weights[:7]))
    real_mass = float(np.sum(weights[7:]))
    assert np.isclose(synthetic_mass, 0.70)
    assert np.isclose(real_mass, 0.30)


def test_hybrid_sampling_weights_reject_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        hybrid_sampling_weights(["synthetic", "real"], synthetic_fraction=1.2)


def test_fixed_ratio_hybrid_batch_sampler_matches_each_batch_fraction() -> None:
    kinds = ["synthetic"] * 7 + ["particle"] * 3
    sampler = fixed_ratio_hybrid_batch_sampler(kinds, synthetic_fraction=0.70, batch_size=10, seed=5)
    assert sampler is not None
    batches = list(sampler)

    assert len(batches) == 1
    assert sum(kinds[index] == "synthetic" for index in batches[0]) == 7
    assert sum(kinds[index] != "synthetic" for index in batches[0]) == 3


def test_fixed_ratio_hybrid_batch_sampler_is_deterministic() -> None:
    kinds = ["synthetic"] * 8 + ["particle"] * 8
    first = list(fixed_ratio_hybrid_batch_sampler(kinds, synthetic_fraction=0.25, batch_size=8, seed=11))
    second = list(fixed_ratio_hybrid_batch_sampler(kinds, synthetic_fraction=0.25, batch_size=8, seed=11))

    assert first == second
    for batch in first:
        assert sum(kinds[index] == "synthetic" for index in batch) == 2
        assert sum(kinds[index] != "synthetic" for index in batch) == 6


def test_synthetic_only_physics_params_masks_non_synthetic_rows() -> None:
    params = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    masked = synthetic_only_physics_params(params, ["synthetic", "particle", "real"])

    assert torch.equal(masked[0], params[0])
    assert torch.isnan(masked[1]).all()
    assert torch.isnan(masked[2]).all()
    assert torch.isfinite(params).all()


def test_synthetic_only_physics_params_rejects_mismatched_batch_size() -> None:
    params = torch.zeros((2, 6), dtype=torch.float32)
    with pytest.raises(ValueError, match="source_kinds length"):
        synthetic_only_physics_params(params, ["synthetic"])
