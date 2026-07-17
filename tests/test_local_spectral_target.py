import math

import pytest
import torch

from p3_ssl.local_spectral_target import (
    LocalSpectralTargetConfig,
    local_spectral_frame_regions,
    local_spectral_frequencies,
    local_spectral_target,
    masked_local_spectral_mse,
)


def _sinusoid(phase: float = 0.0) -> torch.Tensor:
    time = torch.arange(4096, dtype=torch.float32) / 1_000_000.0
    return torch.cos(2.0 * math.pi * 31_250.0 * time + phase)[None, None]


def test_local_spectral_target_has_frozen_shape_frequency_and_peak() -> None:
    target = local_spectral_target(_sinusoid())
    frequencies = local_spectral_frequencies()
    assert target.shape == (1, 240, 24)
    assert target.dtype == torch.float32
    assert frequencies.shape == (24,)
    assert float(frequencies[0]) == 7_812.5
    assert float(frequencies[-1]) == 97_656.25
    assert int(target.mean(dim=1).argmax(dim=1)[0]) == 6


def test_local_spectral_target_is_phase_invariant() -> None:
    first = local_spectral_target(_sinusoid(0.0))
    second = local_spectral_target(_sinusoid(1.234))
    relative_error = torch.mean((first - second).square()) / torch.mean(first.square())
    assert float(relative_error) <= 0.002


def test_masked_local_spectral_mse_counts_feature_dimensions() -> None:
    config = LocalSpectralTargetConfig()
    prediction = torch.ones(2, 256, 24)
    target = torch.zeros(2, 240, 24)
    token_mask = torch.zeros(2, 256, dtype=torch.bool)
    token_mask[0, 10] = True
    token_mask[1, 200] = True
    assert float(masked_local_spectral_mse(prediction, target, token_mask, config)) == 1.0


def test_target_is_independent_of_mask_and_rejects_empty_mask() -> None:
    signals = torch.randn(2, 1, 4096)
    target = local_spectral_target(signals)
    first_mask = torch.rand(2, 256) < 0.25
    second_mask = torch.rand(2, 256) < 0.25
    assert torch.equal(target, local_spectral_target(signals))
    prediction = torch.zeros(2, 256, 24)
    assert torch.isfinite(masked_local_spectral_mse(prediction, target, first_mask))
    assert torch.isfinite(masked_local_spectral_mse(prediction, target, second_mask))
    with pytest.raises(ValueError, match="at least one"):
        masked_local_spectral_mse(prediction, target, torch.zeros_like(first_mask))


def test_local_spectral_regions_partition_frames() -> None:
    events = torch.zeros(2, 4096, dtype=torch.bool)
    events[0, 1024:3072] = True
    regions = local_spectral_frame_regions(events)
    assert all(mask.shape == (2, 240) for mask in regions.values())
    total = sum(mask.to(torch.int64) for mask in regions.values())
    assert torch.equal(total, torch.ones_like(total))
    assert regions["event"][0].any()
    assert regions["boundary"][0].any()
    assert regions["background"][1].all()
