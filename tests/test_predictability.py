from __future__ import annotations

import numpy as np
import pytest

from p3_ssl.predictability import (
    harmonic_regression_prediction,
    masked_mse_numpy,
    sample_region_block_mask,
)


def test_region_block_mask_stays_inside_regions_when_possible() -> None:
    event = np.zeros(64, dtype=bool)
    event[16:48] = True
    rng = np.random.default_rng(3)
    event_target = sample_region_block_mask(event, 8, "event", rng)
    background_target = sample_region_block_mask(event, 8, "background", rng)
    assert event_target.sum() == 8
    assert np.all(event[event_target])
    assert np.all(~event[background_target])


def test_harmonic_regression_recovers_known_sinusoidal_gap() -> None:
    sampling_frequency = 1_000_000.0
    index = np.arange(4096)
    signal = 0.8 * np.sin(2.0 * np.pi * 15_000.0 * index / sampling_frequency + 0.7)
    mask = np.zeros(signal.size, dtype=bool)
    mask[1900:2100] = True
    prediction = harmonic_regression_prediction(
        signal,
        mask,
        sampling_frequency_hz=sampling_frequency,
        frequency_bins=501,
    )
    zero_mse = masked_mse_numpy(np.zeros_like(signal), signal, mask)
    harmonic_mse = masked_mse_numpy(prediction, signal, mask)
    assert harmonic_mse < zero_mse * 1e-8


def test_harmonic_regression_handles_multiple_gaps() -> None:
    sampling_frequency = 1_000_000.0
    index = np.arange(1024)
    signal = np.sin(2.0 * np.pi * 10_000.0 * index / sampling_frequency + 0.2)
    mask = np.zeros(signal.size, dtype=bool)
    mask[200:216] = True
    mask[700:716] = True
    prediction = harmonic_regression_prediction(
        signal,
        mask,
        sampling_frequency_hz=sampling_frequency,
        frequency_bins=251,
    )
    assert masked_mse_numpy(prediction, signal, mask) < 1e-10


def test_region_block_mask_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        sample_region_block_mask(
            np.zeros(16, dtype=bool), 4, "unsupported", np.random.default_rng(1)
        )
