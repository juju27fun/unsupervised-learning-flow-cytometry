from __future__ import annotations

import numpy as np
import pytest

from p3_ssl.bead_patching import (
    BeadPatchingConfig,
    build_patching_example,
    serializable_patching_example,
)


def _metadata() -> dict[str, str]:
    return {
        "latent_id": "train-example",
        "view_index": "1",
        "signal_row": "7",
        "duration_ms": "0.7",
        "doppler_khz": "12.0",
        "event_position_fraction": "0.5",
    }


def test_p25_masks_exactly_one_quarter_of_patch_tokens() -> None:
    time = np.arange(4096) / 1_000_000.0
    signal = np.cos(2.0 * np.pi * 12_000.0 * time).astype(np.float32)
    example = build_patching_example(signal, _metadata())

    assert example["n_tokens"] == 256
    assert example["n_masked_tokens"] == 64
    assert example["masked_sample_fraction"] == pytest.approx(0.25)
    masked = np.flatnonzero(example["token_mask"])
    assert np.all(np.diff(masked) >= 2)
    assert float(np.mean(example["signal"])) == pytest.approx(0.0, abs=1e-6)
    assert float(np.std(example["signal"])) == pytest.approx(1.0, abs=1e-6)


def test_p25_public_contract_is_label_free() -> None:
    signal = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    public = serializable_patching_example(
        build_patching_example(signal, _metadata(), BeadPatchingConfig(seed=9))
    )

    assert public["masking"]["policy"] == "P25"
    assert public["masking"]["event_biased_probability"] == 0.0
    assert public["patching"]["patch_duration_us"] == pytest.approx(16.0)
    assert "orientation only" in public["claim_boundary"]
