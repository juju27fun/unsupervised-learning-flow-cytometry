from __future__ import annotations

import numpy as np

from p3_ssl.masking import PatchSpec, make_time_mask, time_mask_to_token_mask, token_mask_to_time_mask


def test_patch_spec_token_count() -> None:
    spec = PatchSpec(input_length=16, patch_size=4, patch_stride=4)
    assert spec.n_tokens == 4
    np.testing.assert_array_equal(spec.starts, np.array([0, 4, 8, 12]))


def test_time_mask_to_token_mask_disjoint() -> None:
    spec = PatchSpec(input_length=16, patch_size=4, patch_stride=4)
    target, hidden = make_time_mask(16, [(5, 7)], guard_points=0)
    assert target.sum() == 2
    token_mask = time_mask_to_token_mask(hidden, spec)
    np.testing.assert_array_equal(token_mask, np.array([False, True, False, False]))


def test_guard_band_expands_hidden_tokens() -> None:
    spec = PatchSpec(input_length=16, patch_size=4, patch_stride=4)
    _, hidden = make_time_mask(16, [(5, 7)], guard_points=3)
    token_mask = time_mask_to_token_mask(hidden, spec)
    np.testing.assert_array_equal(token_mask, np.array([True, True, True, False]))


def test_token_mask_to_time_mask() -> None:
    spec = PatchSpec(input_length=16, patch_size=4, patch_stride=4)
    time_mask = token_mask_to_time_mask(np.array([False, True, False, True]), spec)
    assert time_mask[:4].sum() == 0
    assert time_mask[4:8].all()
    assert time_mask[12:16].all()
