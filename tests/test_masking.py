from __future__ import annotations

import numpy as np

from p3_ssl.masking import (
    PatchSpec,
    build_patch_aligned_isolated_masks,
    build_ssl_masks,
    make_time_mask,
    mask_coherence_summary,
    mask_is_event_coherent,
    time_mask_to_token_mask,
    token_mask_to_time_mask,
)


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


def test_mask_coherence_summary_tracks_event_overlap() -> None:
    target = np.zeros(16, dtype=bool)
    hidden = np.zeros(16, dtype=bool)
    event = np.zeros(16, dtype=bool)
    target[4:8] = True
    hidden[3:9] = True
    event[5:7] = True

    summary = mask_coherence_summary(target, hidden, event)

    assert summary["event_count"] == 1
    assert summary["event_points"] == 2
    assert summary["event_target_points"] == 2
    assert summary["event_hidden_points"] == 2
    assert summary["event_target_fraction"] == 1.0
    assert summary["event_hidden_fraction"] == 1.0
    assert summary["target_event_fraction"] == 0.5
    assert summary["fully_hidden_event_count"] == 1


def test_mask_is_event_coherent_rejects_full_and_over_cap_hiding() -> None:
    event = np.zeros(16, dtype=bool)
    event[4:8] = True
    hidden = np.zeros(16, dtype=bool)
    hidden[4:8] = True

    assert not mask_is_event_coherent(hidden, event, avoid_fully_hidden_events=True)
    assert not mask_is_event_coherent(hidden, event, max_event_hidden_fraction=0.75)

    hidden[6:8] = False
    assert not mask_is_event_coherent(hidden, event, max_event_hidden_fraction=0.49)
    assert mask_is_event_coherent(hidden, event, max_event_hidden_fraction=0.5)


def test_build_ssl_masks_respects_event_hidden_cap_when_feasible() -> None:
    spec = PatchSpec(input_length=32, patch_size=4, patch_stride=4)
    signal = np.zeros(32, dtype=np.float32)
    event = np.zeros(32, dtype=bool)
    event[10:22] = True

    masks = build_ssl_masks(
        signal=signal,
        spec=spec,
        rng=np.random.default_rng(0),
        mask_ratio=0.25,
        min_block_length=8,
        max_block_length=8,
        guard_points=0,
        high_derivative_probability=0.0,
        event_mask=event,
        event_biased_probability=1.0,
        avoid_fully_hidden_events=True,
        max_event_hidden_fraction=0.75,
        max_mask_attempts=8,
    )
    summary = mask_coherence_summary(masks["target_time_mask"], masks["token_time_mask"], event)

    assert bool(masks["mask_accepted"])
    assert summary["event_target_points"] > 0
    assert summary["max_event_hidden_fraction"] <= 0.75


def test_patch_aligned_masks_do_not_amplify_hidden_support() -> None:
    spec = PatchSpec(input_length=64, patch_size=4, patch_stride=4)
    signal = np.sin(np.arange(64, dtype=np.float32))
    event = np.zeros(64, dtype=bool)
    event[16:48] = True
    masks = build_patch_aligned_isolated_masks(
        signal,
        spec,
        np.random.default_rng(4),
        mask_ratio=0.25,
        event_mask=event,
        minimum_visible_tokens_between_masks=1,
        max_mask_attempts=4,
    )
    selected = np.flatnonzero(masks["token_mask"])
    assert len(selected) == 4
    assert np.all(np.diff(selected) >= 2)
    np.testing.assert_array_equal(masks["target_time_mask"], masks["token_time_mask"])
    assert masks["target_time_mask"].mean() == 0.25
