from __future__ import annotations

import numpy as np
import pytest

from p3_ssl.masking import (
    PatchSpec,
    apply_temporal_mask,
    build_balanced_event_mask_cycle,
    build_event_complete_balanced_masks,
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


def test_event_complete_balanced_masks_cover_event_context_and_balance() -> None:
    spec = PatchSpec(input_length=256, patch_size=16, patch_stride=16)
    signal = np.zeros(256, dtype=np.float32)
    event = np.zeros(256, dtype=bool)
    event[73:119] = True
    result = build_event_complete_balanced_masks(
        signal,
        spec,
        np.random.default_rng(42),
        event_mask=event,
        context_tokens=1,
    )
    event_tokens = result["event_token_mask"]
    background_tokens = result["background_token_mask"]
    assert event_tokens.sum() == background_tokens.sum()
    assert not np.any(event_tokens & background_tokens)
    assert np.all(result["event_target_time_mask"][event])
    assert np.flatnonzero(event_tokens).tolist() == [3, 4, 5, 6, 7, 8]
    assert result["event_target_time_mask"][48:144].all()
    assert not result["event_target_time_mask"][:48].any()
    assert not result["event_target_time_mask"][144:].any()


def test_balanced_event_mask_cycle_covers_every_event_window_once_or_more() -> None:
    spec = PatchSpec(input_length=256, patch_size=16, patch_stride=8)
    event = np.zeros(256, dtype=bool)
    event[73:119] = True
    result = build_balanced_event_mask_cycle(
        event,
        spec,
        np.random.default_rng(42),
        event_windows_per_pass=3,
        background_windows_per_pass=3,
    )

    pass_event = result["pass_event_window_indices"]
    pass_background = result["pass_background_window_indices"]
    expected_event = set(result["event_window_indices"].tolist())
    selected_event = set(pass_event.reshape(-1).tolist())

    assert pass_event.shape == pass_background.shape
    assert pass_event.shape[1] == 3
    assert expected_event <= selected_event
    assert result["cumulative_event_window_coverage"][-1] == pytest.approx(1.0)
    assert np.all(result["cumulative_event_time_masks"][-1][event])
    assert not np.any(
        result["event_target_time_masks"]
        & result["background_target_time_masks"]
    )
    assert np.all(result["target_time_masks"].sum(axis=1) == 6 * 16)
    assert np.all(result["visibility_time_masks"] == ~result["target_time_masks"])
    for event_indices, background_indices in zip(
        pass_event,
        pass_background,
        strict=True,
    ):
        selected = np.concatenate([event_indices, background_indices])
        selected_spans = spec.spans[selected]
        for first in range(selected.size):
            for second in range(first + 1, selected.size):
                first_start, first_end = selected_spans[first]
                second_start, second_end = selected_spans[second]
                assert min(first_end, second_end) <= max(
                    first_start,
                    second_start,
                )

    left_context, right_context = result["context_window_indices"]
    assert left_context in set(pass_background.reshape(-1).tolist())
    assert right_context in set(pass_background.reshape(-1).tolist())
    left_start, left_end = spec.spans[left_context]
    right_start, right_end = spec.spans[right_context]
    assert left_end <= np.flatnonzero(event)[0]
    assert right_start > np.flatnonzero(event)[-1]


def test_balanced_event_mask_cycle_is_deterministic_and_seeded() -> None:
    spec = PatchSpec(input_length=256, patch_size=16, patch_stride=8)
    event = np.zeros(256, dtype=bool)
    event[64:160] = True

    first = build_balanced_event_mask_cycle(
        event,
        spec,
        np.random.default_rng(7),
    )
    repeated = build_balanced_event_mask_cycle(
        event,
        spec,
        np.random.default_rng(7),
    )
    changed = build_balanced_event_mask_cycle(
        event,
        spec,
        np.random.default_rng(8),
    )

    np.testing.assert_array_equal(
        first["pass_event_window_indices"],
        repeated["pass_event_window_indices"],
    )
    np.testing.assert_array_equal(
        first["pass_background_window_indices"],
        repeated["pass_background_window_indices"],
    )
    assert not np.array_equal(
        first["pass_event_window_indices"],
        changed["pass_event_window_indices"],
    )


def test_apply_temporal_mask_hides_every_target_value() -> None:
    signal = np.linspace(-2.0, 2.0, 32, dtype=np.float32)
    target = np.zeros(32, dtype=bool)
    target[7:19] = True

    corrupted, visibility = apply_temporal_mask(signal, target)

    np.testing.assert_array_equal(corrupted[~target], signal[~target])
    assert np.all(corrupted[target] == 0.0)
    assert np.all(visibility[target] == 0.0)
    assert np.all(visibility[~target] == 1.0)


def test_balanced_event_mask_cycle_rejects_missing_required_context() -> None:
    spec = PatchSpec(input_length=64, patch_size=16, patch_stride=8)
    event = np.zeros(64, dtype=bool)
    event[:24] = True

    with pytest.raises(ValueError, match="context window on both sides"):
        build_balanced_event_mask_cycle(
            event,
            spec,
            np.random.default_rng(3),
            require_context_each_side=True,
        )


def test_balanced_event_mask_cycle_allows_a_missing_edge_context() -> None:
    spec = PatchSpec(input_length=128, patch_size=16, patch_stride=8)
    event = np.zeros(128, dtype=bool)
    event[:40] = True

    result = build_balanced_event_mask_cycle(
        event,
        spec,
        np.random.default_rng(3),
        require_context_each_side=False,
    )

    assert result["context_window_indices"][0] == -1
    assert result["context_window_indices"][1] >= 0
    assert np.all(result["target_time_masks"].sum(axis=1) == 6 * 16)
    assert not np.any(
        result["event_target_time_masks"]
        & result["background_target_time_masks"]
    )
    assert result["cumulative_event_window_coverage"][-1] == pytest.approx(1.0)
