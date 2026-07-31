from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class PatchSpec:
    input_length: int
    patch_size: int
    patch_stride: int

    def __post_init__(self) -> None:
        if self.input_length <= 0:
            raise ValueError("input_length must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if self.patch_size > self.input_length:
            raise ValueError("patch_size cannot exceed input_length")

    @property
    def n_tokens(self) -> int:
        return 1 + (self.input_length - self.patch_size) // self.patch_stride

    @cached_property
    def starts(self) -> np.ndarray:
        return np.arange(self.n_tokens, dtype=np.int64) * self.patch_stride

    @cached_property
    def spans(self) -> np.ndarray:
        starts = self.starts
        return np.stack([starts, starts + self.patch_size], axis=1)


def make_time_mask(
    input_length: int,
    blocks: list[tuple[int, int]],
    guard_points: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create target and hidden masks from time blocks.

    The target mask marks samples used in the reconstruction loss. The hidden
    mask includes the guard band and marks samples that must not be visible.
    """
    target = np.zeros(input_length, dtype=bool)
    hidden = np.zeros(input_length, dtype=bool)
    for start, end in blocks:
        s = max(0, int(start))
        e = min(input_length, int(end))
        if e <= s:
            continue
        target[s:e] = True
        hs = max(0, s - guard_points)
        he = min(input_length, e + guard_points)
        hidden[hs:he] = True
    return target, hidden


def mask_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous true spans as half-open intervals."""
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 1:
        raise ValueError("mask must be 1D")
    active = np.flatnonzero(arr)
    if active.size == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = int(active[0])
    prev = int(active[0])
    for idx_raw in active[1:]:
        idx = int(idx_raw)
        if idx != prev + 1:
            spans.append((start, prev + 1))
            start = idx
        prev = idx
    spans.append((start, prev + 1))
    return spans


def mask_coherence_summary(
    target_time_mask: np.ndarray,
    hidden_time_mask: np.ndarray,
    event_mask: np.ndarray | None,
) -> dict[str, float | int]:
    """Summarize how a sampled reconstruction mask overlaps labeled events."""
    target = np.asarray(target_time_mask, dtype=bool)
    hidden = np.asarray(hidden_time_mask, dtype=bool)
    if target.ndim != 1 or hidden.ndim != 1 or target.shape != hidden.shape:
        raise ValueError("target_time_mask and hidden_time_mask must be matching 1D masks")
    event = np.zeros_like(target, dtype=bool) if event_mask is None else np.asarray(event_mask, dtype=bool)
    if event.ndim != 1 or event.shape != target.shape:
        raise ValueError("event_mask must be 1D and match target_time_mask")

    event_points = int(event.sum())
    target_points = int(target.sum())
    hidden_points = int(hidden.sum())
    event_target_points = int((event & target).sum())
    event_hidden_points = int((event & hidden).sum())
    background_target_points = int((target & ~event).sum())
    spans = mask_spans(event)
    event_hidden_fracs = [
        float(hidden[start:end].sum()) / float(end - start)
        for start, end in spans
        if end > start
    ]
    event_target_fracs = [
        float(target[start:end].sum()) / float(end - start)
        for start, end in spans
        if end > start
    ]
    fully_hidden = sum(1 for value in event_hidden_fracs if value >= 1.0)
    fully_targeted = sum(1 for value in event_target_fracs if value >= 1.0)

    return {
        "event_count": len(spans),
        "event_points": event_points,
        "target_points": target_points,
        "hidden_points": hidden_points,
        "background_target_points": background_target_points,
        "event_target_points": event_target_points,
        "event_hidden_points": event_hidden_points,
        "event_target_fraction": float(event_target_points) / float(event_points) if event_points else 0.0,
        "event_hidden_fraction": float(event_hidden_points) / float(event_points) if event_points else 0.0,
        "target_event_fraction": float(event_target_points) / float(target_points) if target_points else 0.0,
        "fully_hidden_event_count": fully_hidden,
        "fully_targeted_event_count": fully_targeted,
        "max_event_hidden_fraction": max(event_hidden_fracs, default=0.0),
        "max_event_target_fraction": max(event_target_fracs, default=0.0),
    }


def mask_is_event_coherent(
    hidden_time_mask: np.ndarray,
    event_mask: np.ndarray | None,
    avoid_fully_hidden_events: bool = False,
    max_event_hidden_fraction: float | None = None,
) -> bool:
    """Return whether hidden samples leave enough labeled event context visible."""
    if event_mask is None:
        return True
    event = np.asarray(event_mask, dtype=bool)
    if event.sum() == 0:
        return True
    hidden = np.asarray(hidden_time_mask, dtype=bool)
    if hidden.ndim != 1 or event.ndim != 1 or hidden.shape != event.shape:
        raise ValueError("hidden_time_mask and event_mask must be matching 1D masks")
    if max_event_hidden_fraction is not None and not 0.0 <= max_event_hidden_fraction <= 1.0:
        raise ValueError("max_event_hidden_fraction must be in [0, 1]")
    for start, end in mask_spans(event):
        frac = float(hidden[start:end].sum()) / float(end - start)
        if avoid_fully_hidden_events and frac >= 1.0:
            return False
        if max_event_hidden_fraction is not None and frac > max_event_hidden_fraction:
            return False
    return True


def time_mask_to_token_mask(time_mask: np.ndarray, spec: PatchSpec) -> np.ndarray:
    """Mask every token whose patch intersects `time_mask`."""
    mask = np.asarray(time_mask, dtype=bool)
    if mask.ndim != 1 or mask.size != spec.input_length:
        raise ValueError("time_mask must be 1D and match spec.input_length")
    token_mask = np.zeros(spec.n_tokens, dtype=bool)
    for i, (start, end) in enumerate(spec.spans):
        token_mask[i] = bool(mask[start:end].any())
    return token_mask


def token_mask_to_time_mask(token_mask: np.ndarray, spec: PatchSpec) -> np.ndarray:
    """Expand a token mask back to covered time samples."""
    tokens = np.asarray(token_mask, dtype=bool)
    if tokens.ndim != 1 or tokens.size != spec.n_tokens:
        raise ValueError("token_mask must be 1D and match spec.n_tokens")
    time_mask = np.zeros(spec.input_length, dtype=bool)
    for enabled, (start, end) in zip(tokens, spec.spans):
        if enabled:
            time_mask[start:end] = True
    return time_mask


def sample_mask_blocks(
    input_length: int,
    mask_ratio: float,
    min_block_length: int,
    max_block_length: int,
    rng: np.random.Generator,
    anchor_candidates: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Sample time blocks until approximately `mask_ratio` samples are targeted."""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in (0, 1)")
    if min_block_length <= 0 or max_block_length < min_block_length:
        raise ValueError("invalid block length range")
    target = max(1, int(round(input_length * mask_ratio)))
    covered = np.zeros(input_length, dtype=bool)
    blocks: list[tuple[int, int]] = []
    attempts = 0
    while int(covered.sum()) < target and attempts < 1000:
        attempts += 1
        length = int(rng.integers(min_block_length, max_block_length + 1))
        length = min(length, input_length)
        if anchor_candidates is not None and anchor_candidates.size > 0:
            center = int(rng.choice(anchor_candidates))
            start = center - length // 2
        else:
            start = int(rng.integers(0, max(1, input_length - length + 1)))
        start = max(0, min(start, input_length - length))
        end = start + length
        blocks.append((start, end))
        covered[start:end] = True
    return blocks


def high_derivative_candidates(signal: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    """Return time indices with high absolute derivative."""
    x = np.asarray(signal, dtype=np.float32)
    if x.size < 2:
        return np.array([], dtype=np.int64)
    diff = np.abs(np.diff(x, prepend=x[0]))
    threshold = float(np.quantile(diff, quantile))
    return np.flatnonzero(diff >= threshold).astype(np.int64)


def build_ssl_masks(
    signal: np.ndarray,
    spec: PatchSpec,
    rng: np.random.Generator,
    mask_ratio: float = 0.25,
    min_block_length: int = 24,
    max_block_length: int = 128,
    guard_points: int = 8,
    high_derivative_probability: float = 0.25,
    event_mask: np.ndarray | None = None,
    event_biased_probability: float = 0.0,
    avoid_fully_hidden_events: bool = False,
    max_event_hidden_fraction: float | None = None,
    max_mask_attempts: int = 1,
) -> dict[str, np.ndarray]:
    """Build consistent time, hidden, and token masks for one signal."""
    if not 0.0 <= high_derivative_probability <= 1.0:
        raise ValueError("high_derivative_probability must be in [0, 1]")
    if not 0.0 <= event_biased_probability <= 1.0:
        raise ValueError("event_biased_probability must be in [0, 1]")
    if max_mask_attempts <= 0:
        raise ValueError("max_mask_attempts must be positive")
    event = None
    event_anchors = None
    if event_mask is not None:
        event = np.asarray(event_mask, dtype=bool)
        if event.ndim != 1 or event.size != spec.input_length:
            raise ValueError("event_mask must be 1D and match spec.input_length")
        event_anchors = np.flatnonzero(event).astype(np.int64)

    def draw_once() -> dict[str, np.ndarray]:
        anchors = None
        if event_anchors is not None and event_anchors.size > 0 and rng.random() < event_biased_probability:
            anchors = event_anchors
        elif rng.random() < high_derivative_probability:
            anchors = high_derivative_candidates(signal)
        blocks = sample_mask_blocks(
            input_length=spec.input_length,
            mask_ratio=mask_ratio,
            min_block_length=min_block_length,
            max_block_length=max_block_length,
            rng=rng,
            anchor_candidates=anchors,
        )
        target_time_mask, hidden_time_mask = make_time_mask(
            spec.input_length,
            blocks,
            guard_points=guard_points,
        )
        token_mask = time_mask_to_token_mask(hidden_time_mask, spec)
        token_time_mask = token_mask_to_time_mask(token_mask, spec)
        return {
            "target_time_mask": target_time_mask,
            "hidden_time_mask": hidden_time_mask,
            "token_mask": token_mask,
            "token_time_mask": token_time_mask,
            "blocks": np.asarray(blocks, dtype=np.int64),
        }

    best: dict[str, np.ndarray] | None = None
    best_score: tuple[int, float] | None = None
    accepted = False
    attempts = 0
    for attempts in range(1, max_mask_attempts + 1):
        candidate = draw_once()
        summary = mask_coherence_summary(candidate["target_time_mask"], candidate["token_time_mask"], event)
        score = (
            int(summary["fully_hidden_event_count"]),
            float(summary["max_event_hidden_fraction"]),
        )
        if best is None or best_score is None or score < best_score:
            best = candidate
            best_score = score
        if mask_is_event_coherent(
            candidate["token_time_mask"],
            event,
            avoid_fully_hidden_events=avoid_fully_hidden_events,
            max_event_hidden_fraction=max_event_hidden_fraction,
        ):
            best = candidate
            accepted = True
            break
    if best is None:
        raise RuntimeError("failed to sample an SSL mask")
    best["mask_attempts"] = np.asarray(attempts, dtype=np.int64)
    best["mask_accepted"] = np.asarray(accepted, dtype=bool)
    return best


def build_patch_aligned_isolated_masks(
    signal: np.ndarray,
    spec: PatchSpec,
    rng: np.random.Generator,
    *,
    mask_ratio: float,
    event_mask: np.ndarray | None = None,
    event_biased_probability: float = 0.0,
    high_derivative_probability: float = 0.0,
    minimum_visible_tokens_between_masks: int = 1,
    avoid_fully_hidden_events: bool = False,
    max_event_hidden_fraction: float | None = None,
    max_mask_attempts: int = 1,
) -> dict[str, np.ndarray]:
    """Mask isolated complete patches so target and hidden support are identical."""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in (0, 1)")
    if not 0.0 <= event_biased_probability <= 1.0:
        raise ValueError("event_biased_probability must be in [0, 1]")
    if not 0.0 <= high_derivative_probability <= 1.0:
        raise ValueError("high_derivative_probability must be in [0, 1]")
    if minimum_visible_tokens_between_masks < 0:
        raise ValueError("minimum_visible_tokens_between_masks must be non-negative")
    if max_mask_attempts <= 0:
        raise ValueError("max_mask_attempts must be positive")
    if spec.patch_stride != spec.patch_size:
        raise ValueError("patch-aligned isolated masking requires non-overlapping patches")
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or signal.size != spec.input_length:
        raise ValueError("signal must be 1D and match spec.input_length")
    event = (
        np.zeros(spec.input_length, dtype=bool)
        if event_mask is None
        else np.asarray(event_mask, dtype=bool)
    )
    if event.ndim != 1 or event.size != spec.input_length:
        raise ValueError("event_mask must be 1D and match spec.input_length")

    event_tokens = np.asarray(
        [bool(event[start:end].any()) for start, end in spec.spans], dtype=bool
    )
    derivative_points = high_derivative_candidates(signal)
    derivative_mask = np.zeros(spec.input_length, dtype=bool)
    derivative_mask[derivative_points] = True
    derivative_tokens = np.asarray(
        [bool(derivative_mask[start:end].any()) for start, end in spec.spans], dtype=bool
    )
    requested_tokens = max(1, int(round(spec.n_tokens * mask_ratio)))

    def draw_once() -> dict[str, np.ndarray]:
        selected = np.zeros(spec.n_tokens, dtype=bool)
        available = np.ones(spec.n_tokens, dtype=bool)
        for _ in range(requested_tokens):
            candidates = np.flatnonzero(available)
            if candidates.size == 0:
                break
            if rng.random() < event_biased_probability:
                preferred = candidates[event_tokens[candidates]]
                if preferred.size:
                    candidates = preferred
            elif rng.random() < high_derivative_probability:
                preferred = candidates[derivative_tokens[candidates]]
                if preferred.size:
                    candidates = preferred
            token = int(rng.choice(candidates))
            selected[token] = True
            start = max(0, token - minimum_visible_tokens_between_masks)
            end = min(spec.n_tokens, token + minimum_visible_tokens_between_masks + 1)
            available[start:end] = False
        target = token_mask_to_time_mask(selected, spec)
        return {
            "target_time_mask": target,
            "hidden_time_mask": target.copy(),
            "token_mask": selected,
            "token_time_mask": target.copy(),
            "blocks": np.asarray(mask_spans(target), dtype=np.int64),
        }

    best: dict[str, np.ndarray] | None = None
    best_score: tuple[int, float] | None = None
    accepted = False
    attempts = 0
    for attempts in range(1, max_mask_attempts + 1):
        candidate = draw_once()
        summary = mask_coherence_summary(candidate["target_time_mask"], candidate["token_time_mask"], event)
        score = (
            int(summary["fully_hidden_event_count"]),
            float(summary["max_event_hidden_fraction"]),
        )
        if best is None or best_score is None or score < best_score:
            best = candidate
            best_score = score
        if mask_is_event_coherent(
            candidate["token_time_mask"],
            event,
            avoid_fully_hidden_events=avoid_fully_hidden_events,
            max_event_hidden_fraction=max_event_hidden_fraction,
        ):
            best = candidate
            accepted = True
            break
    if best is None:
        raise RuntimeError("failed to sample a patch-aligned SSL mask")
    best["mask_attempts"] = np.asarray(attempts, dtype=np.int64)
    best["mask_accepted"] = np.asarray(accepted, dtype=bool)
    return best


def build_event_complete_balanced_masks(
    signal: np.ndarray,
    spec: PatchSpec,
    rng: np.random.Generator,
    *,
    event_mask: np.ndarray,
    context_tokens: int = 1,
) -> dict[str, np.ndarray]:
    """Hide the complete event plus context and equally many background tokens.

    Balancing is defined on unique model tokens. Overlapping proposal windows
    therefore cannot inflate the apparent coverage without changing the union
    that is actually hidden from the Transformer.
    """
    if spec.patch_stride != spec.patch_size:
        raise ValueError("event-complete masking requires non-overlapping model tokens")
    if context_tokens < 0:
        raise ValueError("context_tokens must be non-negative")
    signal = np.asarray(signal, dtype=np.float32)
    event = np.asarray(event_mask, dtype=bool)
    if signal.ndim != 1 or signal.size != spec.input_length:
        raise ValueError("signal must be 1D and match spec.input_length")
    if event.ndim != 1 or event.size != spec.input_length:
        raise ValueError("event_mask must be 1D and match spec.input_length")
    if not event.any():
        raise ValueError("event-complete masking requires a non-empty event")

    intersecting = np.flatnonzero(
        [bool(event[start:end].any()) for start, end in spec.spans]
    )
    event_start = max(0, int(intersecting[0]) - context_tokens)
    event_end = min(spec.n_tokens, int(intersecting[-1]) + context_tokens + 1)
    event_tokens = np.zeros(spec.n_tokens, dtype=bool)
    event_tokens[event_start:event_end] = True
    event_count = int(event_tokens.sum())

    background_candidates = np.flatnonzero(~event_tokens)
    if background_candidates.size < event_count:
        raise ValueError(
            "Not enough background tokens to balance the event-complete mask"
        )
    background_selected = np.sort(
        rng.choice(background_candidates, size=event_count, replace=False)
    )
    background_tokens = np.zeros(spec.n_tokens, dtype=bool)
    background_tokens[background_selected] = True
    token_mask = event_tokens | background_tokens
    event_time_mask = token_mask_to_time_mask(event_tokens, spec)
    background_time_mask = token_mask_to_time_mask(background_tokens, spec)
    target = event_time_mask | background_time_mask
    if not np.all(target[event]):
        raise RuntimeError("event-complete mask failed to cover the full event support")
    return {
        "target_time_mask": target,
        "hidden_time_mask": target.copy(),
        "token_mask": token_mask,
        "token_time_mask": target.copy(),
        "event_target_time_mask": event_time_mask,
        "background_target_time_mask": background_time_mask,
        "event_token_mask": event_tokens,
        "background_token_mask": background_tokens,
        "blocks": np.asarray(mask_spans(target), dtype=np.int64),
        "event_blocks": np.asarray(mask_spans(event_time_mask), dtype=np.int64),
        "background_blocks": np.asarray(
            mask_spans(background_time_mask), dtype=np.int64
        ),
        "mask_attempts": np.asarray(1, dtype=np.int64),
        "mask_accepted": np.asarray(True, dtype=bool),
    }


def _window_union(
    window_indices: np.ndarray,
    spec: PatchSpec,
) -> np.ndarray:
    selected = np.asarray(window_indices, dtype=np.int64).reshape(-1)
    result = np.zeros(spec.input_length, dtype=bool)
    for index in selected:
        if index < 0 or index >= spec.n_tokens:
            raise ValueError(f"window index outside patch grid: {int(index)}")
        start, end = spec.spans[int(index)]
        result[start:end] = True
    return result


def _windows_overlap(
    first_indices: np.ndarray,
    second_index: int,
    spec: PatchSpec,
) -> bool:
    second_start, second_end = spec.spans[int(second_index)]
    for first_index in np.asarray(first_indices, dtype=np.int64).reshape(-1):
        first_start, first_end = spec.spans[int(first_index)]
        if max(int(first_start), int(second_start)) < min(
            int(first_end), int(second_end)
        ):
            return True
    return False


def _schedule_nonoverlapping_event_windows(
    window_indices: np.ndarray,
    *,
    windows_per_pass: int,
    pass_count: int,
    spec: PatchSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    """Place every event window while keeping each pass mutually disjoint."""
    windows = np.asarray(window_indices, dtype=np.int64).reshape(-1)
    if windows.size > pass_count * windows_per_pass:
        raise ValueError("not enough event-window slots for complete coverage")
    lane_count = int(np.ceil(spec.patch_size / float(spec.patch_stride)))
    lanes = [
        windows[windows % lane_count == lane]
        for lane in range(lane_count)
        if np.any(windows % lane_count == lane)
    ]
    if any(lane.size < windows_per_pass for lane in lanes):
        raise ValueError(
            "not enough mutually disjoint event windows for one pass"
        )
    required_passes = sum(
        int(np.ceil(lane.size / float(windows_per_pass)))
        for lane in lanes
    )
    if pass_count < required_passes:
        raise ValueError("not enough passes for disjoint lane scheduling")

    passes: list[np.ndarray] = []
    for lane in lanes:
        ordered = np.asarray(rng.permutation(lane), dtype=np.int64)
        for start in range(0, ordered.size, windows_per_pass):
            selected = ordered[start : start + windows_per_pass].tolist()
            if len(selected) < windows_per_pass:
                remaining = [
                    int(window)
                    for window in rng.permutation(lane)
                    if int(window) not in selected
                ]
                selected.extend(
                    remaining[: windows_per_pass - len(selected)]
                )
            passes.append(np.asarray(selected, dtype=np.int64))
    while len(passes) < pass_count:
        passes.append(
            np.asarray(
                rng.choice(
                    lanes[len(passes) % len(lanes)],
                    size=windows_per_pass,
                    replace=False,
                ),
                dtype=np.int64,
            )
        )
    scheduled = np.stack(passes)
    return scheduled[rng.permutation(scheduled.shape[0])]


def apply_temporal_mask(
    signal: np.ndarray,
    target_time_mask: np.ndarray,
    *,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace reconstruction targets and return their explicit visibility.

    The returned visibility channel is one for observed samples and zero for
    reconstruction targets. This allows target windows to be offset from the
    model-token grid without exposing their original values in neighbouring
    input tokens.
    """
    values = np.asarray(signal, dtype=np.float32)
    target = np.asarray(target_time_mask, dtype=bool)
    if values.ndim != 1 or target.ndim != 1 or values.shape != target.shape:
        raise ValueError("signal and target_time_mask must be matching 1D arrays")
    corrupted = values.copy()
    corrupted[target] = np.float32(fill_value)
    visibility = (~target).astype(np.float32)
    return corrupted, visibility


def build_balanced_event_mask_cycle(
    event_mask: np.ndarray,
    target_spec: PatchSpec,
    rng: np.random.Generator,
    *,
    event_windows_per_pass: int = 3,
    background_windows_per_pass: int = 3,
    require_context_each_side: bool = True,
) -> dict[str, np.ndarray]:
    """Build a complete event-window cycle with balanced background targets.

    Every target window intersecting the event appears at least once. The
    background is deliberately under-sampled to the same number of window
    selections as the event group. The half-stride catalogue may contain
    overlapping candidates, but every window selected within one pass is
    mutually disjoint; overlapping candidates are presented on different
    passes.
    """
    event = np.asarray(event_mask, dtype=bool)
    if event.ndim != 1 or event.size != target_spec.input_length:
        raise ValueError("event_mask must be 1D and match target_spec.input_length")
    if not event.any():
        raise ValueError("mask cycle requires a non-empty event")
    if event_windows_per_pass <= 0 or background_windows_per_pass <= 0:
        raise ValueError("windows per pass must be positive")
    if event_windows_per_pass != background_windows_per_pass:
        raise ValueError(
            "event and background window counts must match for the balanced cycle"
        )

    intersects_event = np.asarray(
        [bool(event[start:end].any()) for start, end in target_spec.spans],
        dtype=bool,
    )
    event_windows = np.flatnonzero(intersects_event)
    lane_count = int(
        np.ceil(target_spec.patch_size / float(target_spec.patch_stride))
    )
    if event_windows.size:
        expansion_order = np.argsort(
            np.minimum(
                np.abs(np.arange(target_spec.n_tokens) - event_windows[0]),
                np.abs(np.arange(target_spec.n_tokens) - event_windows[-1]),
            ),
            kind="stable",
        )
        for candidate in expansion_order:
            lane = int(candidate) % lane_count
            if (
                np.sum(event_windows % lane_count == lane)
                >= event_windows_per_pass
            ):
                continue
            intersects_event[int(candidate)] = True
            event_windows = np.flatnonzero(intersects_event)
            if all(
                np.sum(event_windows % lane_count == lane)
                >= event_windows_per_pass
                for lane in range(lane_count)
            ):
                break
    background_candidates = np.flatnonzero(~intersects_event)
    if event_windows.size == 0 or background_candidates.size == 0:
        raise ValueError("cycle requires both event and background target windows")

    pass_count = sum(
        int(
            np.ceil(
                np.sum(event_windows % lane_count == lane)
                / float(event_windows_per_pass)
            )
        )
        for lane in range(lane_count)
    )

    event_points = np.flatnonzero(event)
    event_start = int(event_points[0])
    event_end = int(event_points[-1]) + 1
    spans = target_spec.spans
    left_candidates = background_candidates[spans[background_candidates, 1] <= event_start]
    right_candidates = background_candidates[spans[background_candidates, 0] >= event_end]
    if require_context_each_side and (
        left_candidates.size == 0 or right_candidates.size == 0
    ):
        raise ValueError("event does not leave a context window on both sides")
    pass_event_windows = _schedule_nonoverlapping_event_windows(
        event_windows,
        windows_per_pass=event_windows_per_pass,
        pass_count=pass_count,
        spec=target_spec,
        rng=rng,
    )

    pass_background_windows = np.full(
        (pass_count, background_windows_per_pass),
        -1,
        dtype=np.int64,
    )
    used_background: set[int] = set()
    required_context: dict[int, int] = {}
    selected_context: list[int] = []
    context_orders = (
        left_candidates[
            np.argsort(spans[left_candidates, 1], kind="stable")[::-1]
        ],
        right_candidates[
            np.argsort(spans[right_candidates, 0], kind="stable")
        ],
    )
    for candidates in context_orders:
        chosen_pair: tuple[int, int] | None = None
        for context_window in candidates:
            compatible = [
                pass_index
                for pass_index in range(pass_count)
                if pass_index not in required_context
                and not _windows_overlap(
                    pass_event_windows[pass_index],
                    int(context_window),
                    target_spec,
                )
            ]
            if compatible:
                chosen_pair = (compatible[0], int(context_window))
                break
        if chosen_pair is None:
            if require_context_each_side:
                raise ValueError(
                    "could not place required context without target overlap"
                )
            selected_context.append(-1)
            continue
        required_context[chosen_pair[0]] = chosen_pair[1]
        selected_context.append(chosen_pair[1])
    left_context, right_context = selected_context
    for pass_index in range(pass_count):
        required = required_context.get(pass_index, -1)
        lane_order = (
            [required % lane_count]
            if required >= 0
            else list(rng.permutation(lane_count))
        )
        selected: np.ndarray | None = None
        for lane in lane_order:
            lane_candidates = background_candidates[
                background_candidates % lane_count == lane
            ]
            candidate_spans = spans[lane_candidates]
            event_spans = spans[pass_event_windows[pass_index]]
            overlaps_event = np.any(
                (
                    candidate_spans[:, None, 0]
                    < event_spans[None, :, 1]
                )
                & (
                    event_spans[None, :, 0]
                    < candidate_spans[:, None, 1]
                ),
                axis=1,
            )
            eligible = lane_candidates[~overlaps_event]
            if required >= 0 and required not in eligible:
                continue
            if eligible.size < background_windows_per_pass:
                continue
            preferred = eligible[
                ~np.isin(eligible, np.fromiter(used_background, dtype=np.int64))
                & (eligible != required)
            ]
            draw_count = background_windows_per_pass - int(required >= 0)
            pool = (
                preferred
                if preferred.size >= draw_count
                else eligible[eligible != required]
            )
            if pool.size < draw_count:
                continue
            chosen = np.asarray(
                rng.choice(pool, size=draw_count, replace=False),
                dtype=np.int64,
            )
            selected = (
                np.concatenate(
                    [np.asarray([required], dtype=np.int64), chosen]
                )
                if required >= 0
                else chosen
            )
            break
        if selected is None:
            raise ValueError("not enough disjoint background windows for cycle")
        pass_background_windows[pass_index] = selected
        used_background.update(int(value) for value in selected)

    event_target_masks = np.stack(
        [_window_union(indices, target_spec) for indices in pass_event_windows]
    )
    background_target_masks = np.stack(
        [
            _window_union(indices, target_spec)
            for indices in pass_background_windows
        ]
    )
    if np.any(event_target_masks & background_target_masks):
        raise RuntimeError("event and background target unions overlap within a pass")
    target_masks = event_target_masks | background_target_masks

    covered: set[int] = set()
    cumulative_window_coverage = []
    for indices in pass_event_windows:
        covered.update(int(index) for index in indices)
        cumulative_window_coverage.append(
            len(covered.intersection(int(index) for index in event_windows))
            / float(event_windows.size)
        )
    cumulative_event_union = np.logical_or.accumulate(event_target_masks, axis=0)
    if not np.all(cumulative_event_union[-1][event]):
        raise RuntimeError("complete cycle failed to cover the event support")

    return {
        "event_window_indices": event_windows.astype(np.int64),
        "background_candidate_indices": background_candidates.astype(np.int64),
        "pass_event_window_indices": pass_event_windows.astype(np.int64),
        "pass_background_window_indices": pass_background_windows.astype(np.int64),
        "event_target_time_masks": event_target_masks,
        "background_target_time_masks": background_target_masks,
        "target_time_masks": target_masks,
        "visibility_time_masks": (~target_masks).astype(np.float32),
        "cumulative_event_time_masks": cumulative_event_union,
        "cumulative_event_window_coverage": np.asarray(
            cumulative_window_coverage,
            dtype=np.float32,
        ),
        "context_window_indices": np.asarray(
            [left_context, right_context],
            dtype=np.int64,
        ),
        "pass_count": np.asarray(pass_count, dtype=np.int64),
        "event_windows_per_pass": np.asarray(
            event_windows_per_pass,
            dtype=np.int64,
        ),
        "background_windows_per_pass": np.asarray(
            background_windows_per_pass,
            dtype=np.int64,
        ),
    }
