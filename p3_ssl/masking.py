from __future__ import annotations

from dataclasses import dataclass

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

    @property
    def starts(self) -> np.ndarray:
        return np.arange(self.n_tokens, dtype=np.int64) * self.patch_stride

    @property
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
