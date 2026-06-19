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
) -> dict[str, np.ndarray]:
    """Build consistent time, hidden, and token masks for one signal."""
    anchors = None
    if rng.random() < high_derivative_probability:
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
    return {
        "target_time_mask": target_time_mask,
        "hidden_time_mask": hidden_time_mask,
        "token_mask": token_mask,
        "blocks": np.asarray(blocks, dtype=np.int64),
    }

