from __future__ import annotations

import numpy as np
import torch

from p3_ssl.embedding import (
    balanced_event_indices,
    label_to_interval,
    pool_token_embeddings,
    token_indices_for_interval,
)
from p3_ssl.masking import PatchSpec


def test_label_to_interval_clips_to_signal() -> None:
    assert label_to_interval(0.5, 0.25, 2048) == (768, 1280)
    assert label_to_interval(0.0, 0.2, 100) == (0, 10)
    assert label_to_interval(1.0, 0.2, 100) == (90, 100)


def test_token_indices_for_interval_disjoint_patch() -> None:
    spec = PatchSpec(input_length=32, patch_size=4, patch_stride=4)
    idx = token_indices_for_interval(6, 10, spec)
    np.testing.assert_array_equal(idx, np.array([1, 2]))


def test_token_indices_for_interval_overlap_patch() -> None:
    spec = PatchSpec(input_length=32, patch_size=8, patch_stride=4)
    idx = token_indices_for_interval(6, 10, spec)
    np.testing.assert_array_equal(idx, np.array([0, 1, 2]))


def test_pool_token_embeddings() -> None:
    tokens = torch.arange(20, dtype=torch.float32).view(5, 4)
    pooled = pool_token_embeddings(tokens, [1, 3])
    expected = (tokens[1] + tokens[3]) / 2.0
    assert torch.allclose(pooled, expected)


def test_balanced_event_indices_caps_per_class() -> None:
    class_ids = np.array([0] * 10 + [1] * 3 + [2] * 7)
    selected = balanced_event_indices(class_ids, max_per_class=4, seed=1)
    counts = {c: int(np.sum(class_ids[selected] == c)) for c in sorted(set(class_ids.tolist()))}
    assert counts == {0: 4, 1: 3, 2: 4}
