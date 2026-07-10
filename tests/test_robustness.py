from __future__ import annotations

import torch

from p3_ssl.robustness import cosine_distances_rows, perturb_signal_batch


def test_perturb_signal_batch_variants_preserve_shape() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 1, 128)
    for perturbation in ("noise_0p10", "scale_1p25", "shift_8", "center_mask_64"):
        y = perturb_signal_batch(x, perturbation)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
    masked = perturb_signal_batch(x, "center_mask_64")
    start = 128 // 2 - 64 // 2
    assert torch.count_nonzero(masked[..., start : start + 64]) == 0


def test_cosine_distances_rows() -> None:
    a = torch.eye(3)
    b = torch.eye(3)
    dist = cosine_distances_rows(a, b)
    assert torch.allclose(dist, torch.zeros(3))
