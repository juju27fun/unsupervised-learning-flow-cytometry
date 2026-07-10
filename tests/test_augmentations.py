from __future__ import annotations

import torch

from p3_ssl.augmentations import cosine_distance_loss, positive_signal_augmentation


def test_positive_signal_augmentation_preserves_shape_and_changes_signal() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 1, 64)
    y = positive_signal_augmentation(
        x,
        noise_std_fraction=0.01,
        max_shift_points=2,
        amplitude_scale_min=0.95,
        amplitude_scale_max=1.05,
        phase_jitter_rad=0.01,
    )
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert not torch.allclose(x, y)


def test_cosine_distance_loss_zero_for_identical_embeddings() -> None:
    z = torch.randn(5, 8)
    loss = cosine_distance_loss(z, z)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1.0e-6)
