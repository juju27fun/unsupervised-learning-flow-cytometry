from __future__ import annotations

import torch

from p3_ssl.losses import composite_reconstruction_loss, masked_mse


def test_masked_mse_uses_only_masked_positions() -> None:
    pred = torch.tensor([[[0.0, 2.0, 10.0]]])
    target = torch.tensor([[[0.0, 0.0, 0.0]]])
    mask = torch.tensor([[False, True, False]])
    loss = masked_mse(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(4.0))


def test_composite_loss_is_finite() -> None:
    pred = torch.zeros(2, 1, 8)
    target = torch.ones(2, 1, 8)
    mask = torch.ones(2, 8, dtype=torch.bool)
    loss, parts = composite_reconstruction_loss(pred, target, mask)
    assert torch.isfinite(loss)
    assert {"loss", "signal_mse", "derivative_huber", "energy_huber"} <= set(parts)
