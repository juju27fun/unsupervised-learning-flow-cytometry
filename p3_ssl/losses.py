from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_reduce(values: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=values.dtype, device=values.device)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_reduce((pred - target).pow(2), mask)


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    values = F.huber_loss(pred, target, reduction="none", delta=delta)
    return _masked_reduce(values, mask)


def derivative_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    pred_diff = pred[..., 1:] - pred[..., :-1]
    target_diff = target[..., 1:] - target[..., :-1]
    diff_mask = mask[..., 1:] | mask[..., :-1]
    return masked_huber(pred_diff, target_diff, diff_mask, delta=delta)


def energy_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    mask_f = mask.to(dtype=pred.dtype, device=pred.device)
    while mask_f.ndim < pred.ndim:
        mask_f = mask_f.unsqueeze(1)
    pred_energy = (pred.pow(2) * mask_f).sum(dim=-1)
    target_energy = (target.pow(2) * mask_f).sum(dim=-1)
    return F.huber_loss(pred_energy, target_energy, reduction="mean", delta=delta)


def composite_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lambda_signal: float = 1.0,
    lambda_derivative: float = 0.2,
    lambda_energy: float = 0.05,
    huber_delta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Signal-space SSL loss used by the first P3_SSL experiment."""
    signal_loss = masked_mse(pred, target, mask)
    deriv_loss = derivative_huber(pred, target, mask, delta=huber_delta)
    eng_loss = energy_huber(pred, target, mask, delta=huber_delta)
    total = (
        lambda_signal * signal_loss
        + lambda_derivative * deriv_loss
        + lambda_energy * eng_loss
    )
    return total, {
        "loss": total.detach(),
        "signal_mse": signal_loss.detach(),
        "derivative_huber": deriv_loss.detach(),
        "energy_huber": eng_loss.detach(),
    }

