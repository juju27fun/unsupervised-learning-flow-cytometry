from __future__ import annotations

import torch


@torch.no_grad()
def reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Compute masked reconstruction metrics for logging/evaluation."""
    mask_f = mask.to(dtype=pred.dtype, device=pred.device)
    while mask_f.ndim < pred.ndim:
        mask_f = mask_f.unsqueeze(1)
    denom = mask_f.sum().clamp_min(1.0)
    err = pred - target
    mse = (err.pow(2) * mask_f).sum() / denom
    mae = (err.abs() * mask_f).sum() / denom

    diff_mask = (mask[..., 1:] | mask[..., :-1]).to(dtype=pred.dtype, device=pred.device)
    while diff_mask.ndim < pred.ndim:
        diff_mask = diff_mask.unsqueeze(1)
    pred_diff = pred[..., 1:] - pred[..., :-1]
    target_diff = target[..., 1:] - target[..., :-1]
    dd = pred_diff - target_diff
    derivative_mse = (dd.pow(2) * diff_mask).sum() / diff_mask.sum().clamp_min(1.0)

    pred_energy = (pred.pow(2) * mask_f).sum(dim=-1)
    target_energy = (target.pow(2) * mask_f).sum(dim=-1)
    energy_abs_error = (pred_energy - target_energy).abs().mean()
    return {
        "masked_mse": float(mse.cpu()),
        "masked_mae": float(mae.cpu()),
        "derivative_mse": float(derivative_mse.cpu()),
        "energy_abs_error": float(energy_abs_error.cpu()),
    }

