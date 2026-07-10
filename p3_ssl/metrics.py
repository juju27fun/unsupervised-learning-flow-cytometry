from __future__ import annotations

import torch


def _as_time_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected time mask shape (B, L), got {tuple(mask.shape)}")
    return mask.to(dtype=torch.bool)


def _broadcast_time_mask(mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    mask_f = _as_time_mask(mask).to(dtype=values.dtype, device=values.device)
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(1)
    return mask_f


@torch.no_grad()
def reconstruction_metric_sums(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Return additive reconstruction metric sums for exact aggregation."""
    time_mask = _as_time_mask(mask).to(device=pred.device)
    mask_f = _broadcast_time_mask(time_mask, pred)
    err = pred - target
    points = float(mask_f.sum().cpu())
    mse_sum = float((err.pow(2) * mask_f).sum().cpu())
    mae_sum = float((err.abs() * mask_f).sum().cpu())

    diff_mask = (time_mask[..., 1:] | time_mask[..., :-1]).to(device=pred.device)
    diff_mask_f = diff_mask.to(dtype=pred.dtype, device=pred.device)
    while diff_mask_f.ndim < pred.ndim:
        diff_mask_f = diff_mask_f.unsqueeze(1)
    pred_diff = pred[..., 1:] - pred[..., :-1]
    target_diff = target[..., 1:] - target[..., :-1]
    dd = pred_diff - target_diff
    derivative_points = float(diff_mask_f.sum().cpu())
    derivative_mse_sum = float((dd.pow(2) * diff_mask_f).sum().cpu())

    pred_energy = (pred.pow(2) * mask_f).sum(dim=-1).flatten()
    target_energy = (target.pow(2) * mask_f).sum(dim=-1).flatten()
    valid_energy = (mask_f.sum(dim=-1).flatten() > 0).to(dtype=pred.dtype, device=pred.device)
    energy_count = float(valid_energy.sum().cpu())
    energy_abs_error_sum = float(((pred_energy - target_energy).abs() * valid_energy).sum().cpu())

    return {
        "points": points,
        "mse_sum": mse_sum,
        "mae_sum": mae_sum,
        "derivative_points": derivative_points,
        "derivative_mse_sum": derivative_mse_sum,
        "energy_count": energy_count,
        "energy_abs_error_sum": energy_abs_error_sum,
    }


def finalize_reconstruction_metric_sums(sums: dict[str, float], prefix: str = "") -> dict[str, float | None]:
    """Convert additive reconstruction sums into metrics."""
    key = f"{prefix}_" if prefix else ""
    points = float(sums.get("points", 0.0))
    derivative_points = float(sums.get("derivative_points", 0.0))
    energy_count = float(sums.get("energy_count", 0.0))
    return {
        f"{key}masked_points": points,
        f"{key}masked_mse": float(sums.get("mse_sum", 0.0)) / points if points else None,
        f"{key}masked_mae": float(sums.get("mae_sum", 0.0)) / points if points else None,
        f"{key}derivative_points": derivative_points,
        f"{key}derivative_mse": float(sums.get("derivative_mse_sum", 0.0)) / derivative_points if derivative_points else None,
        f"{key}energy_count": energy_count,
        f"{key}energy_abs_error": float(sums.get("energy_abs_error_sum", 0.0)) / energy_count if energy_count else None,
    }


@torch.no_grad()
def reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Compute masked reconstruction metrics for logging/evaluation."""
    sums = reconstruction_metric_sums(pred, target, mask)
    metrics = finalize_reconstruction_metric_sums(sums)
    return {
        "masked_mse": float(metrics["masked_mse"] or 0.0),
        "masked_mae": float(metrics["masked_mae"] or 0.0),
        "derivative_mse": float(metrics["derivative_mse"] or 0.0),
        "energy_abs_error": float(metrics["energy_abs_error"] or 0.0),
    }


@torch.no_grad()
def reconstruction_strata_masks(
    target_mask: torch.Tensor,
    event_mask: torch.Tensor,
    hidden_mask: torch.Tensor | None = None,
    mostly_event_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Classify samples by how much of the labeled event is in the loss mask."""
    target_time = _as_time_mask(target_mask)
    event_time = _as_time_mask(event_mask).to(device=target_time.device)
    if event_time.shape != target_time.shape:
        raise ValueError("event_mask and target_mask must have the same shape")
    event_points = event_time.sum(dim=-1)
    event_target_points = (event_time & target_time).sum(dim=-1)
    event_target_fraction = torch.zeros_like(event_points, dtype=torch.float32)
    has_event = event_points > 0
    event_target_fraction[has_event] = event_target_points[has_event].float() / event_points[has_event].float()

    if hidden_mask is not None:
        hidden_time = _as_time_mask(hidden_mask).to(device=target_time.device)
        if hidden_time.shape != target_time.shape:
            raise ValueError("hidden_mask and target_mask must have the same shape")
        event_hidden_points = (event_time & hidden_time).sum(dim=-1)
        impossible_event = has_event & (event_hidden_points >= event_points)
    else:
        impossible_event = torch.zeros_like(has_event)
    return {
        "background_only": (event_target_points == 0) & ~impossible_event,
        "partial_event": (event_target_points > 0) & (event_target_fraction < mostly_event_threshold) & ~impossible_event,
        "mostly_event": (event_target_fraction >= mostly_event_threshold) & ~impossible_event,
        "impossible_event": impossible_event,
    }


@torch.no_grad()
def mask_coherence_batch_sums(
    target_mask: torch.Tensor,
    hidden_mask: torch.Tensor,
    event_mask: torch.Tensor,
) -> dict[str, float]:
    """Return additive mask/event overlap sums for a batch."""
    target_time = _as_time_mask(target_mask)
    hidden_time = _as_time_mask(hidden_mask).to(device=target_time.device)
    event_time = _as_time_mask(event_mask).to(device=target_time.device)
    if hidden_time.shape != target_time.shape or event_time.shape != target_time.shape:
        raise ValueError("target, hidden, and event masks must have the same shape")

    event_points = event_time.sum(dim=-1)
    target_points = target_time.sum(dim=-1)
    hidden_points = hidden_time.sum(dim=-1)
    event_target_points = (event_time & target_time).sum(dim=-1)
    event_hidden_points = (event_time & hidden_time).sum(dim=-1)
    has_event = event_points > 0
    event_target_fraction = torch.zeros_like(event_points, dtype=torch.float32)
    event_hidden_fraction = torch.zeros_like(event_points, dtype=torch.float32)
    event_target_fraction[has_event] = event_target_points[has_event].float() / event_points[has_event].float()
    event_hidden_fraction[has_event] = event_hidden_points[has_event].float() / event_points[has_event].float()

    return {
        "samples": float(target_time.shape[0]),
        "event_samples": float(has_event.sum().cpu()),
        "target_points": float(target_points.sum().cpu()),
        "hidden_points": float(hidden_points.sum().cpu()),
        "event_points": float(event_points.sum().cpu()),
        "event_target_points": float(event_target_points.sum().cpu()),
        "event_hidden_points": float(event_hidden_points.sum().cpu()),
        "event_target_fraction_sum": float(event_target_fraction[has_event].sum().cpu()),
        "event_hidden_fraction_sum": float(event_hidden_fraction[has_event].sum().cpu()),
        "fully_hidden_event_samples": float((has_event & (event_hidden_points >= event_points)).sum().cpu()),
        "fully_targeted_event_samples": float((has_event & (event_target_points >= event_points)).sum().cpu()),
    }


def finalize_mask_coherence_sums(sums: dict[str, float]) -> dict[str, float | None]:
    event_samples = float(sums.get("event_samples", 0.0))
    event_points = float(sums.get("event_points", 0.0))
    target_points = float(sums.get("target_points", 0.0))
    hidden_points = float(sums.get("hidden_points", 0.0))
    return {
        "mask_samples": float(sums.get("samples", 0.0)),
        "event_samples": event_samples,
        "target_points": target_points,
        "hidden_points": hidden_points,
        "event_points": event_points,
        "event_target_points": float(sums.get("event_target_points", 0.0)),
        "event_hidden_points": float(sums.get("event_hidden_points", 0.0)),
        "target_event_fraction": float(sums.get("event_target_points", 0.0)) / target_points if target_points else None,
        "hidden_event_fraction": float(sums.get("event_hidden_points", 0.0)) / hidden_points if hidden_points else None,
        "mean_event_target_fraction": float(sums.get("event_target_fraction_sum", 0.0)) / event_samples if event_samples else None,
        "mean_event_hidden_fraction": float(sums.get("event_hidden_fraction_sum", 0.0)) / event_samples if event_samples else None,
        "fully_hidden_event_sample_rate": float(sums.get("fully_hidden_event_samples", 0.0)) / event_samples if event_samples else None,
        "fully_targeted_event_sample_rate": float(sums.get("fully_targeted_event_samples", 0.0)) / event_samples if event_samples else None,
    }
