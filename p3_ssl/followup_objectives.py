from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .losses import masked_mse


VALID_FOLLOWUP_CELLS = frozenset({"R0", "R1", "R2", "R3"})


@dataclass(frozen=True)
class FollowupObjectiveConfig:
    spectral_windows: tuple[int, ...] = (128, 256, 512)
    spectral_hop_divisor: int = 4
    spectral_center: bool = False
    spectral_epsilon: float = 1.0e-6
    time_weight: float = 1.0
    spectral_weight: float = 0.20
    vicreg_weight: float = 0.10
    invariance_weight: float = 1.0
    variance_weight: float = 1.0
    covariance_weight: float = 0.04
    variance_floor: float = 0.50
    variance_epsilon: float = 1.0e-4


def multi_resolution_spectral_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    windows: tuple[int, ...] = (128, 256, 512),
    hop_divisor: int = 4,
    center: bool = False,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if prediction.ndim != 3 or prediction.shape[1] != 1:
        raise ValueError("spectral reconstruction expects shape (batch, 1, time)")
    if not windows or hop_divisor <= 0 or epsilon <= 0.0:
        raise ValueError("invalid spectral objective configuration")
    expanded_mask = mask
    if expanded_mask.ndim == 2:
        expanded_mask = expanded_mask.unsqueeze(1)
    if expanded_mask.shape != prediction.shape:
        raise ValueError("mask must match prediction time support")
    pred_masked = prediction * expanded_mask.to(prediction.dtype)
    target_masked = target * expanded_mask.to(target.dtype)
    terms = {}
    for window_length in windows:
        if window_length > prediction.shape[-1] or window_length % hop_divisor:
            raise ValueError(f"invalid STFT window: {window_length}")
        window = torch.hann_window(
            window_length, dtype=prediction.dtype, device=prediction.device
        )
        kwargs = {
            "n_fft": window_length,
            "hop_length": window_length // hop_divisor,
            "win_length": window_length,
            "window": window,
            "center": center,
            "return_complex": True,
        }
        pred_stft = torch.stft(pred_masked[:, 0], **kwargs)
        target_stft = torch.stft(target_masked[:, 0], **kwargs)
        pred_log = torch.log(torch.abs(pred_stft) + epsilon)
        target_log = torch.log(torch.abs(target_stft) + epsilon)
        terms[f"stft_{window_length}"] = F.smooth_l1_loss(pred_log, target_log)
    total = torch.stack(list(terms.values())).mean()
    return total, terms


def vicreg_pair_loss(
    embeddings: torch.Tensor,
    *,
    invariance_weight: float = 1.0,
    variance_weight: float = 1.0,
    covariance_weight: float = 0.04,
    variance_floor: float = 0.50,
    epsilon: float = 1.0e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if embeddings.ndim != 3 or embeddings.shape[1] < 2:
        raise ValueError("embeddings must have shape (batch, views>=2, dimensions)")
    if embeddings.shape[0] < 2:
        raise ValueError("VICReg requires at least two independent latents per batch")
    first = embeddings[:, 0]
    second = embeddings[:, 1]
    invariance = F.mse_loss(first, second)
    variance_terms = []
    covariance_terms = []
    for values in (first, second):
        standard_deviation = torch.sqrt(values.var(dim=0, unbiased=True) + epsilon)
        variance_terms.append(F.relu(variance_floor - standard_deviation).mean())
        centered = values - values.mean(dim=0)
        covariance = centered.T @ centered / (len(values) - 1)
        off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
        covariance_terms.append(off_diagonal.pow(2).sum() / values.shape[1])
    variance = torch.stack(variance_terms).mean()
    covariance = torch.stack(covariance_terms).mean()
    total = (
        invariance_weight * invariance
        + variance_weight * variance
        + covariance_weight * covariance
    )
    return total, {
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
    }


def followup_ssl_objective(
    cell: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    paired_embeddings: torch.Tensor | None,
    config: FollowupObjectiveConfig = FollowupObjectiveConfig(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if cell not in VALID_FOLLOWUP_CELLS:
        raise ValueError(f"Unknown follow-up cell: {cell}")
    time = masked_mse(prediction, target, mask)
    total = config.time_weight * time
    terms = {"time_reconstruction": time}
    if cell in {"R2", "R3"}:
        spectral, spectral_terms = multi_resolution_spectral_loss(
            prediction,
            target,
            mask,
            windows=config.spectral_windows,
            hop_divisor=config.spectral_hop_divisor,
            center=config.spectral_center,
            epsilon=config.spectral_epsilon,
        )
        total = total + config.spectral_weight * spectral
        terms["spectral_reconstruction"] = spectral
        terms.update(spectral_terms)
    if cell in {"R1", "R3"}:
        if paired_embeddings is None:
            raise ValueError(f"{cell} requires paired embeddings")
        vicreg, vicreg_terms = vicreg_pair_loss(
            paired_embeddings,
            invariance_weight=config.invariance_weight,
            variance_weight=config.variance_weight,
            covariance_weight=config.covariance_weight,
            variance_floor=config.variance_floor,
            epsilon=config.variance_epsilon,
        )
        total = total + config.vicreg_weight * vicreg
        terms["vicreg"] = vicreg
        terms.update({f"vicreg_{name}": value for name, value in vicreg_terms.items()})
    terms["total"] = total
    return total, terms
