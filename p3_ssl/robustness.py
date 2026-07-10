from __future__ import annotations

from typing import Any

import numpy as np
import torch


def perturb_signal_batch(signals: torch.Tensor, perturbation: str) -> torch.Tensor:
    if signals.ndim != 3 or signals.shape[1] != 1:
        raise ValueError(f"Expected signals with shape (B, 1, L), got {tuple(signals.shape)}")
    x = signals.clone()
    if perturbation == "noise_0p10":
        scale = x.std(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return x + torch.randn_like(x) * scale * 0.10
    if perturbation == "scale_1p25":
        return x * 1.25
    if perturbation == "shift_8":
        return torch.roll(x, shifts=8, dims=-1)
    if perturbation == "center_mask_64":
        width = min(64, x.shape[-1])
        start = max(0, x.shape[-1] // 2 - width // 2)
        end = min(x.shape[-1], start + width)
        x[..., start:end] = 0.0
        return x
    raise ValueError(f"Unsupported perturbation: {perturbation}")


def cosine_distances_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"embedding shapes must match, got {tuple(a.shape)} and {tuple(b.shape)}")
    a_norm = a / a.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    b_norm = b / b.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return 1.0 - (a_norm * b_norm).sum(dim=-1)


@torch.no_grad()
def embedding_robustness_metrics(
    model,
    loader,
    device: torch.device,
    pool: str,
    perturbations: list[str] | None = None,
    max_samples: int | None = 128,
    real_only: bool = True,
) -> dict[str, Any]:
    selected: list[torch.Tensor] = []
    total_seen = 0
    real_seen = 0
    for batch in loader:
        signal = batch["signal"]
        source_kind = batch.get("source_kind")
        if real_only and source_kind is not None:
            mask = torch.as_tensor([str(item) != "synthetic" for item in source_kind], dtype=torch.bool)
            real_seen += int(mask.sum().item())
            if not bool(mask.any()):
                continue
            signal = signal[mask]
        total_seen += int(signal.shape[0])
        selected.append(signal)
        if max_samples is not None and sum(int(chunk.shape[0]) for chunk in selected) >= max_samples:
            break
    if not selected:
        return {"status": "not_run", "reason": "no matching samples", "real_only": real_only, "total_seen": total_seen, "real_seen": real_seen}

    signals = torch.cat(selected, dim=0)
    if max_samples is not None:
        signals = signals[:max_samples]
    signals = signals.to(device)
    model.eval()
    base = model.global_embedding(signals, token_mask=None, pool=pool)
    rows: dict[str, Any] = {}
    for perturbation in perturbations or ["noise_0p10", "scale_1p25", "shift_8", "center_mask_64"]:
        perturbed = perturb_signal_batch(signals, perturbation)
        emb = model.global_embedding(perturbed, token_mask=None, pool=pool)
        dist = cosine_distances_rows(base, emb).detach().cpu().numpy().astype(np.float64)
        rows[perturbation] = {
            "cosine_distance_mean": float(np.mean(dist)),
            "cosine_distance_std": float(np.std(dist)),
            "cosine_distance_median": float(np.median(dist)),
        }
    return {
        "status": "ok",
        "real_only": real_only,
        "n_samples": int(signals.shape[0]),
        "total_seen": total_seen,
        "real_seen": real_seen,
        "perturbations": rows,
    }
