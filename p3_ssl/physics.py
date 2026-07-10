from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


PHYSICS_PARAM_NAMES = ("A", "fD_khz", "phi_rad", "t0_fraction", "tau_ms", "snr_db")
NONCIRCULAR_PARAM_NAMES = ("A", "fD_khz", "t0_fraction", "tau_ms", "snr_db")


@dataclass(frozen=True)
class PhysicsRanges:
    low: np.ndarray
    high: np.ndarray


def _as_float_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D parameter matrix, got shape {arr.shape}")
    if arr.shape[1] != len(PHYSICS_PARAM_NAMES):
        raise ValueError(f"Expected {len(PHYSICS_PARAM_NAMES)} physics columns, got {arr.shape[1]}")
    return arr


def infer_physics_ranges(params: np.ndarray) -> PhysicsRanges:
    arr = _as_float_matrix(params)
    low = np.zeros(arr.shape[1], dtype=np.float64)
    high = np.ones(arr.shape[1], dtype=np.float64)
    for idx in range(arr.shape[1]):
        finite = np.isfinite(arr[:, idx])
        if finite.any():
            low[idx] = float(np.min(arr[finite, idx]))
            high[idx] = float(np.max(arr[finite, idx]))
    same = ~np.isfinite(low) | ~np.isfinite(high) | np.isclose(high, low)
    low[same] = 0.0
    high[same] = 1.0
    return PhysicsRanges(low=low.astype(np.float64), high=high.astype(np.float64))


def normalized_physics_features(params: np.ndarray, ranges: PhysicsRanges | None = None) -> np.ndarray:
    arr = _as_float_matrix(params)
    ranges = ranges or infer_physics_ranges(arr)
    out = (arr - ranges.low[None, :]) / np.maximum(ranges.high - ranges.low, 1.0e-12)[None, :]
    phi = np.mod(arr[:, 2], 2.0 * np.pi)
    out[:, 2] = phi / (2.0 * np.pi)
    return out.astype(np.float64)


def circular_phase_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    delta = np.abs(a - b)
    return np.minimum(delta, 2.0 * np.pi - delta)


def normalized_physical_distance_matrix(
    params: np.ndarray,
    ranges: PhysicsRanges | None = None,
    include_snr: bool = True,
) -> np.ndarray:
    arr = _as_float_matrix(params)
    ranges = ranges or infer_physics_ranges(arr)
    diffs: list[np.ndarray] = []
    for idx, name in enumerate(PHYSICS_PARAM_NAMES):
        if name == "snr_db" and not include_snr:
            continue
        values = arr[:, idx]
        if name == "phi_rad":
            diff = circular_phase_delta(values[:, None], values[None, :]) / np.pi
        else:
            scale = max(float(ranges.high[idx] - ranges.low[idx]), 1.0e-12)
            diff = np.abs(values[:, None] - values[None, :]) / scale
        valid = np.isfinite(diff)
        diffs.append(np.where(valid, diff, np.nan))
    stacked = np.stack(diffs, axis=0)
    dist = np.nanmean(stacked, axis=0)
    return np.where(np.isfinite(dist), dist, np.inf).astype(np.float32)


def physical_contrastive_loss(
    embeddings: torch.Tensor,
    physics_params: torch.Tensor,
    positive_distance: float = 0.18,
    negative_distance: float = 0.55,
    margin: float = 1.0,
) -> torch.Tensor:
    """Pairwise physical contrastive loss for synthetic rows with valid parameters."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (B, D)")
    if physics_params.ndim != 2 or physics_params.shape[1] != len(PHYSICS_PARAM_NAMES):
        raise ValueError("physics_params must have shape (B, 6)")
    finite_rows = torch.isfinite(physics_params).all(dim=1)
    if int(finite_rows.sum().item()) < 2:
        return embeddings.sum() * 0.0
    z = F.normalize(embeddings[finite_rows], dim=-1)
    p = physics_params[finite_rows]
    latent_dist = torch.cdist(z, z, p=2)

    columns: list[torch.Tensor] = []
    for idx, name in enumerate(PHYSICS_PARAM_NAMES):
        values = p[:, idx]
        if name == "phi_rad":
            delta = torch.abs(values[:, None] - values[None, :])
            columns.append(torch.minimum(delta, (2.0 * torch.pi) - delta) / torch.pi)
        else:
            scale = (values.max() - values.min()).clamp_min(1.0e-6)
            columns.append(torch.abs(values[:, None] - values[None, :]) / scale)
    phys_dist = torch.stack(columns, dim=0).mean(dim=0)
    eye = torch.eye(phys_dist.shape[0], dtype=torch.bool, device=phys_dist.device)
    positive = (phys_dist <= positive_distance) & ~eye
    negative = phys_dist >= negative_distance

    losses: list[torch.Tensor] = []
    if bool(positive.any()):
        losses.append(latent_dist[positive].pow(2).mean())
    if bool(negative.any()):
        losses.append(F.relu(margin - latent_dist[negative]).pow(2).mean())
    if not losses:
        return embeddings.sum() * 0.0
    return torch.stack(losses).mean()


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return float("nan")
    x = a[valid].astype(np.float64)
    y = b[valid].astype(np.float64)
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _linear_probe_predictions(embeddings: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    valid = np.isfinite(target)
    x = np.asarray(embeddings[valid], dtype=np.float64)
    y = np.asarray(target[valid], dtype=np.float64)
    if x.shape[0] < 3 or np.isclose(np.std(y), 0.0):
        return np.full_like(target, np.nan, dtype=np.float64), float("nan")
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred_valid = design @ coef
    pred = np.full_like(target, np.nan, dtype=np.float64)
    pred[valid] = pred_valid
    denom = float(np.sum(np.square(y - y.mean())))
    r2 = 1.0 - float(np.sum(np.square(y - pred_valid))) / denom if denom > 0 else float("nan")
    return pred, r2


def _local_monotonicity_score(target: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(target) & np.isfinite(prediction)
    if int(valid.sum()) < 3:
        return float("nan")
    y = np.asarray(target[valid], dtype=np.float64)
    pred = np.asarray(prediction[valid], dtype=np.float64)
    order = np.argsort(y, kind="mergesort")
    dy = np.diff(y[order])
    dp = np.diff(pred[order])
    increasing = dy > 0
    if int(increasing.sum()) < 2:
        return float("nan")
    local = np.where(dp[increasing] > 0, 1.0, np.where(np.isclose(dp[increasing], 0.0), 0.5, 0.0))
    return float(np.mean(local))


def _knn_indices(dist: np.ndarray, k: int) -> np.ndarray:
    masked = dist.copy()
    np.fill_diagonal(masked, np.inf)
    return np.argsort(masked, axis=1)[:, :k]


def euclidean_distance_matrix(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    norms = np.sum(np.square(x), axis=1, keepdims=True)
    dist_sq = np.maximum(norms + norms.T - 2.0 * (x @ x.T), 0.0)
    return np.sqrt(dist_sq, out=dist_sq)


def evaluate_physical_latent_space(
    embeddings: np.ndarray,
    physics_params: np.ndarray,
    k_neighbors: int = 5,
    pass_threshold: float = 0.05,
) -> dict[str, object]:
    """Evaluate whether latent geometry preserves known physical parameters."""
    z = np.asarray(embeddings, dtype=np.float64)
    params = _as_float_matrix(physics_params)
    finite_rows = np.isfinite(params).any(axis=1) & np.isfinite(z).all(axis=1)
    z = z[finite_rows]
    params = params[finite_rows]
    if z.shape[0] < 3:
        return {"n_samples": int(z.shape[0]), "physical_score": 0.0, "status": "insufficient_samples"}

    z = (z - z.mean(axis=0, keepdims=True)) / np.maximum(z.std(axis=0, keepdims=True), 1.0e-8)
    per_param: dict[str, dict[str, float]] = {}
    r2_values: list[float] = []
    spearman_values: list[float] = []
    monotonicity_values: list[float] = []
    sensitivity: dict[str, float] = {}
    for idx, name in enumerate(PHYSICS_PARAM_NAMES):
        values = params[:, idx]
        finite_values = np.isfinite(values)
        if int(finite_values.sum()) < 3 or np.isclose(np.nanstd(values), 0.0):
            per_param[name] = {
                "pearson": float("nan"),
                "spearman": float("nan"),
                "linear_probe_r2": float("nan"),
                "local_monotonicity": float("nan"),
            }
            continue
        if name == "phi_rad":
            target = np.column_stack([np.sin(values), np.cos(values)])
            pred_sin, r2_sin = _linear_probe_predictions(z, target[:, 0])
            pred_cos, r2_cos = _linear_probe_predictions(z, target[:, 1])
            pred_angle = np.mod(np.arctan2(pred_sin, pred_cos), 2.0 * np.pi)
            circular_err = circular_phase_delta(np.mod(values, 2.0 * np.pi), pred_angle)
            score_r2 = float(np.nanmean([r2_sin, r2_cos]))
            score_spearman = 1.0 - float(np.nanmean(circular_err) / np.pi)
            pearson = float("nan")
            r2 = score_r2
            local_monotonicity = float("nan")
        else:
            pred, r2 = _linear_probe_predictions(z, values)
            pearson = _corr(values, pred)
            score_spearman = _corr(_rankdata(values), _rankdata(pred))
            local_monotonicity = _local_monotonicity_score(values, pred)
        if np.isfinite(r2):
            r2_values.append(float(r2))
        if np.isfinite(score_spearman):
            spearman_values.append(float(score_spearman))
        if np.isfinite(local_monotonicity):
            monotonicity_values.append(float(local_monotonicity))
        order = np.argsort(values[finite_values])
        if order.size >= 2:
            finite_z = z[finite_values]
            step = np.linalg.norm(np.diff(finite_z[order], axis=0), axis=1)
            sensitivity[name] = float(np.nanmedian(step))
        per_param[name] = {
            "pearson": pearson,
            "spearman": float(score_spearman),
            "linear_probe_r2": float(r2),
            "local_monotonicity": float(local_monotonicity),
        }

    latent_dist = euclidean_distance_matrix(z)
    physical_dist = normalized_physical_distance_matrix(params)
    k = max(1, min(k_neighbors, z.shape[0] - 1))
    nearest = _knn_indices(latent_dist, k)
    row_idx = np.arange(z.shape[0])[:, None]
    latent_neighbor_physical_distance = float(np.mean(physical_dist[row_idx, nearest]))
    rng = np.random.default_rng(123)
    random_idx = np.stack([rng.choice(z.shape[0] - 1, size=k, replace=False) for _ in range(z.shape[0])], axis=0)
    random_idx = random_idx + (random_idx >= np.arange(z.shape[0])[:, None])
    random_physical_distance = float(np.mean(physical_dist[row_idx, random_idx]))
    neighbor_gain = (
        1.0 - latent_neighbor_physical_distance / random_physical_distance
        if random_physical_distance > 0
        else 0.0
    )
    score_parts = [
        float(np.nanmean(np.clip(r2_values, 0.0, 1.0))) if r2_values else 0.0,
        float(np.nanmean(np.clip(spearman_values, -1.0, 1.0))) if spearman_values else 0.0,
        float(np.clip(neighbor_gain, -1.0, 1.0)),
    ]
    physical_score = float(np.mean(score_parts))
    return {
        "n_samples": int(z.shape[0]),
        "physical_score": physical_score,
        "physical_validation_pass": bool(physical_score > float(pass_threshold) and neighbor_gain > 0.0),
        "pass_threshold": float(pass_threshold),
        "mean_spearman": float(np.nanmean(spearman_values)) if spearman_values else None,
        "mean_linear_probe_r2": float(np.nanmean(r2_values)) if r2_values else None,
        "mean_local_monotonicity": float(np.nanmean(monotonicity_values)) if monotonicity_values else None,
        "latent_neighbor_physical_distance": latent_neighbor_physical_distance,
        "random_neighbor_physical_distance": random_physical_distance,
        "neighbor_gain": float(neighbor_gain),
        "per_parameter": per_param,
        "sensitivity_by_parameter": sensitivity,
    }
