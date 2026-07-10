from __future__ import annotations

import numpy as np
import torch

from p3_ssl.physics import (
    evaluate_physical_latent_space,
    normalized_physical_distance_matrix,
    physical_contrastive_loss,
)


def test_normalized_distance_uses_circular_phase() -> None:
    params = np.asarray(
        [
            [1.0, 10.0, 0.01, 0.4, 0.5, 20.0],
            [1.0, 10.0, 2.0 * np.pi - 0.01, 0.4, 0.5, 20.0],
        ],
        dtype=np.float32,
    )
    dist = normalized_physical_distance_matrix(params)
    assert dist.shape == (2, 2)
    assert 0.0 < dist[0, 1] < 0.01


def test_physical_contrastive_loss_is_finite() -> None:
    embeddings = torch.randn(4, 8)
    params = torch.tensor(
        [
            [1.0, 10.0, 0.0, 0.4, 0.5, 20.0],
            [1.1, 10.5, 0.1, 0.41, 0.52, 19.0],
            [3.0, 30.0, 3.0, 0.8, 1.1, 5.0],
            [3.1, 31.0, 3.1, 0.79, 1.0, 6.0],
        ]
    )
    loss = physical_contrastive_loss(embeddings, params)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_evaluate_physical_latent_space_reports_neighbor_gain() -> None:
    values = np.linspace(0.0, 1.0, 24, dtype=np.float32)
    embeddings = np.stack([values, values**2], axis=1)
    params = np.stack(
        [
            1.0 + values,
            10.0 + 5.0 * values,
            np.pi * values,
            0.2 + 0.6 * values,
            0.3 + values,
            20.0 - values,
        ],
        axis=1,
    )
    metrics = evaluate_physical_latent_space(embeddings, params, k_neighbors=3)
    assert metrics["n_samples"] == 24
    assert metrics["physical_score"] > 0.0
    assert metrics["neighbor_gain"] > 0.0
    assert metrics["mean_local_monotonicity"] > 0.9
    assert metrics["per_parameter"]["A"]["local_monotonicity"] > 0.9


def test_evaluate_physical_latent_space_uses_configurable_pass_threshold() -> None:
    values = np.linspace(0.0, 1.0, 24, dtype=np.float32)
    embeddings = np.stack([values, values**2], axis=1)
    params = np.stack(
        [
            1.0 + values,
            10.0 + 5.0 * values,
            np.pi * values,
            0.2 + 0.6 * values,
            0.3 + values,
            20.0 - values,
        ],
        axis=1,
    )
    metrics = evaluate_physical_latent_space(embeddings, params, k_neighbors=3, pass_threshold=2.0)
    assert metrics["pass_threshold"] == 2.0
    assert metrics["physical_validation_pass"] is False


def test_evaluate_physical_latent_space_handles_partial_real_estimates() -> None:
    values = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    embeddings = np.stack([values, values**2, 1.0 - values], axis=1)
    params = np.full((16, 6), np.nan, dtype=np.float32)
    params[:, 1] = 20.0 + values
    params[:, 3] = 0.2 + 0.4 * values
    params[:, 4] = 0.1 + 0.2 * values
    params[:, 5] = 5.0 + values

    metrics = evaluate_physical_latent_space(embeddings, params, k_neighbors=3)
    assert metrics["n_samples"] == 16
    assert metrics["per_parameter"]["fD_khz"]["spearman"] > 0.0
    assert metrics["per_parameter"]["fD_khz"]["local_monotonicity"] > 0.9
    assert metrics["per_parameter"]["t0_fraction"]["spearman"] > 0.0
    assert metrics["per_parameter"]["tau_ms"]["spearman"] > 0.0
    assert np.isnan(metrics["per_parameter"]["A"]["spearman"])
    assert np.isnan(metrics["per_parameter"]["A"]["local_monotonicity"])
    assert np.isnan(metrics["per_parameter"]["phi_rad"]["spearman"])
    assert np.isnan(metrics["per_parameter"]["phi_rad"]["local_monotonicity"])
