from __future__ import annotations

import pytest
import torch

from p3_ssl.followup_objectives import (
    followup_ssl_objective,
    multi_resolution_spectral_loss,
    vicreg_pair_loss,
)


@pytest.mark.parametrize("cell", ["R0", "R1", "R2", "R3"])
def test_followup_objectives_have_finite_gradients(cell: str) -> None:
    torch.manual_seed(3)
    target = torch.randn(4, 1, 1024)
    prediction = torch.randn(4, 1, 1024, requires_grad=True)
    mask = torch.zeros(4, 1, 1024, dtype=torch.bool)
    mask[..., 256:768] = True
    embeddings = torch.randn(4, 2, 96, requires_grad=True)
    loss, terms = followup_ssl_objective(
        cell,
        prediction,
        target,
        mask,
        paired_embeddings=embeddings if cell in {"R1", "R3"} else None,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert all(torch.isfinite(value) for value in terms.values())
    if cell in {"R1", "R3"}:
        assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()


def test_spectral_loss_rejects_incompatible_window() -> None:
    values = torch.ones(2, 1, 64)
    with pytest.raises(ValueError, match="window"):
        multi_resolution_spectral_loss(values, values, torch.ones_like(values, dtype=torch.bool))


def test_spectral_loss_uses_uncentered_stft_without_reflection_padding() -> None:
    values = torch.randn(2, 1, 1024, requires_grad=True)
    mask = torch.ones_like(values, dtype=torch.bool)
    loss, terms = multi_resolution_spectral_loss(values, values.detach(), mask, center=False)
    loss.backward()
    assert set(terms) == {"stft_128", "stft_256", "stft_512"}
    assert torch.isfinite(values.grad).all()


def test_vicreg_rejects_single_latent_batch() -> None:
    with pytest.raises(ValueError, match="independent"):
        vicreg_pair_loss(torch.ones(1, 2, 4))


def test_vicreg_covariance_penalizes_redundant_dimensions() -> None:
    base = torch.linspace(-1.0, 1.0, 32).unsqueeze(1)
    redundant = base.repeat(1, 8)
    independent = torch.randn(32, 8)
    _, redundant_terms = vicreg_pair_loss(torch.stack([redundant, redundant], dim=1))
    _, independent_terms = vicreg_pair_loss(torch.stack([independent, independent], dim=1))
    assert redundant_terms["covariance"] > independent_terms["covariance"]
