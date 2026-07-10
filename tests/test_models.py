from __future__ import annotations

import torch

from p3_ssl.config import ssl_token_count
from p3_ssl.models import MomentLikeConfig, MomentLikeReconstructor, TinyTCNAutoencoder


def test_moment_like_reconstructor_shape() -> None:
    model = MomentLikeReconstructor(
        MomentLikeConfig(
            input_length=32,
            patch_size=4,
            patch_stride=4,
            d_model=16,
            n_heads=4,
            n_layers=1,
            dim_feedforward=32,
            max_tokens=16,
        )
    )
    x = torch.randn(2, 1, 32)
    token_mask = torch.zeros(2, 8, dtype=torch.bool)
    token_mask[:, 2] = True
    y = model(x, token_mask=token_mask)
    assert y.shape == x.shape


def test_moment_like_overlap_shape() -> None:
    model = MomentLikeReconstructor(
        MomentLikeConfig(
            input_length=32,
            patch_size=8,
            patch_stride=4,
            d_model=16,
            n_heads=4,
            n_layers=1,
            dim_feedforward=32,
            max_tokens=16,
        )
    )
    x = torch.randn(2, 1, 32)
    token_mask = torch.zeros(2, model.n_tokens, dtype=torch.bool)
    y = model(x, token_mask=token_mask)
    assert y.shape == x.shape


def test_moment_like_global_embedding_shape() -> None:
    model = MomentLikeReconstructor(
        MomentLikeConfig(
            input_length=32,
            patch_size=4,
            patch_stride=4,
            d_model=16,
            n_heads=4,
            n_layers=1,
            dim_feedforward=32,
            max_tokens=16,
        )
    )
    x = torch.randn(2, 1, 32)
    embedding = model.global_embedding(x)
    assert embedding.shape == (2, 16)


def test_tcn_autoencoder_preserves_length() -> None:
    model = TinyTCNAutoencoder(channels=8)
    x = torch.randn(2, 1, 64)
    y = model(x)
    assert y.shape == x.shape


def test_full_window_4096_patch_4_stride_4_uses_1024_tokens() -> None:
    cfg = MomentLikeConfig(
        input_length=4096,
        patch_size=4,
        patch_stride=4,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        max_tokens=1024,
    )
    model = MomentLikeReconstructor(cfg)

    assert ssl_token_count(cfg.input_length, cfg.patch_size, cfg.patch_stride) == 1024
    assert model.n_tokens == 1024
