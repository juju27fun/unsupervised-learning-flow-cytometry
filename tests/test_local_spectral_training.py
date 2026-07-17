import torch

from p3_ssl.config import load_config
from p3_ssl.local_spectral_target import LocalSpectralTargetConfig
from p3_ssl.local_spectral_training import local_spectral_ssl_loss
from p3_ssl.study_model import YeastStudyModel, YeastStudyModelConfig


def test_local_spectral_ssl_loss_has_finite_encoder_and_head_gradients() -> None:
    config = load_config("configs/yeast_ssl_rebuild_v1.yaml")
    config["masking"].update(
        {
            "strategy": "patch_aligned_isolated",
            "mask_ratio": 0.25,
            "min_block_ms": 0.016,
            "max_block_ms": 0.016,
            "guard_ms": 0.0,
            "minimum_visible_tokens_between_masks": 1,
            "event_biased_probability": 0.75,
        }
    )
    model = YeastStudyModel(
        YeastStudyModelConfig(
            d_model=16,
            n_heads=4,
            n_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            local_spectral_features=24,
        )
    )
    signals = torch.randn(2, 1, 4096)
    events = torch.zeros(2, 4096, dtype=torch.bool)
    events[:, 512:3584] = True
    loss, metrics = local_spectral_ssl_loss(
        model,
        signals,
        events,
        config,
        seed=42,
        target_config=LocalSpectralTargetConfig(),
        vicreg_weight=1.0,
        vicreg_config={
            "invariance_weight": 1.0,
            "variance_weight": 1.0,
            "covariance_weight": 0.04,
            "variance_floor": 0.5,
            "epsilon": 0.0001,
            "second_view_seed_offset": 10000019,
        },
    )
    loss.backward()
    encoder_gradients = [
        parameter.grad for parameter in model.reconstructor.parameters() if parameter.grad is not None
    ]
    head_gradients = [
        parameter.grad for parameter in model.local_spectral_head.parameters() if parameter.grad is not None
    ]
    assert torch.isfinite(loss)
    assert metrics["local_spectral_prediction"] > 0.0
    assert metrics["vicreg"] > 0.0
    assert encoder_gradients and all(torch.isfinite(value).all() for value in encoder_gradients)
    assert head_gradients and all(torch.isfinite(value).all() for value in head_gradients)
    assert sum(float(value.abs().sum()) for value in encoder_gradients) > 0.0
    assert sum(float(value.abs().sum()) for value in head_gradients) > 0.0
