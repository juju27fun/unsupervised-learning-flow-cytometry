from __future__ import annotations

from pathlib import Path

import torch

from p3_ssl.config import load_config
from p3_ssl.mask_collapse_training import mask_collapse_loss
from p3_ssl.study_model import YeastStudyModel, YeastStudyModelConfig


CONFIG_PATH = Path(__file__).parents[1] / "configs/yeast_ssl_rebuild_v1.yaml"


def _tiny_config() -> dict:
    config = load_config(CONFIG_PATH)
    config["data"]["sampling_frequency_hz"] = 1_000
    config["masking"].update(
        {
            "strategy": "patch_aligned_isolated",
            "mask_ratio": 0.25,
            "minimum_visible_tokens_between_masks": 1,
            "high_derivative_probability": 0.0,
            "event_biased_probability": 0.0,
        }
    )
    config["model"].update(
        {"patch_size": 4, "patch_stride": 4, "max_tokens": 16}
    )
    return config


def _tiny_model() -> YeastStudyModel:
    return YeastStudyModel(
        YeastStudyModelConfig(
            input_length=64,
            patch_size=4,
            patch_stride=4,
            d_model=16,
            n_heads=4,
            n_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            max_tokens=16,
        )
    )


def test_mask_collapse_cells_have_finite_gradients() -> None:
    signals = torch.stack(
        (
            torch.sin(torch.arange(64, dtype=torch.float32) * 0.2),
            torch.cos(torch.arange(64, dtype=torch.float32) * 0.17),
        )
    ).unsqueeze(1)
    events = torch.zeros(2, 64, dtype=torch.bool)
    events[:, 8:56] = True
    for cell in ("C0", "C1"):
        model = _tiny_model()
        loss, metrics = mask_collapse_loss(
            model,
            signals,
            events,
            _tiny_config(),
            cell=cell,
            seed=3,
            vicreg_weight=1.0,
        )
        loss.backward()
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        assert metrics["vicreg"] == 0.0 if cell == "C0" else metrics["vicreg"] > 0.0
