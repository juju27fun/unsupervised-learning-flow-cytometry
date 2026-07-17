import torch
from torch.utils.data import DataLoader, Dataset

from p3_ssl.config import load_config
from p3_ssl.local_spectral_target import LocalSpectralTargetConfig, local_spectral_target
from p3_ssl.local_spectral_training import evaluate_local_spectral_controls
from p3_ssl.study_model import YeastStudyModel, YeastStudyModelConfig


class _Signals(Dataset):
    def __init__(self) -> None:
        self.signals = torch.randn(4, 1, 4096)
        self.events = torch.zeros(4, 4096, dtype=torch.bool)
        self.events[:, 512:3584] = True

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"signal": self.signals[index], "event_mask": self.events[index]}


def test_local_spectral_evaluation_reports_all_controls_and_regions() -> None:
    data = _Signals()
    loader = DataLoader(data, batch_size=2, shuffle=False)
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
    target_config = LocalSpectralTargetConfig()
    constant = local_spectral_target(data.signals, target_config).mean(dim=0)
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
    result = evaluate_local_spectral_controls(
        model,
        loader,
        constant,
        config,
        seed=42,
        target_config=target_config,
        device=torch.device("cpu"),
    )
    assert set(result["regions"]) == {
        "model",
        "zero",
        "train_constant",
        "feature_of_interpolation",
    }
    assert set(result["regions"]["model"]) == {
        "all",
        "event",
        "background",
        "boundary",
    }
    assert result["model_masked_feature_mse"] > 0.0
    assert result["feature_of_interpolation_masked_feature_mse"] >= 0.0
    assert result["model_output_rms_fraction_of_target"] > 0.0
