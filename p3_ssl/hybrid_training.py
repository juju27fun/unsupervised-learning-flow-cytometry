from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class TrainingStage:
    name: str
    epochs: int
    synthetic_fraction: float


def profile_value(config_value: Any, profile: str) -> Any:
    if isinstance(config_value, dict):
        if profile in config_value:
            return config_value[profile]
        if "full" in config_value:
            return config_value["full"]
    return config_value


def build_training_stages(config: dict[str, Any], profile: str) -> list[TrainingStage]:
    training_cfg = config.get("training", {})
    hybrid_cfg = config.get("hybrid_sampling", {})
    adaptation_cfg = config.get("real_adaptation", {})
    stages: list[TrainingStage] = []
    pretrain_epochs = int(profile_value(training_cfg.get("epochs", 0), profile) or 0)
    if pretrain_epochs > 0:
        stages.append(
            TrainingStage(
                name="hybrid_pretrain",
                epochs=pretrain_epochs,
                synthetic_fraction=float(hybrid_cfg.get("synthetic_fraction_pretrain", 0.70)),
            )
        )
    adaptation_epochs = int(profile_value(adaptation_cfg.get("epochs", 0), profile) or 0)
    if bool(adaptation_cfg.get("enabled", False)) and adaptation_epochs > 0:
        stages.append(
            TrainingStage(
                name="real_adaptation",
                epochs=adaptation_epochs,
                synthetic_fraction=float(hybrid_cfg.get("synthetic_fraction_adaptation", 0.30)),
            )
        )
    if not stages:
        raise ValueError("At least one training stage must have positive epochs")
    return stages


def hybrid_sampling_weights(source_kinds: Sequence[str], synthetic_fraction: float) -> np.ndarray | None:
    if not 0.0 <= synthetic_fraction <= 1.0:
        raise ValueError("synthetic_fraction must be in [0, 1]")
    synthetic = np.asarray([str(value) == "synthetic" for value in source_kinds], dtype=bool)
    n_syn = int(synthetic.sum())
    n_real = int((~synthetic).sum())
    if n_syn == 0 or n_real == 0:
        return None
    weights = np.where(synthetic, synthetic_fraction / n_syn, (1.0 - synthetic_fraction) / n_real)
    return weights.astype(np.float64)


class FixedRatioHybridBatchSampler(Sampler[list[int]]):
    """Yield batches with a fixed synthetic:real count whenever both sources exist."""

    def __init__(
        self,
        source_kinds: Sequence[str],
        synthetic_fraction: float,
        batch_size: int,
        seed: int = 42,
        num_batches: int | None = None,
    ) -> None:
        if not 0.0 <= synthetic_fraction <= 1.0:
            raise ValueError("synthetic_fraction must be in [0, 1]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        synthetic_mask = np.asarray([str(value) == "synthetic" for value in source_kinds], dtype=bool)
        self.synthetic_indices = np.flatnonzero(synthetic_mask).astype(np.int64)
        self.real_indices = np.flatnonzero(~synthetic_mask).astype(np.int64)
        if self.synthetic_indices.size == 0 or self.real_indices.size == 0:
            raise ValueError("FixedRatioHybridBatchSampler requires both synthetic and non-synthetic rows")
        self.batch_size = int(batch_size)
        self.synthetic_fraction = float(synthetic_fraction)
        self.seed = int(seed)
        self.num_batches = int(num_batches) if num_batches is not None else int(np.ceil(len(source_kinds) / batch_size))
        synthetic_per_batch = int(round(self.batch_size * self.synthetic_fraction))
        if 0.0 < self.synthetic_fraction < 1.0 and self.batch_size > 1:
            synthetic_per_batch = min(max(synthetic_per_batch, 1), self.batch_size - 1)
        self.synthetic_per_batch = min(max(synthetic_per_batch, 0), self.batch_size)
        self.real_per_batch = self.batch_size - self.synthetic_per_batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_batches):
            parts: list[np.ndarray] = []
            if self.synthetic_per_batch > 0:
                parts.append(rng.choice(self.synthetic_indices, size=self.synthetic_per_batch, replace=True))
            if self.real_per_batch > 0:
                parts.append(rng.choice(self.real_indices, size=self.real_per_batch, replace=True))
            batch = np.concatenate(parts).astype(np.int64)
            rng.shuffle(batch)
            yield [int(index) for index in batch.tolist()]

    def __len__(self) -> int:
        return self.num_batches


def fixed_ratio_hybrid_batch_sampler(
    source_kinds: Sequence[str],
    synthetic_fraction: float,
    batch_size: int,
    seed: int = 42,
) -> FixedRatioHybridBatchSampler | None:
    synthetic = np.asarray([str(value) == "synthetic" for value in source_kinds], dtype=bool)
    if int(synthetic.sum()) == 0 or int((~synthetic).sum()) == 0:
        return None
    return FixedRatioHybridBatchSampler(
        source_kinds=source_kinds,
        synthetic_fraction=synthetic_fraction,
        batch_size=batch_size,
        seed=seed,
    )


def synthetic_only_physics_params(physics_params: torch.Tensor, source_kinds: Sequence[str]) -> torch.Tensor:
    """Mask non-synthetic rows so parameter losses cannot use real estimated physics."""
    if physics_params.ndim != 2:
        raise ValueError("physics_params must be a 2D tensor")
    if len(source_kinds) != int(physics_params.shape[0]):
        raise ValueError("source_kinds length must match physics_params batch size")
    synthetic = torch.as_tensor(
        [str(value) == "synthetic" for value in source_kinds],
        dtype=torch.bool,
        device=physics_params.device,
    )
    masked = physics_params.clone()
    masked[~synthetic] = torch.nan
    return masked
