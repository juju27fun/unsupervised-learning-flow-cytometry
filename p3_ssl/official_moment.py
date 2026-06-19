from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_PYTHON = PROJECT_ROOT / "vendor" / "python"
VENDOR_MOMENT_RESEARCH = PROJECT_ROOT / "vendor" / "moment-research"
DEFAULT_HF_CACHE = PROJECT_ROOT / "outputs" / "hf_cache"


@dataclass(frozen=True)
class OfficialMomentSpec:
    model_id: str = "AutonLab/MOMENT-1-large"
    seq_len: int = 512
    patch_len: int = 8
    patch_stride_len: int = 8
    n_channels: int = 1


def configure_official_moment_paths(cache_dir: Path | None = None) -> None:
    """Expose vendored MOMENT code/deps and keep HF artifacts inside P3_SSL."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_HF_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    for env_name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ.setdefault(env_name, str(cache))

    for path in (VENDOR_PYTHON, VENDOR_MOMENT_RESEARCH):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_official_moment(
    model_id: str = "AutonLab/MOMENT-1-large",
    device: torch.device | str = "cpu",
    cache_dir: Path | None = None,
    task_name: str = "pre-training",
    seq_len: int = 512,
    n_channels: int = 1,
):
    configure_official_moment_paths(cache_dir)
    from moment.models.moment import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        model_id,
        model_kwargs={
            "task_name": task_name,
            "n_channels": n_channels,
            "seq_len": seq_len,
        },
    )
    model.init()
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_with_official_moment(
    model,
    signals: np.ndarray,
    batch_size: int,
    device: torch.device | str,
    reduction: str = "mean",
) -> np.ndarray:
    """Return MOMENT sequence embeddings for univariate signals.

    The official zero-shot representation script calls `model.embed(...,
    reduction="mean")`; this helper mirrors that path.
    """
    if signals.ndim != 2:
        raise ValueError("signals must have shape (n_samples, seq_len)")

    vectors: list[np.ndarray] = []
    device = torch.device(device)
    for start in range(0, signals.shape[0], batch_size):
        batch_np = signals[start : start + batch_size].astype(np.float32, copy=False)
        batch = torch.from_numpy(batch_np).unsqueeze(1).to(device)
        input_mask = torch.ones((batch.shape[0], batch.shape[-1]), dtype=torch.long, device=device)
        outputs = model.embed(x_enc=batch, input_mask=input_mask, reduction=reduction)
        embeddings = outputs.embeddings
        if embeddings.ndim > 2:
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        vectors.append(embeddings.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(vectors, axis=0)
