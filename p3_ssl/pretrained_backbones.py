from __future__ import annotations

import copy
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as scipy_signal

from .data import ManifestRow, parse_yolo_1d_labels, read_manifest
from .decimation import crop_or_pad, decimate_signal, ensure_1d_signal, normalize_signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_VENDOR = PROJECT_ROOT / "vendor" / "python_pretrained"
DEFAULT_HF_CACHE = PROJECT_ROOT / "outputs" / "hf_cache"

PATCHTST_DEFAULT_ID = "namctin/patchtst_etth1_pretrain"
SWIN_DEFAULT_ID = "microsoft/swin-tiny-patch4-window7-224"
MOMENT_DEFAULT_ID = "AutonLab/MOMENT-1-large"
VENDOR_PYTHON = PROJECT_ROOT / "vendor" / "python"
VENDOR_MOMENT_RESEARCH = PROJECT_ROOT / "vendor" / "moment-research"

CLASS_NAMES = {
    0: "2um",
    1: "4um",
    2: "10um",
    3: "unclear",
}


@dataclass(frozen=True)
class ParticleEvent:
    event_id: str
    sample_id: str
    split: str
    signal_path: str
    label_path: str
    class_id: int
    class_name: str
    center_norm: float
    width_norm: float
    center_index: int
    crop_start: int
    crop_end: int


@dataclass(frozen=True)
class PretrainedTransferReport:
    source_model_id: str
    target_num_input_channels: int
    source_num_input_channels: int
    loaded_keys: int
    skipped_keys: int
    missing_keys: int
    unexpected_keys: int
    skipped: list[dict[str, object]]
    missing: list[str]
    unexpected: list[str]


def configure_pretrained_paths(cache_dir: Path | None = None) -> None:
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_HF_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache))
    vendor = str(PRETRAINED_VENDOR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def load_decimated_signal(
    row: ManifestRow,
    input_length_raw: int,
    decimation_factor: int,
    input_length_ssl: int,
    decimation_method: str = "mean",
) -> np.ndarray:
    signal = ensure_1d_signal(np.load(row.signal_path))
    signal = crop_or_pad(signal, input_length_raw, mode="center")
    signal = decimate_signal(signal, decimation_factor, method=decimation_method)
    signal = crop_or_pad(signal, input_length_ssl, mode="center")
    return signal.astype(np.float32, copy=False)


def collect_particle_events(
    manifest_csv: Path,
    input_length_raw: int = 16384,
    decimation_factor: int = 8,
    input_length_ssl: int = 2048,
    event_length: int = 512,
    normalization: str = "window_zscore",
    decimation_method: str = "mean",
    class_names: dict[int, str] | None = None,
    max_events_per_class: int | None = None,
    seed: int = 42,
) -> tuple[list[ParticleEvent], np.ndarray]:
    names = class_names or CLASS_NAMES
    rows = read_manifest(manifest_csv)
    events: list[ParticleEvent] = []
    crops: list[np.ndarray] = []
    per_sample_count: dict[str, int] = {}

    for row in rows:
        labels = parse_yolo_1d_labels(row.label_path)
        if labels.size == 0:
            continue
        decimated = load_decimated_signal(
            row=row,
            input_length_raw=input_length_raw,
            decimation_factor=decimation_factor,
            input_length_ssl=input_length_ssl,
            decimation_method=decimation_method,
        )
        for class_float, center_norm, width_norm in labels:
            class_id = int(class_float)
            if class_id not in names:
                continue
            center_index = int(round(float(center_norm) * input_length_ssl))
            crop_start = center_index - event_length // 2
            crop_end = crop_start + event_length
            src_start = max(0, crop_start)
            src_end = min(input_length_ssl, crop_end)
            crop = np.zeros(event_length, dtype=np.float32)
            dst_start = src_start - crop_start
            dst_end = dst_start + max(0, src_end - src_start)
            if src_end > src_start:
                crop[dst_start:dst_end] = decimated[src_start:src_end]
            crop = normalize_signal(crop, mode=normalization)
            local_idx = per_sample_count.get(row.sample_id, 0)
            per_sample_count[row.sample_id] = local_idx + 1
            events.append(
                ParticleEvent(
                    event_id=f"{row.sample_id}::{local_idx}",
                    sample_id=row.sample_id,
                    split=row.split,
                    signal_path=str(row.signal_path),
                    label_path="" if row.label_path is None else str(row.label_path),
                    class_id=class_id,
                    class_name=names[class_id],
                    center_norm=float(center_norm),
                    width_norm=float(width_norm),
                    center_index=center_index,
                    crop_start=crop_start,
                    crop_end=crop_end,
                )
            )
            crops.append(crop)

    if not events:
        raise ValueError(f"No labeled particle events found in {manifest_csv}")

    crops_arr = np.stack(crops).astype(np.float32)
    if max_events_per_class is not None and max_events_per_class > 0:
        rng = np.random.default_rng(seed)
        selected: list[int] = []
        class_ids = np.asarray([event.class_id for event in events], dtype=np.int64)
        for class_id in sorted(set(class_ids.tolist())):
            idx = np.flatnonzero(class_ids == class_id)
            if idx.size > max_events_per_class:
                idx = rng.choice(idx, size=max_events_per_class, replace=False)
            selected.extend(int(i) for i in idx)
        selected_arr = np.asarray(selected, dtype=np.int64)
        selected_arr.sort()
        events = [events[int(i)] for i in selected_arr]
        crops_arr = crops_arr[selected_arr]

    return events, crops_arr


def write_event_metadata(path: Path, events: list[ParticleEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = list(asdict(events[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(asdict(event))


def save_transfer_report(path: Path, report: PretrainedTransferReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(asdict(report), f, indent=2, sort_keys=True)


def load_patchtst_1ch_model(
    model_id: str = PATCHTST_DEFAULT_ID,
    cache_dir: Path | None = None,
    device: torch.device | str = "cpu",
):
    configure_pretrained_paths(cache_dir)
    from transformers import AutoConfig, PatchTSTModel

    source_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    source_model = PatchTSTModel.from_pretrained(model_id, cache_dir=cache_dir)
    target_config = copy.deepcopy(source_config)
    target_config.num_input_channels = 1
    target_model = PatchTSTModel(target_config)

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    load_state: dict[str, torch.Tensor] = {}
    skipped: list[dict[str, object]] = []
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            load_state[key] = value
        else:
            skipped.append(
                {
                    "key": key,
                    "source_shape": list(value.shape),
                    "target_shape": None if key not in target_state else list(target_state[key].shape),
                }
            )
    missing, unexpected = target_model.load_state_dict(load_state, strict=False)
    report = PretrainedTransferReport(
        source_model_id=model_id,
        target_num_input_channels=1,
        source_num_input_channels=int(source_config.num_input_channels),
        loaded_keys=len(load_state),
        skipped_keys=len(skipped),
        missing_keys=len(missing),
        unexpected_keys=len(unexpected),
        skipped=skipped,
        missing=list(missing),
        unexpected=list(unexpected),
    )
    target_model.to(device).eval()
    return target_model, report


def load_swin_model(
    model_id: str = SWIN_DEFAULT_ID,
    cache_dir: Path | None = None,
    device: torch.device | str = "cpu",
):
    configure_pretrained_paths(cache_dir)
    from transformers import SwinModel

    model = SwinModel.from_pretrained(model_id, cache_dir=cache_dir)
    model.to(device).eval()
    return model


def configure_moment_with_pretrained_transformers(cache_dir: Path | None = None) -> None:
    configure_pretrained_paths(cache_dir)
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_HF_CACHE
    for env_name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        os.environ.setdefault(env_name, str(cache))
    ordered = [str(VENDOR_MOMENT_RESEARCH), str(PRETRAINED_VENDOR), str(VENDOR_PYTHON)]
    for path_str in ordered:
        while path_str in sys.path:
            sys.path.remove(path_str)
    for path_str in reversed(ordered):
        sys.path.insert(0, path_str)


def load_moment_official_model(
    model_id: str = MOMENT_DEFAULT_ID,
    cache_dir: Path | None = None,
    device: torch.device | str = "cpu",
    seq_len: int = 512,
):
    configure_moment_with_pretrained_transformers(cache_dir)
    from moment.models.moment import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        model_kwargs={"task_name": "pre-training", "seq_len": seq_len, "n_channels": 1},
    )
    model.init()
    model.to(device).eval()
    return model


def signal_to_spectrogram_image(signal: np.ndarray, image_size: int = 224) -> torch.Tensor:
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {x.shape}")
    x = normalize_signal(x, mode="window_zscore")
    _, _, spec = scipy_signal.spectrogram(
        x,
        fs=1.0,
        window="hann",
        nperseg=min(96, max(16, x.size // 4)),
        noverlap=min(72, max(8, x.size // 8)),
        detrend=False,
        scaling="spectrum",
        mode="magnitude",
    )
    spec = np.log1p(spec.astype(np.float32))
    spec = (spec - float(spec.min())) / max(float(spec.max() - spec.min()), 1.0e-6)
    img = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
    img = img.repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (img - mean) / std


def signals_to_spectrogram_images(signals: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    arrays = signals.detach().cpu().numpy()
    images = [signal_to_spectrogram_image(arr, image_size=image_size) for arr in arrays]
    return torch.stack(images, dim=0)


def encode_patchtst_batch(model, signals: torch.Tensor) -> torch.Tensor:
    past_values = signals.unsqueeze(-1)
    outputs = model(past_values=past_values)
    hidden = outputs.last_hidden_state
    return hidden.mean(dim=(1, 2))


def encode_swin_batch(model, signals: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    images = signals_to_spectrogram_images(signals).to(device)
    outputs = model(pixel_values=images)
    if outputs.pooler_output is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state.mean(dim=1)


def encode_moment_official_batch(model, signals: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    x = signals.to(device).unsqueeze(1)
    input_mask = torch.ones((x.shape[0], x.shape[-1]), dtype=torch.long, device=x.device)
    outputs = model.embed(x_enc=x, input_mask=input_mask, reduction="mean")
    return outputs.embeddings


def encode_batch(
    model_key: Literal["moment_official", "patchtst_pretrained", "swin2d_pretrained"],
    model,
    signals: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    signals = signals.to(device)
    if model_key == "moment_official":
        return encode_moment_official_batch(model, signals, device=device)
    if model_key == "patchtst_pretrained":
        return encode_patchtst_batch(model, signals)
    if model_key == "swin2d_pretrained":
        return encode_swin_batch(model, signals, device=device)
    raise ValueError(f"Unsupported model_key: {model_key}")
