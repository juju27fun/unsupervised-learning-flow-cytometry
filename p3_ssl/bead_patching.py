from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from .decimation import normalize_signal
from .masking import PatchSpec, build_patch_aligned_isolated_masks, mask_spans


SAMPLING_FREQUENCY_HZ = 1_000_000.0


@dataclass(frozen=True)
class BeadPatchingConfig:
    input_length: int = 4096
    patch_size: int = 16
    patch_stride: int = 16
    mask_ratio: float = 0.25
    high_derivative_probability: float = 0.25
    minimum_visible_tokens_between_masks: int = 1
    seed: int = 42
    normalization: str = "window_zscore"


def load_simulation_example(
    root: Path,
    *,
    latent_id: str,
    view_index: int,
) -> tuple[np.ndarray, dict[str, str]]:
    metadata_path = root / "simulation_metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        row = next(
            (
                candidate
                for candidate in csv.DictReader(handle)
                if candidate["latent_id"] == latent_id
                and int(candidate["view_index"]) == view_index
            ),
            None,
        )
    if row is None:
        raise ValueError(
            f"Simulation example not found: {latent_id}:view-{view_index}"
        )
    signals = np.load(root / "signals.npy", mmap_mode="r")
    signal = np.asarray(signals[int(row["signal_row"])], dtype=np.float32)
    return signal.copy(), row


def build_patching_example(
    signal: np.ndarray,
    metadata: dict[str, str],
    config: BeadPatchingConfig = BeadPatchingConfig(),
) -> dict[str, Any]:
    values = np.asarray(signal, dtype=np.float32)
    if values.shape != (config.input_length,):
        raise ValueError(
            f"Expected signal shape {(config.input_length,)}, got {values.shape}"
        )
    normalized = normalize_signal(values, mode=config.normalization)
    spec = PatchSpec(
        input_length=config.input_length,
        patch_size=config.patch_size,
        patch_stride=config.patch_stride,
    )
    masks = build_patch_aligned_isolated_masks(
        normalized,
        spec,
        np.random.default_rng(config.seed),
        mask_ratio=config.mask_ratio,
        event_mask=None,
        event_biased_probability=0.0,
        high_derivative_probability=config.high_derivative_probability,
        minimum_visible_tokens_between_masks=(
            config.minimum_visible_tokens_between_masks
        ),
    )
    center = float(metadata["event_position_fraction"]) * (
        config.input_length - 1
    )
    half_width = (
        float(metadata["duration_ms"])
        / 1000.0
        * SAMPLING_FREQUENCY_HZ
        / 2.0
    )
    event_start = max(0, int(round(center - half_width)))
    event_end = min(config.input_length, int(round(center + half_width)))
    token_mask = np.asarray(masks["token_mask"], dtype=bool)
    target_mask = np.asarray(masks["target_time_mask"], dtype=bool)
    return {
        "signal": normalized,
        "metadata": dict(metadata),
        "config": config,
        "spec": spec,
        "token_mask": token_mask,
        "target_time_mask": target_mask,
        "target_spans": mask_spans(target_mask),
        "event_bounds": (event_start, event_end),
        "n_masked_tokens": int(token_mask.sum()),
        "n_tokens": spec.n_tokens,
        "masked_sample_fraction": float(np.mean(target_mask)),
    }


def serializable_patching_example(example: dict[str, Any]) -> dict[str, Any]:
    config: BeadPatchingConfig = example["config"]
    return {
        "sample": {
            "latent_id": example["metadata"]["latent_id"],
            "view_index": int(example["metadata"]["view_index"]),
            "signal_row": int(example["metadata"]["signal_row"]),
            "duration_ms": float(example["metadata"]["duration_ms"]),
            "doppler_khz": float(example["metadata"]["doppler_khz"]),
            "event_position_fraction": float(
                example["metadata"]["event_position_fraction"]
            ),
        },
        "input": {
            "length": config.input_length,
            "sampling_frequency_hz": SAMPLING_FREQUENCY_HZ,
            "normalization": config.normalization,
        },
        "patching": {
            "patch_size": config.patch_size,
            "patch_stride": config.patch_stride,
            "n_tokens": example["n_tokens"],
            "patch_duration_us": (
                config.patch_size / SAMPLING_FREQUENCY_HZ * 1.0e6
            ),
        },
        "masking": {
            "policy": "P25",
            "mask_ratio_requested": config.mask_ratio,
            "masked_sample_fraction": example["masked_sample_fraction"],
            "n_masked_tokens": example["n_masked_tokens"],
            "masked_token_indices": np.flatnonzero(
                example["token_mask"]
            ).tolist(),
            "target_spans": [
                [int(start), int(end)]
                for start, end in example["target_spans"]
            ],
            "high_derivative_probability": (
                config.high_derivative_probability
            ),
            "event_biased_probability": 0.0,
            "minimum_visible_tokens_between_masks": (
                config.minimum_visible_tokens_between_masks
            ),
            "seed": config.seed,
        },
        "event_bounds": list(example["event_bounds"]),
        "claim_boundary": (
            "The shaded event support is shown for orientation only. It is not "
            "used to select P25 masks; the mask policy is label-free."
        ),
    }


def render_patching_figure(
    example: dict[str, Any],
    destination: Path,
) -> None:
    signal = np.asarray(example["signal"])
    target_mask = np.asarray(example["target_time_mask"])
    token_mask = np.asarray(example["token_mask"])
    spec: PatchSpec = example["spec"]
    event_start, event_end = example["event_bounds"]
    time_ms = np.arange(signal.size) / SAMPLING_FREQUENCY_HZ * 1000.0
    event_center = (event_start + event_end) / 2.0
    zoom_half_width = max(480, int((event_end - event_start) * 1.2))
    zoom_start = max(0, int(round(event_center - zoom_half_width)))
    zoom_end = min(signal.size, int(round(event_center + zoom_half_width)))

    figure = Figure(figsize=(15.0, 10.0), constrained_layout=True)
    grid = figure.add_gridspec(4, 1, height_ratios=(1.0, 1.2, 1.0, 0.35))
    full_axis = figure.add_subplot(grid[0])
    zoom_axis = figure.add_subplot(grid[1])
    visible_axis = figure.add_subplot(grid[2])
    token_axis = figure.add_subplot(grid[3])

    full_axis.plot(time_ms, signal, color="#2c7fb8", linewidth=0.8)
    full_axis.axvspan(
        event_start / 1000.0,
        event_end / 1000.0,
        color="#74c476",
        alpha=0.18,
        label="Simulated event support (orientation only)",
    )
    full_axis.set_title(
        "1. The model input is one 4.096 ms normalized simulated trace",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    full_axis.set_ylabel("Normalized amplitude")
    full_axis.legend(frameon=False, loc="upper right")
    full_axis.grid(alpha=0.15)

    zoom_axis.plot(time_ms, signal, color="#2c7fb8", linewidth=1.0)
    for token_index, (start, end) in enumerate(spec.spans):
        if end <= zoom_start or start >= zoom_end:
            continue
        if token_mask[token_index]:
            zoom_axis.axvspan(
                start / 1000.0,
                end / 1000.0,
                color="#fdae61",
                alpha=0.72,
            )
        else:
            zoom_axis.axvline(
                start / 1000.0,
                color="#9e9e9e",
                linewidth=0.35,
                alpha=0.55,
            )
    zoom_axis.set_xlim(zoom_start / 1000.0, zoom_end / 1000.0)
    zoom_axis.set_title(
        "2. P25 hides isolated complete 16 µs patches while neighbours stay visible",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    zoom_axis.set_ylabel("Normalized amplitude")
    zoom_axis.grid(alpha=0.12)

    visible = signal.copy()
    visible[target_mask] = np.nan
    visible_axis.plot(
        time_ms,
        visible,
        color="#2c7fb8",
        linewidth=0.85,
        label="Visible context",
    )
    target_values = np.ma.masked_where(~target_mask, signal)
    visible_axis.plot(
        time_ms,
        target_values,
        color="#e6550d",
        linewidth=1.0,
        label="Hidden reconstruction targets",
    )
    visible_axis.set_title(
        "3. Only blue context reaches the encoder; orange targets drive the loss",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    visible_axis.set_ylabel("Normalized amplitude")
    visible_axis.set_xlabel("Time (ms)")
    visible_axis.legend(frameon=False, loc="upper right")
    visible_axis.grid(alpha=0.15)

    token_strip = np.where(token_mask, 1.0, 0.0)[None, :]
    from matplotlib.colors import ListedColormap

    token_axis.imshow(
        token_strip,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#9ecae1", "#fdae61"]),
        vmin=0.0,
        vmax=1.0,
    )
    token_axis.set_yticks([])
    token_axis.set_xticks([0, 63, 127, 191, 255])
    token_axis.set_xticklabels(["1", "64", "128", "192", "256"])
    token_axis.set_xlabel("Patch token index")
    token_axis.set_title(
        f"Exact mask: {example['n_masked_tokens']}/{example['n_tokens']} "
        "tokens hidden; hidden positions receive a learned mask token",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )

    figure.suptitle(
        "What the new bead SSL model actually sees",
        fontsize=20,
        fontweight="bold",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=170, facecolor="white")
