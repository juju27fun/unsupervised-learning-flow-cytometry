from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .masking import PatchSpec, build_ssl_masks, token_mask_to_time_mask


def _event_spans_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    active = np.flatnonzero(mask)
    if active.size == 0:
        return spans
    start = int(active[0])
    prev = int(active[0])
    for idx in active[1:]:
        idx = int(idx)
        if idx != prev + 1:
            spans.append((start, prev + 1))
            start = idx
        prev = idx
    spans.append((start, prev + 1))
    return spans


def add_patch_stride_page(
    pdf: PdfPages,
    signal: np.ndarray,
    sample_id: str,
    patch_size: int,
    patch_stride: int,
    guard_points: int = 8,
    event_mask: np.ndarray | None = None,
    seed: int = 42,
) -> None:
    spec = PatchSpec(len(signal), patch_size, patch_stride)
    rng = np.random.default_rng(seed)
    masks = build_ssl_masks(
        signal=signal,
        spec=spec,
        rng=rng,
        guard_points=guard_points,
    )
    token_hidden_time = token_mask_to_time_mask(masks["token_mask"], spec)

    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    x = np.arange(len(signal))
    primary = " | primary full-window 4096" if len(signal) == 4096 and patch_size == 4 and patch_stride == 4 else ""
    axes[0].plot(x, signal, linewidth=0.8, color="black")
    axes[0].set_title(f"{sample_id} | patch={patch_size}, stride={patch_stride}, tokens={spec.n_tokens}{primary}")
    axes[0].set_ylabel("signal")
    if event_mask is not None:
        for start, end in _event_spans_from_mask(event_mask):
            axes[0].axvspan(start, end, color="tab:green", alpha=0.18)

    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([])
    axes[1].set_ylabel("patches")
    for i, (start, end) in enumerate(spec.spans):
        color = "tab:blue" if i % 2 == 0 else "tab:cyan"
        axes[1].axvspan(start, end, ymin=0.15, ymax=0.85, color=color, alpha=0.25)

    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_ylabel("masks")
    for start, end in masks["blocks"]:
        axes[2].axvspan(start, end, color="tab:red", alpha=0.55, label="loss target")
    hidden = masks["hidden_time_mask"]
    for start, end in _event_spans_from_mask(hidden):
        axes[2].axvspan(start, end, color="tab:orange", alpha=0.18)

    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    axes[3].set_ylabel("hidden tokens")
    for start, end in _event_spans_from_mask(token_hidden_time):
        axes[3].axvspan(start, end, color="tab:purple", alpha=0.35)
    axes[3].set_xlabel("decimated sample index")
    pdf.savefig(fig)
    plt.close(fig)


def write_patch_stride_audit_pdf(
    output_pdf: str | Path,
    samples: list[dict[str, np.ndarray | str]],
    configs: list[dict[str, int]],
    guard_points: int = 8,
    seed: int = 42,
) -> None:
    output = Path(output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for sample_index, sample in enumerate(samples):
            signal = np.asarray(sample["signal"], dtype=np.float32)
            event_mask = sample.get("event_mask")
            for cfg in configs:
                add_patch_stride_page(
                    pdf=pdf,
                    signal=signal,
                    sample_id=str(sample.get("sample_id", f"sample_{sample_index}")),
                    patch_size=int(cfg["patch_size"]),
                    patch_stride=int(cfg["patch_stride"]),
                    guard_points=guard_points,
                    event_mask=np.asarray(event_mask, dtype=bool) if event_mask is not None else None,
                    seed=seed + sample_index,
                )
