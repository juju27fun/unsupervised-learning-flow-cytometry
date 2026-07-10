#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p3_ssl.config import load_config
from p3_ssl.decimation import normalize_signal
from p3_ssl.models import MomentLikeConfig, MomentLikeReconstructor
from p3_ssl.official_moment import encode_with_official_moment, load_official_moment


TITLES = [
    r"$y = x^c + \sin(32\pi x) + \epsilon$",
    r"$y = c * \sin(32\pi x) + \epsilon$",
    r"$y = \sin(2c\pi x) + \epsilon$",
    r"$y = c + \sin(32\pi x) + \epsilon$",
    r"$y = \sin(2\pi f x + c) + \epsilon,\ c \in [0, 2\pi]$",
]

SUBTITLES = [
    r"$(i)$ Trend",
    r"$(ii)$ Amplitude",
    r"$(iii)$ Frequency",
    r"$(iv)$ Baseline Shift",
    r"$(v)$ Auto-correlation",
]

PANEL_KEYS = ["trend", "amplitude", "frequency", "baseline_shift", "autocorrelation"]


@dataclass(frozen=True)
class SyntheticPanel:
    key: str
    signal: np.ndarray
    color_value: np.ndarray
    c_value: np.ndarray
    f_value: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _noise(rng: np.random.Generator, shape: tuple[int, int], std: float) -> np.ndarray:
    return rng.normal(0.0, std, size=shape).astype(np.float32)


def generate_trend(rng: np.random.Generator, n: int, length: int, noise_std: float) -> SyntheticPanel:
    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    c = rng.uniform(1.0 / 8.0, 8.0, size=n).astype(np.float32)
    signal = np.power(x[None, :], c[:, None]) + np.sin(32.0 * np.pi * x)[None, :]
    signal = signal.astype(np.float32) + _noise(rng, (n, length), noise_std)
    return SyntheticPanel("trend", signal, c, c, np.full(n, np.nan, dtype=np.float32))


def generate_amplitude(rng: np.random.Generator, n: int, length: int, noise_std: float) -> SyntheticPanel:
    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    c = rng.uniform(1.0 / 4.0, 4.0, size=n).astype(np.float32)
    signal = c[:, None] * np.sin(32.0 * np.pi * x)[None, :]
    signal = signal.astype(np.float32) + _noise(rng, (n, length), noise_std)
    return SyntheticPanel("amplitude", signal, c, c, np.full(n, np.nan, dtype=np.float32))


def generate_frequency(rng: np.random.Generator, n: int, length: int, noise_std: float) -> SyntheticPanel:
    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    c = rng.uniform(1.0, 32.0, size=n).astype(np.float32)
    signal = np.sin(2.0 * c[:, None] * np.pi * x[None, :])
    signal = signal.astype(np.float32) + _noise(rng, (n, length), noise_std)
    return SyntheticPanel("frequency", signal, c, c, np.full(n, np.nan, dtype=np.float32))


def generate_baseline_shift(rng: np.random.Generator, n: int, length: int, noise_std: float) -> SyntheticPanel:
    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    c = rng.uniform(-2.0, 2.0, size=n).astype(np.float32)
    signal = c[:, None] + np.sin(32.0 * np.pi * x)[None, :]
    signal = signal.astype(np.float32) + _noise(rng, (n, length), noise_std)
    return SyntheticPanel("baseline_shift", signal, c, c, np.full(n, np.nan, dtype=np.float32))


def generate_autocorrelation(rng: np.random.Generator, n: int, length: int, noise_std: float) -> SyntheticPanel:
    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    freqs = np.asarray([1, 2, 3, 5], dtype=np.float32)
    base = n // len(freqs)
    rem = n - base * len(freqs)
    f = np.concatenate([np.full(base + (1 if i < rem else 0), freq, dtype=np.float32) for i, freq in enumerate(freqs)])
    rng.shuffle(f)
    c = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
    signal = np.sin(2.0 * np.pi * f[:, None] * x[None, :] + c[:, None])
    signal = signal.astype(np.float32) + _noise(rng, (n, length), noise_std)
    return SyntheticPanel("autocorrelation", signal, c, c, f)


def generate_synthetic_panels(
    n_per_panel: int,
    length: int,
    seed: int,
    noise_std: float = 0.1,
) -> list[SyntheticPanel]:
    rng = np.random.default_rng(seed)
    return [
        generate_trend(rng, n_per_panel, length, noise_std),
        generate_amplitude(rng, n_per_panel, length, noise_std),
        generate_frequency(rng, n_per_panel, length, noise_std),
        generate_baseline_shift(rng, n_per_panel, length, noise_std),
        generate_autocorrelation(rng, n_per_panel, length, noise_std),
    ]


def normalize_batch(signals: np.ndarray, mode: str) -> np.ndarray:
    return np.stack([normalize_signal(row, mode=mode) for row in signals]).astype(np.float32)


def make_p3_ssl_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> MomentLikeReconstructor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("config", config)
    data_cfg = model_config["data"]
    patch_cfg = model_config["patching"]
    net_cfg = model_config["model"]
    model = MomentLikeReconstructor(
        MomentLikeConfig(
            input_length=int(data_cfg["input_length_ssl"]),
            patch_size=int(patch_cfg["patch_size"]),
            patch_stride=int(patch_cfg["patch_stride"]),
            d_model=int(net_cfg["d_model"]),
            n_heads=int(net_cfg["n_heads"]),
            n_layers=int(net_cfg["n_layers"]),
            dim_feedforward=int(net_cfg["dim_feedforward"]),
            dropout=float(net_cfg.get("dropout", 0.1)),
            activation=str(net_cfg.get("activation", "gelu")),
            max_tokens=int(net_cfg.get("max_tokens", 1024)),
        )
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def encode_sequence_embeddings(
    model: MomentLikeReconstructor,
    signals: np.ndarray,
    normalization: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    normalized = normalize_batch(signals, mode=normalization)
    vectors: list[np.ndarray] = []
    for start in range(0, normalized.shape[0], batch_size):
        batch = torch.from_numpy(normalized[start : start + batch_size]).float().unsqueeze(1).to(device)
        tokens = model.encode(batch, token_mask=None)
        pooled = tokens.mean(dim=1)
        vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(vectors, axis=0)


def reduce_embeddings(embeddings: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x = StandardScaler().fit_transform(embeddings)
    n_samples, n_features = x.shape
    pca = PCA(n_components=2, random_state=seed)
    pca_coords = pca.fit_transform(x).astype(np.float32)
    if n_samples < 5:
        return pca_coords, pca_coords.copy(), {
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "trustworthiness": float("nan"),
        }
    pre_dim = min(50, n_features, n_samples - 1)
    x_pre = PCA(n_components=pre_dim, random_state=seed).fit_transform(x) if pre_dim < n_features else x
    perplexity = min(30, max(2, (n_samples - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    tsne_coords = tsne.fit_transform(x_pre).astype(np.float32)
    trust_neighbors = min(10, max(1, (n_samples // 2) - 1))
    trust = float(trustworthiness(x, tsne_coords, n_neighbors=trust_neighbors)) if trust_neighbors >= 1 else float("nan")
    return pca_coords, tsne_coords, {
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "trustworthiness": trust,
        "tsne_perplexity": float(perplexity),
    }


def add_colorbar(ax: plt.Axes, artist: plt.Collection, ticks: list[float] | None = None) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cb = ax.figure.colorbar(artist, cax=cax)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=6, length=2)


def set_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#555555")
    ax.grid(False)


def colorbar_ticks(key: str) -> list[float] | None:
    if key == "trend":
        return [0.125, 1.125, 2.125, 3.125, 4.125, 5.125, 6.125, 7.125, 8.0]
    if key == "amplitude":
        return [0.25, 1.25, 2.25, 3.25, 4.0]
    if key == "frequency":
        return [1, 5, 9, 13, 17, 21, 25, 29, 32]
    if key == "baseline_shift":
        return [-2, -1, 0, 1, 2]
    return [0, 1, 2, 3, 4, 5, 6]


def annotate_autocorrelation(ax: plt.Axes, coords: np.ndarray, panel: SyntheticPanel) -> None:
    for freq in [1, 2, 3, 5]:
        mask = panel.f_value == float(freq)
        if not np.any(mask):
            continue
        center = np.median(coords[mask], axis=0)
        ax.text(center[0], center[1], rf"$f = {freq}$", color="#b2182b", fontsize=8)


def plot_synthetic_figure(
    panels: list[SyntheticPanel],
    reductions: dict[str, dict[str, np.ndarray]],
    output_pdf: Path,
    output_png: Path,
    caption_backend: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 9,
            "axes.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(2, 5, figsize=(15.3, 6.3), constrained_layout=False)
    cmap = "magma"
    for col, panel in enumerate(panels):
        for row, reduction_key in enumerate(["pca", "tsne"]):
            ax = axes[row, col]
            coords = reductions[panel.key][reduction_key]
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=panel.color_value,
                cmap=cmap,
                s=9,
                alpha=0.88,
                linewidths=0,
            )
            ax.set_title(TITLES[col], pad=4)
            set_axis_style(ax)
            add_colorbar(ax, scatter, ticks=colorbar_ticks(panel.key))
            if panel.key == "autocorrelation":
                annotate_autocorrelation(ax, coords, panel)

    fig.subplots_adjust(left=0.045, right=0.985, top=0.91, bottom=0.25, wspace=0.32, hspace=0.42)
    for col, subtitle in enumerate(SUBTITLES):
        bbox = axes[1, col].get_position()
        fig.text((bbox.x0 + bbox.x1) / 2.0, 0.15, subtitle, ha="center", va="center", fontsize=18)

    caption = (
        f"Synthetic Figure 7 reproduction with {caption_backend}. "
        "Signals are generated from the displayed equations, encoded by the selected model, "
        "pooled at sequence level, then visualized with PCA (top) and t-SNE (bottom)."
    )
    fig.text(0.045, 0.035, caption, ha="left", va="bottom", fontsize=12, wrap=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def write_metadata_csv(output_csv: Path, panels: list[SyntheticPanel]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["panel", "index", "c", "color_value", "f"])
        writer.writeheader()
        for panel in panels:
            for i in range(panel.signal.shape[0]):
                writer.writerow(
                    {
                        "panel": panel.key,
                        "index": i,
                        "c": float(panel.c_value[i]),
                        "color_value": float(panel.color_value[i]),
                        "f": "" if np.isnan(panel.f_value[i]) else float(panel.f_value[i]),
                    }
                )


def save_npz_outputs(
    output_dir: Path,
    panels: list[SyntheticPanel],
    embeddings: dict[str, np.ndarray],
    reductions: dict[str, dict[str, np.ndarray]],
) -> None:
    signal_payload: dict[str, np.ndarray] = {}
    embedding_payload: dict[str, np.ndarray] = {}
    for panel in panels:
        signal_payload[f"{panel.key}_signals"] = panel.signal.astype(np.float32)
        signal_payload[f"{panel.key}_c"] = panel.c_value.astype(np.float32)
        signal_payload[f"{panel.key}_color"] = panel.color_value.astype(np.float32)
        signal_payload[f"{panel.key}_f"] = panel.f_value.astype(np.float32)
        embedding_payload[f"{panel.key}_embeddings"] = embeddings[panel.key].astype(np.float32)
        embedding_payload[f"{panel.key}_pca"] = reductions[panel.key]["pca"].astype(np.float32)
        embedding_payload[f"{panel.key}_tsne"] = reductions[panel.key]["tsne"].astype(np.float32)
    np.savez_compressed(output_dir / "synthetic_signals.npz", **signal_payload)
    np.savez_compressed(output_dir / "embeddings.npz", **embedding_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real synthetic MOMENT Fig. 7 embedding pipeline.")
    parser.add_argument("--backend", choices=["p3_ssl", "official_moment"], default="p3_ssl")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-per-panel", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-id", default="AutonLab/MOMENT-1-large")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs" / "hf_cache")
    parser.add_argument("--input-length", type=int, default=None)
    parser.add_argument("--embedding-reduction", default="mean")
    args = parser.parse_args()

    set_seed(args.seed)
    config: dict[str, Any] | None = None
    normalization = "none"
    if args.backend == "p3_ssl":
        if args.config is None or args.checkpoint is None:
            parser.error("--config and --checkpoint are required for --backend p3_ssl")
        config = load_config(args.config)
        length = int(args.input_length or config["data"]["input_length_ssl"])
        normalization = str(config["data"].get("normalization", "window_zscore"))
    else:
        length = int(args.input_length or 4096)

    panels = generate_synthetic_panels(args.n_per_panel, length, args.seed, noise_std=args.noise_std)

    device = torch.device(args.device)
    if args.backend == "p3_ssl":
        assert config is not None and args.checkpoint is not None
        model = make_p3_ssl_model(config, args.checkpoint, device)
        caption_backend = "P3 SSL MOMENT-like embeddings"
    else:
        model = load_official_moment(
            model_id=args.model_id,
            device=device,
            cache_dir=args.cache_dir,
            seq_len=length,
        )
        caption_backend = f"official MOMENT embeddings ({args.model_id}, reduction={args.embedding_reduction})"

    embeddings: dict[str, np.ndarray] = {}
    reductions: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for panel in panels:
        if args.backend == "p3_ssl":
            emb = encode_sequence_embeddings(
                model=model,
                signals=panel.signal,
                normalization=normalization,
                batch_size=args.batch_size,
                device=device,
            )
        else:
            emb = encode_with_official_moment(
                model=model,
                signals=panel.signal,
                batch_size=args.batch_size,
                device=device,
                reduction=args.embedding_reduction,
            )
        pca_coords, tsne_coords, panel_metrics = reduce_embeddings(emb, seed=args.seed)
        embeddings[panel.key] = emb
        reductions[panel.key] = {"pca": pca_coords, "tsne": tsne_coords}
        metrics[panel.key] = panel_metrics

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_npz_outputs(args.output_dir, panels, embeddings, reductions)
    write_metadata_csv(args.output_dir / "synthetic_metadata.csv", panels)
    with (args.output_dir / "reduction_metrics.json").open("w") as f:
        json.dump(
            {
                "n_per_panel": int(args.n_per_panel),
                "length": int(length),
                "seed": int(args.seed),
                "noise_std": float(args.noise_std),
                "backend": args.backend,
                "checkpoint": "" if args.checkpoint is None else str(args.checkpoint),
                "model_id": args.model_id if args.backend == "official_moment" else "",
                "cache_dir": str(args.cache_dir) if args.backend == "official_moment" else "",
                "embedding_reduction": args.embedding_reduction,
                "metrics": metrics,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    plot_synthetic_figure(
        panels=panels,
        reductions=reductions,
        output_pdf=args.output_dir / "synthetic_moment_fig7_pca_tsne.pdf",
        output_png=args.output_dir / "synthetic_moment_fig7_pca_tsne.png",
        caption_backend=caption_backend,
    )
    print(f"Wrote synthetic Figure 7 pipeline outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
