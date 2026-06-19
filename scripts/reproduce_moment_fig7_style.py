#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def rng_noise(rng: np.random.Generator, n: int, scale: float = 1.0) -> np.ndarray:
    return rng.normal(0.0, scale, size=n)


def rotate(x: np.ndarray, y: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(degrees)
    xr = x * np.cos(theta) - y * np.sin(theta)
    yr = x * np.sin(theta) + y * np.cos(theta)
    return xr, yr


def trend_panel(rng: np.random.Generator, n: int, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = rng.uniform(0.125, 8.125, n)
    r = (8.3 - c) / 8.0
    theta = rng.uniform(-1.1, 1.15, n)
    x = scale * (0.08 * r * np.cos(theta) + 0.012 * rng_noise(rng, n))
    y = scale * (0.05 * r * np.sin(theta) - 0.018 * (c - 4.0) / 4.0 + 0.012 * rng_noise(rng, n))
    if scale > 20:
        x, y = rotate(x, y, -18)
        x += 0.4 * (8.0 - c)
    return x, y, c


def amplitude_panel(rng: np.random.Generator, n: int, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = rng.uniform(0.25, 4.25, n)
    base = (4.3 - c) / 4.0
    x = scale * (0.10 * base + 0.028 * rng_noise(rng, n))
    y = scale * (0.028 * rng_noise(rng, n))
    outlier = rng.random(n) < 0.08
    x[outlier] += scale * rng.uniform(0.12, 0.55, outlier.sum())
    y[outlier] += scale * rng_noise(rng, outlier.sum(), 0.055)
    if scale > 20:
        x, y = rotate(x, y, -22)
        y += scale * 0.04 * np.sin(c * 2.0)
    return x, y, c


def frequency_panel(rng: np.random.Generator, n: int, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = rng.uniform(1.0, 32.0, n)
    t = (c - 1.0) / 31.0
    x = scale * (0.70 * np.cos(1.55 * np.pi * t + 0.1))
    y = scale * (0.55 * np.sin(1.55 * np.pi * t + 0.1) - 0.05)
    x += scale * rng_noise(rng, n, 0.018)
    y += scale * rng_noise(rng, n, 0.018)
    if scale > 20:
        x = scale * (0.55 * np.sin(2.2 * np.pi * t) - 0.05 + rng_noise(rng, n, 0.018))
        y = scale * (0.75 * np.cos(1.25 * np.pi * t) + 0.15 * np.sin(4 * np.pi * t) + rng_noise(rng, n, 0.018))
    return x, y, c


def baseline_panel(rng: np.random.Generator, n: int, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = rng.uniform(-2.0, 2.0, n)
    radius = np.sqrt(rng.uniform(0.0, 1.0, n))
    theta = rng.uniform(0, 2 * np.pi, n)
    x = scale * 0.13 * radius * np.cos(theta)
    y = scale * 0.13 * radius * np.sin(theta)
    x += scale * rng_noise(rng, n, 0.01)
    y += scale * rng_noise(rng, n, 0.01)
    return x, y, c


def autocorr_panel(rng: np.random.Generator, n: int, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.asarray([1, 2, 3, 5])
    labels = rng.choice(freqs, size=n)
    phase = rng.uniform(0, 2 * np.pi, n)
    if scale < 5:
        centers = {
            1: (0.52, -0.18),
            2: (0.30, 0.10),
            3: (-0.28, 0.24),
            5: (-0.62, -0.20),
        }
        x = np.zeros(n)
        y = np.zeros(n)
        for f, (cx, cy) in centers.items():
            mask = labels == f
            t = phase[mask]
            x[mask] = cx + 0.10 * np.cos(t) + rng_noise(rng, mask.sum(), 0.025)
            y[mask] = cy + 0.05 * np.sin(t) + rng_noise(rng, mask.sum(), 0.025)
    else:
        centers = {
            1: (18, -22),
            2: (20, 20),
            3: (-24, 24),
            5: (-25, -22),
        }
        x = np.zeros(n)
        y = np.zeros(n)
        for f, (cx, cy) in centers.items():
            mask = labels == f
            t = phase[mask]
            x[mask] = cx + 10 * np.cos(t) + rng_noise(rng, mask.sum(), 0.8)
            y[mask] = cy + 10 * np.sin(t) + rng_noise(rng, mask.sum(), 0.8)
    return x, y, phase


def make_panels(seed: int, n: int) -> list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    top = [
        trend_panel(rng, n, 1.0),
        amplitude_panel(rng, n, 1.0),
        frequency_panel(rng, n, 1.0),
        baseline_panel(rng, n, 1.0),
        autocorr_panel(rng, n, 1.0),
    ]
    bottom = [
        trend_panel(rng, n, 350.0),
        amplitude_panel(rng, n, 350.0),
        frequency_panel(rng, n, 55.0),
        baseline_panel(rng, n, 270.0),
        autocorr_panel(rng, n, 55.0),
    ]
    return [top, bottom]


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


def draw_figure(output_pdf: Path, output_png: Path, seed: int, n: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 9,
            "axes.labelsize": 8,
        }
    )
    panels = make_panels(seed, n)
    fig, axes = plt.subplots(2, 5, figsize=(15.3, 6.3), constrained_layout=False)
    cmap = "magma"

    for row in range(2):
        for col in range(5):
            ax = axes[row, col]
            x, y, c = panels[row][col]
            scatter = ax.scatter(x, y, c=c, cmap=cmap, s=9, alpha=0.88, linewidths=0)
            ax.set_title(TITLES[col], pad=4)
            set_axis_style(ax)
            if col == 0:
                add_colorbar(ax, scatter, ticks=[0.125, 1.125, 2.125, 3.125, 4.125, 5.125, 6.125, 7.125, 8.125])
            elif col == 1:
                add_colorbar(ax, scatter, ticks=[0.25, 1.25, 2.25, 3.25, 4.25])
            elif col == 2:
                add_colorbar(ax, scatter, ticks=[1, 5, 9, 13, 17, 21, 25, 29])
            elif col == 3:
                add_colorbar(ax, scatter, ticks=[-2, -1, 0, 1, 2])
            else:
                add_colorbar(ax, scatter, ticks=[0, 1, 2, 3, 4, 5, 6, 7])
                if row == 0:
                    ann = [(3, -0.30, 0.18), (2, 0.33, 0.20), (5, -0.66, -0.22), (1, 0.43, -0.18)]
                else:
                    ann = [(3, -27, 25), (2, 21, 23), (5, -30, -22), (1, 18, -22)]
                for freq, tx, ty in ann:
                    ax.text(tx, ty, rf"$f = {freq}$", color="#b2182b", fontsize=8)

    fig.subplots_adjust(left=0.045, right=0.985, top=0.91, bottom=0.25, wspace=0.32, hspace=0.42)
    for col, subtitle in enumerate(SUBTITLES):
        bbox = axes[1, col].get_position()
        x_center = (bbox.x0 + bbox.x1) / 2.0
        fig.text(x_center, 0.15, subtitle, ha="center", va="center", fontsize=18)

    caption = (
        "Figure 7. What is MOMENT learning? Structure in the PCA (top) and t-SNE (bottom) "
        "visualizations of the embeddings of synthetically generated sinusoids suggest that MOMENT can "
        "capture subtle trend, scale, frequency, and auto-correlation information."
    )
    fig.text(0.045, 0.035, caption, ha="left", va="bottom", fontsize=12, wrap=True)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the visual style of MOMENT Figure 7 for plotting design.")
    parser.add_argument("--output-dir", type=Path, default=Path("P3_SSL/outputs/figure_design"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=1800)
    args = parser.parse_args()

    draw_figure(
        output_pdf=args.output_dir / "moment_fig7_style_reproduction.pdf",
        output_png=args.output_dir / "moment_fig7_style_reproduction.png",
        seed=args.seed,
        n=args.n,
    )
    print(f"Wrote {args.output_dir / 'moment_fig7_style_reproduction.pdf'}")
    print(f"Wrote {args.output_dir / 'moment_fig7_style_reproduction.png'}")


if __name__ == "__main__":
    main()

