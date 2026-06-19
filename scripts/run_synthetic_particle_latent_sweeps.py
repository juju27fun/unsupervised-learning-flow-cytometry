#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
P0_ROOT = REPO_ROOT / "P0"
for path_entry in (ROOT, P0_ROOT):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

from p3_ssl.decimation import normalize_signal
from p3_ssl.pretrained_backbones import (
    MOMENT_DEFAULT_ID,
    PATCHTST_DEFAULT_ID,
    encode_batch as encode_pretrained_batch,
    load_moment_official_model,
    load_patchtst_1ch_model,
)


MODEL_DISPLAY = {
    "moment_official": "MOMENT official pretrained",
    "patchtst_pretrained": "PatchTST HF pretrained",
    "conv1dgap_same_input_3class": "Conv1D-GAP-L supervised same-input",
}

DEFAULT_CONV_CHECKPOINT = (
    ROOT
    / "outputs"
    / "pretrained_backbones"
    / "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap"
    / "conv1dgap_same_input_3class"
    / "best_model.pt"
)

SINGLE_REFERENCE = {
    "A": 1.0,
    "fD": 16.0,
    "phi": 0.0,
    "t0": 0.5,
    "tau": 0.08,
}

SINGLE_RANGES = {
    "A": (0.25, 2.0),
    "fD": (1.0, 32.0),
    "phi": (0.0, 2.0 * np.pi),
    "t0": (0.15, 0.85),
    "tau": (0.02, 0.18),
}

DOUBLE_REFERENCE = {
    "A": 1.0,
    "B": 1.0,
    "fDA": 16.0,
    "fDB": 16.0,
    "phiA": 0.0,
    "phiB": 0.0,
    "t0_center": 0.5,
    "tauA": 0.08,
    "tauB": 0.08,
}

DOUBLE_RANGES = {
    "delta_t0": (0.0, 0.24),
    "delta_fD": (-8.0, 8.0),
    "delta_phi": (0.0, 2.0 * np.pi),
    "ratio_BA": (0.25, 2.0),
}

SINGLE_TITLES = {
    "A": r"$A$",
    "fD": r"$f_D$",
    "phi": r"$\phi$",
    "t0": r"$t_0$",
    "tau": r"$\tau$",
}

DOUBLE_TITLES = {
    "delta_t0": r"$\Delta t_0$",
    "delta_fD": r"$\Delta f_D$",
    "delta_phi": r"$\Delta \phi$",
    "ratio_BA": r"$B/A$",
}


@dataclass(frozen=True)
class ParticleSweepPanel:
    key: str
    family: Literal["single", "double"]
    swept_parameter: str
    signal: np.ndarray
    color_value: np.ndarray
    parameters: dict[str, np.ndarray]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _noise(rng: np.random.Generator, shape: tuple[int, int], std: float) -> np.ndarray:
    if std <= 0.0:
        return np.zeros(shape, dtype=np.float32)
    return rng.normal(0.0, std, size=shape).astype(np.float32)


def particle_signal(
    t: np.ndarray,
    A: np.ndarray,
    fD: np.ndarray,
    phi: np.ndarray,
    t0: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    carrier = A[:, None] * np.cos(2.0 * np.pi * fD[:, None] * t[None, :] + phi[:, None])
    envelope = np.exp(-np.square(t[None, :] - t0[:, None]) / (2.0 * np.square(tau[:, None])))
    return (carrier * envelope).astype(np.float32)


def _constant(n: int, value: float) -> np.ndarray:
    return np.full(n, float(value), dtype=np.float32)


def generate_single_particle_panel(
    rng: np.random.Generator,
    swept_parameter: str,
    n: int,
    length: int,
    noise_std: float,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> ParticleSweepPanel:
    ranges = SINGLE_RANGES if ranges is None else ranges
    if swept_parameter not in ranges:
        raise ValueError(f"Unsupported single-particle parameter: {swept_parameter}")
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    values = np.linspace(*ranges[swept_parameter], n, dtype=np.float32)
    params = {name: _constant(n, value) for name, value in SINGLE_REFERENCE.items()}
    params[swept_parameter] = values
    signal = particle_signal(t=t, A=params["A"], fD=params["fD"], phi=params["phi"], t0=params["t0"], tau=params["tau"])
    signal = signal + _noise(rng, signal.shape, noise_std)
    return ParticleSweepPanel(
        key=f"single_{swept_parameter}",
        family="single",
        swept_parameter=swept_parameter,
        signal=signal.astype(np.float32),
        color_value=values.astype(np.float32),
        parameters=params,
    )


def generate_double_particle_panel(
    rng: np.random.Generator,
    swept_parameter: str,
    n: int,
    length: int,
    noise_std: float,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> ParticleSweepPanel:
    ranges = DOUBLE_RANGES if ranges is None else ranges
    if swept_parameter not in ranges:
        raise ValueError(f"Unsupported double-particle parameter: {swept_parameter}")
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    values = np.linspace(*ranges[swept_parameter], n, dtype=np.float32)
    params = {name: _constant(n, value) for name, value in DOUBLE_REFERENCE.items()}

    delta_t0 = values if swept_parameter == "delta_t0" else _constant(n, 0.08)
    delta_fD = values if swept_parameter == "delta_fD" else _constant(n, 0.0)
    delta_phi = values if swept_parameter == "delta_phi" else _constant(n, 0.0)
    ratio_ba = values if swept_parameter == "ratio_BA" else _constant(n, 1.0)

    params["A"] = _constant(n, DOUBLE_REFERENCE["A"])
    params["B"] = params["A"] * ratio_ba
    params["fDA"] = _constant(n, DOUBLE_REFERENCE["fDA"])
    params["fDB"] = params["fDA"] + delta_fD
    params["phiA"] = _constant(n, DOUBLE_REFERENCE["phiA"])
    params["phiB"] = params["phiA"] + delta_phi
    params["t0A"] = params["t0_center"] - 0.5 * delta_t0
    params["t0B"] = params["t0_center"] + 0.5 * delta_t0
    params["tauA"] = _constant(n, DOUBLE_REFERENCE["tauA"])
    params["tauB"] = _constant(n, DOUBLE_REFERENCE["tauB"])
    params["delta_t0"] = delta_t0.astype(np.float32)
    params["delta_fD"] = delta_fD.astype(np.float32)
    params["delta_phi"] = delta_phi.astype(np.float32)
    params["ratio_BA"] = ratio_ba.astype(np.float32)

    signal_a = particle_signal(t, params["A"], params["fDA"], params["phiA"], params["t0A"], params["tauA"])
    signal_b = particle_signal(t, params["B"], params["fDB"], params["phiB"], params["t0B"], params["tauB"])
    signal = signal_a + signal_b + _noise(rng, signal_a.shape, noise_std)
    return ParticleSweepPanel(
        key=f"double_{swept_parameter}",
        family="double",
        swept_parameter=swept_parameter,
        signal=signal.astype(np.float32),
        color_value=values.astype(np.float32),
        parameters=params,
    )


def generate_particle_sweep_panels(
    n_per_panel: int,
    length: int,
    seed: int,
    noise_std: float = 0.0,
    include_single: bool = True,
    include_double: bool = True,
    single_ranges: dict[str, tuple[float, float]] | None = None,
    double_ranges: dict[str, tuple[float, float]] | None = None,
) -> list[ParticleSweepPanel]:
    rng = np.random.default_rng(seed)
    panels: list[ParticleSweepPanel] = []
    if include_single:
        panels.extend(
            generate_single_particle_panel(rng, name, n_per_panel, length, noise_std, ranges=single_ranges)
            for name in ["A", "fD", "phi", "t0", "tau"]
        )
    if include_double:
        panels.extend(
            generate_double_particle_panel(rng, name, n_per_panel, length, noise_std, ranges=double_ranges)
            for name in ["delta_t0", "delta_fD", "delta_phi", "ratio_BA"]
        )
    return panels


def normalize_batch(signals: np.ndarray, mode: str) -> np.ndarray:
    return np.stack([normalize_signal(row, mode=mode) for row in signals]).astype(np.float32)


def model_input_signals(model_key: str, signals: np.ndarray, normalization: str) -> np.ndarray:
    if model_key == "moment_official":
        return signals.astype(np.float32, copy=False)
    return normalize_batch(signals, mode=normalization)


def encode_conv1dgap_features(model: nn.Module, signals: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    x = signals.to(device).unsqueeze(1)
    x = torch.relu(model.bn1(model.conv1(x)))
    x = model.drop1(model.pool1(x))
    x = torch.relu(model.bn2(model.conv2(x)))
    x = model.drop2(model.pool2(x))
    x = torch.relu(model.bn3(model.conv3(x)))
    x = model.drop3(model.pool3(x))
    x = model.flatten(model.gap(x))
    return torch.relu(model.fc1(x))


def load_conv1dgap_same_input_model(
    checkpoint_path: Path,
    device: torch.device | str,
    model_name: str = "Conv1DGAP-L",
    input_length: int = 512,
) -> nn.Module:
    from models import create_model

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing Conv1D-GAP checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    class_names = state.get("class_names", ("2um", "4um", "10um")) if isinstance(state, dict) else ("2um", "4um", "10um")
    saved_input_length = int(state.get("input_length", input_length)) if isinstance(state, dict) else int(input_length)
    saved_model_name = str(state.get("model_name", model_name)) if isinstance(state, dict) else model_name
    model = create_model(saved_model_name, input_length=saved_input_length, num_classes=len(class_names))
    model.load_state_dict(model_state, strict=True)
    model.to(device).eval()
    return model


def load_encoder(
    model_key: str,
    args: argparse.Namespace,
    device: torch.device,
):
    if model_key == "moment_official":
        return load_moment_official_model(
            model_id=args.moment_model_id,
            cache_dir=args.cache_dir,
            device=device,
            seq_len=args.input_length,
        )
    if model_key == "patchtst_pretrained":
        model, _ = load_patchtst_1ch_model(
            model_id=args.patchtst_model_id,
            cache_dir=args.cache_dir,
            device=device,
        )
        return model
    if model_key == "conv1dgap_same_input_3class":
        return load_conv1dgap_same_input_model(
            checkpoint_path=args.conv_checkpoint,
            device=device,
            model_name=args.conv_model_name,
            input_length=args.input_length,
        )
    raise ValueError(f"Unsupported model: {model_key}")


@torch.no_grad()
def encode_panel_embeddings(
    model_key: str,
    model,
    signals: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, signals.shape[0], batch_size):
        batch = torch.from_numpy(signals[start : start + batch_size]).float()
        if model_key == "conv1dgap_same_input_3class":
            features = encode_conv1dgap_features(model, batch, device=device)
        else:
            features = encode_pretrained_batch(model_key, model, batch, device=device)
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        chunks.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def reduce_embeddings(embeddings: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x = StandardScaler().fit_transform(embeddings)
    n_samples, n_features = x.shape
    pca_components = min(2, n_samples, n_features)
    pca = PCA(n_components=pca_components, random_state=seed)
    pca_raw = pca.fit_transform(x)
    pca_coords = np.zeros((n_samples, 2), dtype=np.float32)
    pca_coords[:, :pca_components] = pca_raw.astype(np.float32)
    if n_samples < 5:
        return pca_coords, pca_coords.copy(), {
            "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            "trustworthiness": float("nan"),
            "tsne_perplexity": float("nan"),
        }
    pre_dim = min(50, n_features, n_samples - 1)
    x_pre = PCA(n_components=pre_dim, random_state=seed).fit_transform(x) if pre_dim < n_features else x
    perplexity = min(30, max(2, (n_samples - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
    tsne_coords = tsne.fit_transform(x_pre).astype(np.float32)
    trust_neighbors = min(10, max(1, (n_samples // 2) - 1))
    trust = float(trustworthiness(x, tsne_coords, n_neighbors=trust_neighbors)) if trust_neighbors >= 1 else float("nan")
    return pca_coords, tsne_coords, {
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "trustworthiness": trust,
        "tsne_perplexity": float(perplexity),
    }


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 9,
            "axes.labelsize": 8,
        }
    )


def set_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=7, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#555555")
    ax.grid(False)


def add_colorbar(ax: plt.Axes, artist: plt.Collection) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cb = ax.figure.colorbar(artist, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2)


def panel_title(panel: ParticleSweepPanel) -> str:
    if panel.family == "single":
        return SINGLE_TITLES[panel.swept_parameter]
    return DOUBLE_TITLES[panel.swept_parameter]


def plot_family_figure(
    panels: list[ParticleSweepPanel],
    reductions: dict[str, dict[str, dict[str, np.ndarray]]],
    model_key: str,
    family: str,
    output_pdf: Path,
    output_png: Path,
) -> None:
    selected = [panel for panel in panels if panel.family == family]
    if not selected:
        return
    apply_plot_style()
    fig_width = max(10.8, 3.1 * len(selected))
    fig, axes = plt.subplots(2, len(selected), figsize=(fig_width, 6.3), constrained_layout=False, squeeze=False)
    for col, panel in enumerate(selected):
        for row, reduction_key in enumerate(["pca", "tsne"]):
            ax = axes[row, col]
            coords = reductions[model_key][panel.key][reduction_key]
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=panel.color_value,
                cmap="magma",
                s=9,
                alpha=0.88,
                linewidths=0,
            )
            title = f"{panel_title(panel)}\n{reduction_key.upper()}" if row == 0 else reduction_key.upper()
            ax.set_title(title, pad=4)
            set_axis_style(ax)
            add_colorbar(ax, scatter)

    equation = (
        r"$A\cos(2\pi f_D t + \phi)\exp(-(t-t_0)^2/(2\tau^2))$"
        if family == "single"
        else r"$A\cos(2\pi f_{DA}t+\phi_A)e_A + B\cos(2\pi f_{DB}t+\phi_B)e_B$"
    )
    fig.suptitle(f"{MODEL_DISPLAY[model_key]} - {family} particle sweeps", fontsize=12, y=0.965)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.16, wspace=0.32, hspace=0.38)
    fig.text(0.055, 0.045, equation, ha="left", va="bottom", fontsize=11, wrap=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def plot_model_comparison_figure(
    panels: list[ParticleSweepPanel],
    reductions: dict[str, dict[str, dict[str, np.ndarray]]],
    model_keys: list[str],
    family: str,
    output_pdf: Path,
    output_png: Path,
) -> None:
    selected = [panel for panel in panels if panel.family == family]
    if not selected or len(model_keys) < 2:
        return
    apply_plot_style()
    n_rows = 2 * len(model_keys)
    fig_width = max(10.8, 3.1 * len(selected))
    fig_height = max(6.3, 2.4 * n_rows)
    fig, axes = plt.subplots(n_rows, len(selected), figsize=(fig_width, fig_height), constrained_layout=False, squeeze=False)
    for model_index, model_key in enumerate(model_keys):
        for reduction_index, reduction_key in enumerate(["pca", "tsne"]):
            row = 2 * model_index + reduction_index
            for col, panel in enumerate(selected):
                ax = axes[row, col]
                coords = reductions[model_key][panel.key][reduction_key]
                scatter = ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=panel.color_value,
                    cmap="magma",
                    s=8,
                    alpha=0.88,
                    linewidths=0,
                )
                if row == 0:
                    ax.set_title(panel_title(panel), pad=4)
                if col == 0:
                    ax.set_ylabel(f"{MODEL_DISPLAY[model_key]}\n{reduction_key.upper()}", fontsize=8)
                set_axis_style(ax)
                add_colorbar(ax, scatter)

    fig.suptitle(f"{family.capitalize()} particle latent-space comparison", fontsize=12, y=0.985)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.06, wspace=0.32, hspace=0.38)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def reduction_groups(panels: list[ParticleSweepPanel], scope: str) -> dict[str, list[ParticleSweepPanel]]:
    if scope == "panel":
        return {panel.key: [panel] for panel in panels}
    if scope == "family":
        groups: dict[str, list[ParticleSweepPanel]] = {}
        for panel in panels:
            groups.setdefault(panel.family, []).append(panel)
        return groups
    if scope == "global":
        return {"global": panels}
    raise ValueError(f"Unsupported reduction scope: {scope}")


def reduce_model_embeddings(
    model_embeddings: dict[str, np.ndarray],
    panels: list[ParticleSweepPanel],
    scope: str,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]]]:
    panel_reductions: dict[str, dict[str, np.ndarray]] = {}
    group_metrics: dict[str, dict[str, float]] = {}
    for group_key, group_panels in reduction_groups(panels, scope).items():
        lengths = [model_embeddings[panel.key].shape[0] for panel in group_panels]
        joined = np.concatenate([model_embeddings[panel.key] for panel in group_panels], axis=0)
        pca_coords, tsne_coords, metrics = reduce_embeddings(joined, seed=seed)
        group_metrics[group_key] = metrics
        offset = 0
        for panel, length in zip(group_panels, lengths):
            panel_reductions[panel.key] = {
                "pca": pca_coords[offset : offset + length],
                "tsne": tsne_coords[offset : offset + length],
            }
            offset += length
    return panel_reductions, group_metrics


def choose_example_indices(n: int, count: int) -> np.ndarray:
    if n <= count:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, count, dtype=np.int64))


def plot_signal_examples(
    panels: list[ParticleSweepPanel],
    family: str,
    output_pdf: Path,
    output_png: Path,
    examples_per_panel: int = 7,
) -> None:
    selected = [panel for panel in panels if panel.family == family]
    if not selected:
        return
    apply_plot_style()
    fig_width = max(10.8, 3.1 * len(selected))
    fig, axes = plt.subplots(1, len(selected), figsize=(fig_width, 3.4), constrained_layout=False, squeeze=False)
    cmap = plt.get_cmap("magma")
    for col, panel in enumerate(selected):
        ax = axes[0, col]
        idx = choose_example_indices(panel.signal.shape[0], examples_per_panel)
        denom = max(float(panel.color_value.max() - panel.color_value.min()), 1.0e-12)
        for sample_idx in idx.tolist():
            alpha = float((panel.color_value[sample_idx] - panel.color_value.min()) / denom)
            ax.plot(panel.signal[sample_idx], color=cmap(alpha), linewidth=1.0, alpha=0.9)
        ax.set_title(panel_title(panel), pad=4)
        ax.tick_params(axis="both", labelsize=7, length=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#555555")
        ax.grid(False)
    fig.suptitle(f"{family.capitalize()} particle source signal examples", fontsize=12, y=0.96)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.82, bottom=0.18, wspace=0.32)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def build_single_ranges(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    ranges = dict(SINGLE_RANGES)
    ranges["fD"] = (float(args.single_fd_min), float(args.single_fd_max))
    ranges["tau"] = (float(args.single_tau_min), float(args.single_tau_max))
    return ranges


def build_double_ranges(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    ranges = dict(DOUBLE_RANGES)
    ranges["delta_t0"] = (float(args.double_delta_t0_min), float(args.double_delta_t0_max))
    return ranges


def validate_range(name: str, values: tuple[float, float]) -> None:
    lo, hi = values
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError(f"Invalid range for {name}: expected finite min < max, got {values}")


def write_run_summary(
    output_dir: Path,
    args: argparse.Namespace,
    model_keys: list[str],
    single_ranges: dict[str, tuple[float, float]],
    double_ranges: dict[str, tuple[float, float]],
    panels: list[ParticleSweepPanel],
) -> None:
    expected_files = [
        "synthetic_particle_signals.npz",
        "synthetic_particle_metadata.csv",
        "embeddings.npz",
        "reduction_metrics.json",
        "run_summary.json",
        "run_summary.md",
    ]
    for family in ["single", "double"]:
        if any(panel.family == family for panel in panels):
            expected_files.extend([
                f"{family}_particle_source_signal_examples.pdf",
                f"{family}_particle_source_signal_examples.png",
            ])
            if len(model_keys) >= 2:
                expected_files.extend([
                    f"{family}_particle_model_comparison_pca_tsne.pdf",
                    f"{family}_particle_model_comparison_pca_tsne.png",
                ])
            for model_key in model_keys:
                expected_files.extend([
                    f"{model_key}_{family}_particle_sweeps_pca_tsne.pdf",
                    f"{model_key}_{family}_particle_sweeps_pca_tsne.png",
                ])
    payload = {
        "models": model_keys,
        "families": args.families,
        "n_per_panel": int(args.n_per_panel),
        "input_length": int(args.input_length),
        "noise_std": float(args.noise_std),
        "seed": int(args.seed),
        "reduction_scope": args.reduction_scope,
        "normalization_non_moment": args.normalization,
        "single_ranges": single_ranges,
        "double_ranges": double_ranges,
        "generated_files": expected_files,
    }
    with (output_dir / "run_summary.json").open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    lines = [
        "# Synthetic Particle Sweep Run",
        "",
        f"- Models: {', '.join(model_keys)}",
        f"- Families: {args.families}",
        f"- Samples per panel: {args.n_per_panel}",
        f"- Input length: {args.input_length}",
        f"- Noise std: {args.noise_std}",
        f"- Reduction scope: {args.reduction_scope}",
        f"- Non-MOMENT normalization: {args.normalization}",
        "",
        "## Single Particle Ranges",
    ]
    lines.extend(f"- {key}: {value[0]} to {value[1]}" for key, value in single_ranges.items())
    lines.extend(["", "## Double Particle Ranges"])
    lines.extend(f"- {key}: {value[0]} to {value[1]}" for key, value in double_ranges.items())
    lines.extend(["", "## Generated Files"])
    lines.extend(f"- {name}" for name in expected_files)
    (output_dir / "run_summary.md").write_text("\n".join(lines) + "\n")


def write_metadata_csv(output_csv: Path, panels: list[ParticleSweepPanel]) -> None:
    fieldnames = [
        "family",
        "panel",
        "index",
        "swept_parameter",
        "color_value",
        "A",
        "fD",
        "phi",
        "t0",
        "tau",
        "B",
        "fDA",
        "fDB",
        "phiA",
        "phiB",
        "t0A",
        "t0B",
        "tauA",
        "tauB",
        "delta_t0",
        "delta_fD",
        "delta_phi",
        "ratio_BA",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for panel in panels:
            for i in range(panel.signal.shape[0]):
                row: dict[str, Any] = {
                    "family": panel.family,
                    "panel": panel.key,
                    "index": i,
                    "swept_parameter": panel.swept_parameter,
                    "color_value": float(panel.color_value[i]),
                }
                for name in fieldnames[5:]:
                    values = panel.parameters.get(name)
                    row[name] = "" if values is None else float(values[i])
                writer.writerow(row)


def save_npz_outputs(
    output_dir: Path,
    panels: list[ParticleSweepPanel],
    embeddings: dict[str, dict[str, np.ndarray]],
    reductions: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    signal_payload: dict[str, np.ndarray] = {}
    embedding_payload: dict[str, np.ndarray] = {}
    for panel in panels:
        signal_payload[f"{panel.key}_signals"] = panel.signal.astype(np.float32)
        signal_payload[f"{panel.key}_color"] = panel.color_value.astype(np.float32)
        for name, values in panel.parameters.items():
            signal_payload[f"{panel.key}_{name}"] = values.astype(np.float32)
        for model_key, model_embeddings in embeddings.items():
            embedding_payload[f"{model_key}_{panel.key}_embeddings"] = model_embeddings[panel.key].astype(np.float32)
            embedding_payload[f"{model_key}_{panel.key}_pca"] = reductions[model_key][panel.key]["pca"].astype(np.float32)
            embedding_payload[f"{model_key}_{panel.key}_tsne"] = reductions[model_key][panel.key]["tsne"].astype(np.float32)
    np.savez_compressed(output_dir / "synthetic_particle_signals.npz", **signal_payload)
    np.savez_compressed(output_dir / "embeddings.npz", **embedding_payload)


def parse_models(raw: str) -> list[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [model for model in models if model not in MODEL_DISPLAY]
    if unknown:
        raise ValueError(f"Unsupported models {unknown}. Expected one of {sorted(MODEL_DISPLAY)}")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed synthetic particle equation sweeps in pretrained latent spaces.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="moment_official,patchtst_pretrained,conv1dgap_same_input_3class")
    parser.add_argument("--moment-model-id", default=MOMENT_DEFAULT_ID)
    parser.add_argument("--patchtst-model-id", default=PATCHTST_DEFAULT_ID)
    parser.add_argument("--conv-checkpoint", type=Path, default=DEFAULT_CONV_CHECKPOINT)
    parser.add_argument("--conv-model-name", default="Conv1DGAP-L")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs" / "hf_cache")
    parser.add_argument("--n-per-panel", type=int, default=1800)
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--normalization", default="window_zscore")
    parser.add_argument("--families", choices=["single", "double", "both"], default="both")
    parser.add_argument("--reduction-scope", choices=["panel", "family", "global"], default="panel")
    parser.add_argument("--single-fd-min", type=float, default=SINGLE_RANGES["fD"][0])
    parser.add_argument("--single-fd-max", type=float, default=SINGLE_RANGES["fD"][1])
    parser.add_argument("--single-tau-min", type=float, default=SINGLE_RANGES["tau"][0])
    parser.add_argument("--single-tau-max", type=float, default=SINGLE_RANGES["tau"][1])
    parser.add_argument("--double-delta-t0-min", type=float, default=DOUBLE_RANGES["delta_t0"][0])
    parser.add_argument("--double-delta-t0-max", type=float, default=DOUBLE_RANGES["delta_t0"][1])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    model_keys = parse_models(args.models)
    single_ranges = build_single_ranges(args)
    double_ranges = build_double_ranges(args)
    for name, values in {**single_ranges, **double_ranges}.items():
        validate_range(name, values)
    include_single = args.families in ("single", "both")
    include_double = args.families in ("double", "both")
    panels = generate_particle_sweep_panels(
        n_per_panel=args.n_per_panel,
        length=args.input_length,
        seed=args.seed,
        noise_std=args.noise_std,
        include_single=include_single,
        include_double=include_double,
        single_ranges=single_ranges,
        double_ranges=double_ranges,
    )

    device = torch.device(args.device)
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    reductions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for model_key in model_keys:
        model = load_encoder(model_key, args, device)
        embeddings[model_key] = {}
        for panel in panels:
            signals = model_input_signals(model_key, panel.signal, args.normalization)
            emb = encode_panel_embeddings(
                model_key=model_key,
                model=model,
                signals=signals,
                batch_size=args.batch_size,
                device=device,
            )
            embeddings[model_key][panel.key] = emb
        reductions[model_key], metrics[model_key] = reduce_model_embeddings(
            model_embeddings=embeddings[model_key],
            panels=panels,
            scope=args.reduction_scope,
            seed=args.seed,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_npz_outputs(args.output_dir, panels, embeddings, reductions)
    write_metadata_csv(args.output_dir / "synthetic_particle_metadata.csv", panels)
    with (args.output_dir / "reduction_metrics.json").open("w") as f:
        json.dump(
            {
                "models": model_keys,
                "n_per_panel": int(args.n_per_panel),
                "input_length": int(args.input_length),
                "seed": int(args.seed),
                "noise_std": float(args.noise_std),
                "normalization_non_moment": args.normalization,
                "families": args.families,
                "reduction_scope": args.reduction_scope,
                "single_reference": SINGLE_REFERENCE,
                "single_ranges": single_ranges,
                "double_reference": DOUBLE_REFERENCE,
                "double_ranges": double_ranges,
                "moment_model_id": args.moment_model_id,
                "patchtst_model_id": args.patchtst_model_id,
                "conv_checkpoint": str(args.conv_checkpoint),
                "metrics": metrics,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    for family in ["single", "double"]:
        if (family == "single" and not include_single) or (family == "double" and not include_double):
            continue
        plot_signal_examples(
            panels=panels,
            family=family,
            output_pdf=args.output_dir / f"{family}_particle_source_signal_examples.pdf",
            output_png=args.output_dir / f"{family}_particle_source_signal_examples.png",
        )

    for model_key in model_keys:
        for family in ["single", "double"]:
            if (family == "single" and not include_single) or (family == "double" and not include_double):
                continue
            plot_family_figure(
                panels=panels,
                reductions=reductions,
                model_key=model_key,
                family=family,
                output_pdf=args.output_dir / f"{model_key}_{family}_particle_sweeps_pca_tsne.pdf",
                output_png=args.output_dir / f"{model_key}_{family}_particle_sweeps_pca_tsne.png",
            )
    for family in ["single", "double"]:
        if (family == "single" and not include_single) or (family == "double" and not include_double):
            continue
        plot_model_comparison_figure(
            panels=panels,
            reductions=reductions,
            model_keys=model_keys,
            family=family,
            output_pdf=args.output_dir / f"{family}_particle_model_comparison_pca_tsne.pdf",
            output_png=args.output_dir / f"{family}_particle_model_comparison_pca_tsne.png",
        )
    write_run_summary(
        output_dir=args.output_dir,
        args=args,
        model_keys=model_keys,
        single_ranges=single_ranges,
        double_ranges=double_ranges,
        panels=panels,
    )
    print(f"Wrote synthetic particle sweep outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
