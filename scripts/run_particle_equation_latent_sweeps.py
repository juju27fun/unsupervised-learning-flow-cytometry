#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
P0_ROOT = REPO_ROOT / "P0"
for path_entry in (ROOT, P0_ROOT, Path(__file__).resolve().parent):
    if str(path_entry) not in sys.path:
        sys.path.insert(0, str(path_entry))

import run_pretrained_backbone_embeddings as backbone
from p3_ssl.decimation import normalize_signal
from p3_ssl.pretrained_backbones import MOMENT_DEFAULT_ID, PATCHTST_DEFAULT_ID, ParticleEvent


CONV_MODEL_KEY = "conv1dgap_same_input_3class"
DEFAULT_CONV_CHECKPOINT = (
    ROOT
    / "outputs"
    / "pretrained_backbones"
    / "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap"
    / CONV_MODEL_KEY
    / "best_model.pt"
)
MODEL_KEYS = ("moment_official", "patchtst_pretrained", CONV_MODEL_KEY)
MODEL_DISPLAY = {
    "moment_official": "MOMENT frozen pretrained",
    "patchtst_pretrained": "PatchTST frozen pretrained",
    CONV_MODEL_KEY: "Conv1D-GAP-L supervised same-input",
}

SINGLE_BASE = {"A": 1.0, "fD": 16.0, "phi": 0.0, "t0": 0.5, "tau": 0.06}
TWO_BASE = {
    "A": 1.0,
    "B": 1.0,
    "fDA": 16.0,
    "fDB": 16.0,
    "phiA": 0.0,
    "phiB": 0.0,
    "t0A": 0.5,
    "t0B": 0.5,
    "tauA": 0.06,
    "tauB": 0.06,
}


@dataclass(frozen=True)
class ParticleEquationPanel:
    key: str
    title: str
    sweep_param: str
    signal: np.ndarray
    encoded_signal: np.ndarray
    color_value: np.ndarray
    params: dict[str, np.ndarray]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _linspace_param(low: float, high: float, n: int, rng: np.random.Generator, shuffle: bool) -> np.ndarray:
    values = np.linspace(low, high, n, dtype=np.float32)
    if shuffle:
        rng.shuffle(values)
    return values


def _noise(rng: np.random.Generator, shape: tuple[int, int], std: float) -> np.ndarray:
    if std <= 0.0:
        return np.zeros(shape, dtype=np.float32)
    return rng.normal(0.0, std, size=shape).astype(np.float32)


def particle_wave(
    t: np.ndarray,
    amplitude: np.ndarray,
    doppler: np.ndarray,
    phase: np.ndarray,
    center: np.ndarray,
    width: np.ndarray,
) -> np.ndarray:
    carrier = np.cos(2.0 * np.pi * doppler[:, None] * t[None, :] + phase[:, None])
    envelope = np.exp(-np.square(t[None, :] - center[:, None]) / (2.0 * np.square(width[:, None])))
    return (amplitude[:, None] * carrier * envelope).astype(np.float32)


def _normalize_batch(signals: np.ndarray, normalization: str) -> np.ndarray:
    return np.stack([normalize_signal(row, mode=normalization) for row in signals]).astype(np.float32)


def _single_panel(
    rng: np.random.Generator,
    key: str,
    title: str,
    sweep_param: str,
    values: np.ndarray,
    length: int,
    noise_std: float,
    normalization: str,
) -> ParticleEquationPanel:
    n = int(values.size)
    params = {name: np.full(n, value, dtype=np.float32) for name, value in SINGLE_BASE.items()}
    params[sweep_param] = values.astype(np.float32, copy=False)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    signal = particle_wave(t, params["A"], params["fD"], params["phi"], params["t0"], params["tau"])
    signal = signal + _noise(rng, signal.shape, noise_std)
    encoded_signal = _normalize_batch(signal, normalization)
    return ParticleEquationPanel(key, title, sweep_param, signal.astype(np.float32), encoded_signal, values, params)


def generate_single_particle_panels(
    n_per_panel: int,
    length: int,
    seed: int,
    noise_std: float = 0.02,
    normalization: str = "window_zscore",
    shuffle: bool = True,
) -> list[ParticleEquationPanel]:
    rng = np.random.default_rng(seed)
    specs = [
        ("amplitude_A", r"$A$", "A", 0.25, 2.0),
        ("doppler_fD", r"$f_D$", "fD", 2.0, 64.0),
        ("phase_phi", r"$\phi$", "phi", 0.0, 2.0 * np.pi),
        ("center_t0", r"$t_0$", "t0", 0.2, 0.8),
        ("width_tau", r"$\tau$", "tau", 0.02, 0.15),
    ]
    return [
        _single_panel(
            rng=rng,
            key=key,
            title=title,
            sweep_param=sweep_param,
            values=_linspace_param(low, high, n_per_panel, rng, shuffle),
            length=length,
            noise_std=noise_std,
            normalization=normalization,
        )
        for key, title, sweep_param, low, high in specs
    ]


def _two_panel(
    rng: np.random.Generator,
    key: str,
    title: str,
    sweep_param: str,
    values: np.ndarray,
    length: int,
    noise_std: float,
    normalization: str,
) -> ParticleEquationPanel:
    n = int(values.size)
    params = {name: np.full(n, value, dtype=np.float32) for name, value in TWO_BASE.items()}
    if sweep_param == "separation_dt":
        params["t0A"] = (0.5 - values / 2.0).astype(np.float32)
        params["t0B"] = (0.5 + values / 2.0).astype(np.float32)
    elif sweep_param == "amplitude_ratio":
        params["B"] = values.astype(np.float32)
    elif sweep_param == "frequency_delta":
        params["fDB"] = (params["fDA"] + values).astype(np.float32)
    elif sweep_param == "phase_delta":
        params["phiB"] = (params["phiA"] + values).astype(np.float32)
    elif sweep_param == "width_ratio":
        params["tauB"] = (params["tauA"] * values).astype(np.float32)
    else:
        raise ValueError(f"Unsupported two-particle sweep: {sweep_param}")

    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    signal_a = particle_wave(t, params["A"], params["fDA"], params["phiA"], params["t0A"], params["tauA"])
    signal_b = particle_wave(t, params["B"], params["fDB"], params["phiB"], params["t0B"], params["tauB"])
    signal = signal_a + signal_b + _noise(rng, signal_a.shape, noise_std)
    encoded_signal = _normalize_batch(signal, normalization)
    return ParticleEquationPanel(key, title, sweep_param, signal.astype(np.float32), encoded_signal, values, params)


def generate_two_particle_panels(
    n_per_panel: int,
    length: int,
    seed: int,
    noise_std: float = 0.02,
    normalization: str = "window_zscore",
    shuffle: bool = True,
) -> list[ParticleEquationPanel]:
    rng = np.random.default_rng(seed)
    specs = [
        ("separation_dt", r"$\Delta t_0$", "separation_dt", 0.0, 3.0 * TWO_BASE["tauA"]),
        ("amplitude_ratio", r"$B/A$", "amplitude_ratio", 0.25, 2.0),
        ("frequency_delta", r"$f_{DB} - f_{DA}$", "frequency_delta", -16.0, 16.0),
        ("phase_delta", r"$\phi_B - \phi_A$", "phase_delta", 0.0, 2.0 * np.pi),
        ("width_ratio", r"$\tau_B/\tau_A$", "width_ratio", 0.5, 2.0),
    ]
    return [
        _two_panel(
            rng=rng,
            key=key,
            title=title,
            sweep_param=sweep_param,
            values=_linspace_param(low, high, n_per_panel, rng, shuffle),
            length=length,
            noise_std=noise_std,
            normalization=normalization,
        )
        for key, title, sweep_param, low, high in specs
    ]


def generate_panels(
    scenario: str,
    n_per_panel: int,
    length: int,
    seed: int,
    noise_std: float,
    normalization: str,
) -> list[ParticleEquationPanel]:
    if scenario == "single_particle":
        return generate_single_particle_panels(n_per_panel, length, seed, noise_std, normalization)
    if scenario == "two_particles":
        return generate_two_particle_panels(n_per_panel, length, seed, noise_std, normalization)
    raise ValueError(f"Unsupported scenario: {scenario}")


def make_synthetic_events(panels: list[ParticleEquationPanel], scenario: str) -> list[ParticleEvent]:
    events: list[ParticleEvent] = []
    for panel in panels:
        for index in range(panel.signal.shape[0]):
            events.append(
                ParticleEvent(
                    event_id=f"{scenario}/{panel.key}/{index:06d}",
                    sample_id=f"{panel.key}_{index:06d}",
                    split="synthetic",
                    signal_path="",
                    label_path="",
                    class_id=0,
                    class_name=panel.key,
                    center_norm=float(panel.params.get("t0", panel.params.get("t0A"))[index]),
                    width_norm=float(panel.params.get("tau", panel.params.get("tauA"))[index]),
                    center_index=-1,
                    crop_start=0,
                    crop_end=int(panel.signal.shape[1]),
                )
            )
    return events


def save_signal_outputs(output_dir: Path, panels: list[ParticleEquationPanel], scenario: str, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_payload: dict[str, np.ndarray] = {}
    encoded_payload: dict[str, np.ndarray] = {}
    for panel in panels:
        raw_payload[f"{panel.key}_signals"] = panel.signal.astype(np.float32)
        encoded_payload[f"{panel.key}_signals"] = panel.encoded_signal.astype(np.float32)
        raw_payload[f"{panel.key}_color"] = panel.color_value.astype(np.float32)
        encoded_payload[f"{panel.key}_color"] = panel.color_value.astype(np.float32)
        for name, values in panel.params.items():
            raw_payload[f"{panel.key}_{name}"] = values.astype(np.float32)
            encoded_payload[f"{panel.key}_{name}"] = values.astype(np.float32)
    np.savez_compressed(output_dir / "synthetic_signals_raw.npz", **raw_payload)
    np.savez_compressed(output_dir / "synthetic_signals_encoded.npz", **encoded_payload)

    fieldnames = ["scenario", "panel", "index", "sweep_param", "color_value"]
    for panel in panels:
        for key in panel.params:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "synthetic_metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for panel in panels:
            for i in range(panel.signal.shape[0]):
                row: dict[str, Any] = {
                    "scenario": scenario,
                    "panel": panel.key,
                    "index": i,
                    "sweep_param": panel.sweep_param,
                    "color_value": float(panel.color_value[i]),
                }
                row.update({name: float(values[i]) for name, values in panel.params.items()})
                writer.writerow(row)

    with (output_dir / "run_config.json").open("w") as f:
        json.dump(
            {
                "scenario": scenario,
                "n_per_panel": int(args.n_per_panel),
                "input_length": int(args.input_length),
                "seed": int(args.seed),
                "noise_std": float(args.noise_std),
                "normalization": args.normalization,
                "time_domain": "normalized [0, 1]",
                "frequency_units": "cycles per 512-sample window by default",
                "models": args.models,
                "skip_tsne": bool(args.skip_tsne),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def load_conv1dgap_same_input(checkpoint_path: Path, device: torch.device | str) -> nn.Module:
    from models import create_model

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing Conv1D-GAP same-input checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    model_name = str(state.get("model_name", "Conv1DGAP-L")) if isinstance(state, dict) else "Conv1DGAP-L"
    input_length = int(state.get("input_length", 512)) if isinstance(state, dict) else 512
    class_names = state.get("class_names", ("2um", "4um", "10um")) if isinstance(state, dict) else ("2um", "4um", "10um")
    num_classes = len(class_names)
    model = create_model(model_name, input_length=input_length, num_classes=num_classes)
    model_state = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    model.load_state_dict(model_state, strict=True)
    model.to(device).eval()
    return model


def encode_conv_features_all(model: nn.Module, signals: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, signals.shape[0], batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size]).float()
            features = backbone.encode_conv1dgap_features(model, batch, device=device)
            chunks.append(features.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def load_encoder_for_model(model_key: str, args: argparse.Namespace, device: torch.device, model_dir: Path):
    if model_key == "moment_official":
        args.event_length = int(args.input_length)
        return backbone.load_encoder(model_key, args.moment_model_id, args.cache_dir, device, model_dir, args)
    if model_key == "patchtst_pretrained":
        args.event_length = int(args.input_length)
        return backbone.load_encoder(model_key, args.patchtst_model_id, args.cache_dir, device, model_dir, args)
    if model_key == CONV_MODEL_KEY:
        model = load_conv1dgap_same_input(args.conv1dgap_checkpoint, device)
        return model, {
            "source_model_id": str(args.conv1dgap_checkpoint),
            "input_representation": "same normalized 512-sample synthetic tensor as MOMENT/PatchTST",
            "supervised_same_input_checkpoint": True,
            "public_pretrained": False,
        }
    raise ValueError(f"Unsupported model: {model_key}")


def encode_panel_embeddings(
    model_key: str,
    encoder,
    panels: list[ParticleEquationPanel],
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    for panel in panels:
        if model_key == CONV_MODEL_KEY:
            emb = encode_conv_features_all(encoder, panel.encoded_signal, batch_size, device)
        else:
            emb = backbone.encode_all_events(model_key, encoder, panel.encoded_signal, batch_size, device)
        embeddings[panel.key] = emb.astype(np.float32)
    return embeddings


def reduce_panel_embeddings(embeddings: dict[str, np.ndarray], seed: int):
    reductions: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for key, emb in embeddings.items():
        pca, tsne, panel_metrics = backbone.reduce_embeddings(emb, seed=seed)
        reductions[key] = {"pca": pca, "tsne": tsne}
        metrics[key] = panel_metrics
    return reductions, metrics


def reduce_panel_embeddings_fast(embeddings: dict[str, np.ndarray], seed: int, skip_tsne: bool):
    if not skip_tsne:
        return reduce_panel_embeddings(embeddings, seed)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    reductions: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for key, emb in embeddings.items():
        x = StandardScaler().fit_transform(emb)
        pca_model = PCA(n_components=2, random_state=seed)
        pca = pca_model.fit_transform(x).astype(np.float32)
        panel_metrics = {
            "pca_explained_variance_ratio_sum": float(np.sum(pca_model.explained_variance_ratio_)),
            "trustworthiness": float("nan"),
            "tsne_perplexity": float("nan"),
            "tsne_skipped": 1.0,
        }
        reductions[key] = {"pca": pca, "tsne": pca.copy()}
        metrics[key] = panel_metrics
    return reductions, metrics


def save_model_outputs(
    output_dir: Path,
    panels: list[ParticleEquationPanel],
    embeddings: dict[str, np.ndarray],
    reductions: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, dict[str, float]],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for panel in panels:
        payload[f"{panel.key}_embeddings"] = embeddings[panel.key].astype(np.float32)
        payload[f"{panel.key}_pca"] = reductions[panel.key]["pca"].astype(np.float32)
        payload[f"{panel.key}_tsne"] = reductions[panel.key]["tsne"].astype(np.float32)
        payload[f"{panel.key}_color"] = panel.color_value.astype(np.float32)
    np.savez_compressed(output_dir / "embeddings.npz", **payload)
    with (output_dir / "reduction_metrics.json").open("w") as f:
        json.dump({"reduction": metrics, "metadata": metadata}, f, indent=2, sort_keys=True)


def _add_colorbar(ax: plt.Axes, artist: plt.Collection) -> None:
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    cb = ax.figure.colorbar(artist, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2)


def _set_axis_style(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=6, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#555555")
    ax.grid(False)


def plot_model_grid(
    panels: list[ParticleEquationPanel],
    reductions_by_model: dict[str, dict[str, dict[str, np.ndarray]]],
    output_pdf: Path,
    output_png: Path,
    scenario: str,
) -> None:
    backbone.apply_fig7_plot_style()
    model_keys = list(reductions_by_model)
    n_rows = 2 * len(model_keys)
    n_cols = len(panels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(12.0, 3.05 * n_cols), max(6.3, 1.75 * n_rows)),
        constrained_layout=False,
        squeeze=False,
    )
    for model_idx, model_key in enumerate(model_keys):
        for reduction_idx, reduction_key in enumerate(("pca", "tsne")):
            row = 2 * model_idx + reduction_idx
            for col, panel in enumerate(panels):
                ax = axes[row, col]
                coords = reductions_by_model[model_key][panel.key][reduction_key]
                scatter = ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=panel.color_value,
                    cmap="magma",
                    s=9,
                    alpha=0.88,
                    linewidths=0,
                )
                if row == 0:
                    ax.set_title(f"{panel.title}\n{panel.sweep_param}", pad=4)
                if col == 0:
                    ax.set_ylabel(f"{MODEL_DISPLAY.get(model_key, model_key)}\n{reduction_key.upper()}", fontsize=7)
                _set_axis_style(ax)
                _add_colorbar(ax, scatter)
    caption = (
        f"{scenario} latent sweeps. Signals are generated analytically, normalized as configured, "
        "encoded by each backbone, then reduced independently per model and panel."
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.10, wspace=0.36, hspace=0.46)
    fig.text(0.065, 0.035, caption, ha="left", va="bottom", fontsize=10, wrap=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def parse_models(raw: str) -> list[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    unsupported = [item for item in models if item not in MODEL_KEYS]
    if unsupported:
        raise ValueError(f"Unsupported models {unsupported}. Expected any of {MODEL_KEYS}")
    return models


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    panels = generate_panels(
        scenario=args.scenario,
        n_per_panel=args.n_per_panel,
        length=args.input_length,
        seed=args.seed,
        noise_std=args.noise_std,
        normalization=args.normalization,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_signal_outputs(args.output_dir, panels, args.scenario, args)
    events = make_synthetic_events(panels, args.scenario)
    with (args.output_dir / "synthetic_events_metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0].__dataclass_fields__.keys()))
        writer.writeheader()
        for event in events:
            writer.writerow({key: getattr(event, key) for key in event.__dataclass_fields__})

    device = torch.device(args.device)
    reductions_by_model: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for model_key in parse_models(args.models):
        model_dir = args.output_dir / model_key
        encoder, metadata = load_encoder_for_model(model_key, args, device, model_dir)
        embeddings = encode_panel_embeddings(model_key, encoder, panels, args.batch_size, device)
        reductions, metrics = reduce_panel_embeddings_fast(embeddings, args.seed, args.skip_tsne)
        save_model_outputs(
            output_dir=model_dir,
            panels=panels,
            embeddings=embeddings,
            reductions=reductions,
            metrics=metrics,
            metadata={
                **metadata,
                "model_key": model_key,
                "display_name": MODEL_DISPLAY.get(model_key, model_key),
                "input_length": int(args.input_length),
                "normalization": args.normalization,
            },
        )
        reductions_by_model[model_key] = reductions

    plot_model_grid(
        panels=panels,
        reductions_by_model=reductions_by_model,
        output_pdf=args.output_dir / f"{args.scenario}_latent_sweeps_pca_tsne.pdf",
        output_png=args.output_dir / f"{args.scenario}_latent_sweeps_pca_tsne.png",
        scenario=args.scenario,
    )
    print(f"Wrote particle-equation latent sweeps to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode analytic particle equations into MOMENT/PatchTST/Conv1D-GAP latent spaces.")
    parser.add_argument("--scenario", choices=["single_particle", "two_particles"], default="single_particle")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "particle_equation_latent_sweeps")
    parser.add_argument("--models", default="moment_official,patchtst_pretrained,conv1dgap_same_input_3class")
    parser.add_argument("--moment-model-id", default=MOMENT_DEFAULT_ID)
    parser.add_argument("--patchtst-model-id", default=PATCHTST_DEFAULT_ID)
    parser.add_argument("--conv1dgap-checkpoint", type=Path, default=DEFAULT_CONV_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs" / "hf_cache")
    parser.add_argument("--n-per-panel", type=int, default=1800)
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--normalization", choices=["none", "window_zscore", "robust_zscore"], default="window_zscore")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--skip-tsne", action="store_true", help="Use PCA coordinates in the t-SNE row for quick smoke tests.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
