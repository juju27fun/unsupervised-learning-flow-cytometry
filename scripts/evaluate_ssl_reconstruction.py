#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p3_ssl.config import load_config, validate_ssl_config
from p3_ssl.data import SSLManifestDataset
from p3_ssl.metrics import (
    finalize_mask_coherence_sums,
    finalize_reconstruction_metric_sums,
    mask_coherence_batch_sums,
    reconstruction_metric_sums,
    reconstruction_strata_masks,
)
from p3_ssl.models import MomentLikeConfig, MomentLikeReconstructor


STRATA_ORDER = ("impossible_event", "mostly_event", "partial_event", "background_only")


def make_model(config: dict) -> MomentLikeReconstructor:
    validate_ssl_config(config)
    data_cfg = config["data"]
    patch_cfg = config["patching"]
    model_cfg = config["model"]
    return MomentLikeReconstructor(
        MomentLikeConfig(
            input_length=int(data_cfg["input_length_ssl"]),
            patch_size=int(patch_cfg["patch_size"]),
            patch_stride=int(patch_cfg["patch_stride"]),
            d_model=int(model_cfg["d_model"]),
            n_heads=int(model_cfg["n_heads"]),
            n_layers=int(model_cfg["n_layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            activation=str(model_cfg.get("activation", "gelu")),
            max_tokens=int(model_cfg.get("max_tokens", 1024)),
        )
    )


def make_dataset(config: dict, manifest: Path, split: str) -> SSLManifestDataset:
    validate_ssl_config(config)
    return SSLManifestDataset(
        manifest_csv=manifest,
        split=split,
        input_length_raw=int(config["data"]["input_length_raw"]),
        decimation_factor=int(config["data"]["decimation_factor"]),
        input_length_ssl=int(config["data"]["input_length_ssl"]),
        normalization=str(config["data"].get("normalization", "window_zscore")),
        patch_size=int(config["patching"]["patch_size"]),
        patch_stride=int(config["patching"]["patch_stride"]),
        guard_points=int(config["patching"].get("guard_points", 8)),
        mask_ratio=float(config["masking"].get("mask_ratio", 0.25)),
        min_block_length=int(config["masking"].get("min_block_length", 24)),
        max_block_length=int(config["masking"].get("max_block_length", 128)),
        high_derivative_probability=float(config["masking"].get("high_derivative_probability", 0.25)),
        event_biased_probability=float(config["masking"].get("event_biased_probability", 0.0)),
        avoid_fully_hidden_events=bool(config["masking"].get("avoid_fully_hidden_events", False)),
        max_event_hidden_fraction=(
            None
            if config["masking"].get("max_event_hidden_fraction") is None
            else float(config["masking"]["max_event_hidden_fraction"])
        ),
        max_mask_attempts=int(config["masking"].get("max_mask_attempts", 1)),
        seed=int(config["experiment"].get("seed", 42)),
    )


def add_sums(dst: dict[str, float], src: dict[str, float]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0.0) + float(value)


def add_reconstruction_sums(dst: dict[str, dict[str, float]], name: str, src: dict[str, float]) -> None:
    bucket = dst.setdefault(name, {})
    add_sums(bucket, src)


def selected_time_mask(time_mask: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    return time_mask & selected.to(device=time_mask.device, dtype=torch.bool).unsqueeze(-1)


def sample_category(target_mask: torch.Tensor, hidden_mask: torch.Tensor, event_mask: torch.Tensor) -> tuple[str, float, float]:
    event_points = int(event_mask.sum().item())
    if event_points == 0:
        return "background_only", 0.0, 0.0
    event_target = int((target_mask & event_mask).sum().item())
    event_hidden = int((hidden_mask & event_mask).sum().item())
    target_frac = float(event_target) / float(event_points)
    hidden_frac = float(event_hidden) / float(event_points)
    if hidden_frac >= 1.0:
        return "impossible_event", target_frac, hidden_frac
    if target_frac >= 0.5:
        return "mostly_event", target_frac, hidden_frac
    if target_frac > 0.0:
        return "partial_event", target_frac, hidden_frac
    return "background_only", target_frac, hidden_frac


def maybe_collect_examples(
    examples: dict[str, list[dict]],
    batch: dict,
    pred: torch.Tensor,
    max_examples: int,
) -> None:
    if max_examples <= 0:
        return
    quota = max(1, math.ceil(max_examples / len(STRATA_ORDER)))
    for i in range(pred.shape[0]):
        category, target_frac, hidden_frac = sample_category(
            batch["target_time_mask"][i].cpu(),
            batch["token_time_mask"][i].cpu(),
            batch["event_mask"][i].cpu(),
        )
        if len(examples[category]) >= quota:
            continue
        examples[category].append(
            {
                "category": category,
                "event_target_fraction": target_frac,
                "event_hidden_fraction": hidden_frac,
                "sample_id": str(batch["sample_id"][i]),
                "signal": batch["signal"][i].detach().cpu(),
                "pred": pred[i].detach().cpu(),
                "target_time_mask": batch["target_time_mask"][i].detach().cpu(),
                "hidden_time_mask": batch["hidden_time_mask"][i].detach().cpu(),
                "token_time_mask": batch["token_time_mask"][i].detach().cpu(),
                "event_mask": batch["event_mask"][i].detach().cpu(),
            }
        )


def plot_examples(pdf_path: Path, examples_by_category: dict[str, list[dict]], max_examples: int) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    examples: list[dict] = []
    for category in STRATA_ORDER:
        examples.extend(examples_by_category[category])
    examples = examples[:max_examples]
    if not examples:
        return
    with PdfPages(pdf_path) as pdf:
        for item in examples:
            signal = item["signal"][0].numpy()
            pred = item["pred"][0].numpy()
            target_mask = item["target_time_mask"].numpy()
            hidden_mask = item["hidden_time_mask"].numpy()
            token_mask = item["token_time_mask"].numpy()
            event_mask = item["event_mask"].numpy()
            x = np.arange(signal.shape[-1])
            ymin = float(min(signal.min(), pred.min()))
            ymax = float(max(signal.max(), pred.max()))
            if ymin == ymax:
                ymin -= 1.0
                ymax += 1.0

            fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
            ax.plot(x, signal, color="black", linewidth=0.8, label="target")
            ax.plot(x, pred, color="tab:red", linewidth=0.8, label="reconstruction")
            ax.fill_between(x, ymin, ymax, where=token_mask, color="tab:purple", alpha=0.10, label="token-hidden")
            ax.fill_between(x, ymin, ymax, where=hidden_mask, color="tab:orange", alpha=0.14, label="hidden+guard")
            ax.fill_between(x, ymin, ymax, where=target_mask, color="tab:red", alpha=0.16, label="loss target")
            ax.fill_between(x, ymin, ymax, where=event_mask, color="tab:green", alpha=0.14, label="event")
            ax.set_ylim(ymin, ymax)
            ax.set_title(
                f"{item['sample_id']} | {item['category']} | "
                f"event target={item['event_target_fraction']:.2f}, hidden={item['event_hidden_fraction']:.2f}"
            )
            ax.legend(loc="upper right", ncol=3, fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a P3_SSL reconstruction checkpoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-plot-examples", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = make_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = make_dataset(config, args.manifest, args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    recon_sums: dict[str, dict[str, float]] = {}
    coherence_sums: dict[str, float] = {}
    strata_counts = {name: 0.0 for name in STRATA_ORDER}
    examples = {name: [] for name in STRATA_ORDER}
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            token_mask = batch["token_mask"].to(device)
            target_mask = batch["target_time_mask"].to(device)
            hidden_mask = batch["token_time_mask"].to(device)
            event_mask = batch["event_mask"].to(device)
            pred = model(signal, token_mask=token_mask)

            add_reconstruction_sums(recon_sums, "all", reconstruction_metric_sums(pred, signal, target_mask))
            add_reconstruction_sums(recon_sums, "event_region", reconstruction_metric_sums(pred, signal, target_mask & event_mask))
            add_reconstruction_sums(recon_sums, "background_region", reconstruction_metric_sums(pred, signal, target_mask & ~event_mask))
            strata = reconstruction_strata_masks(target_mask, event_mask, hidden_mask=hidden_mask)
            for name, selected in strata.items():
                if name in strata_counts:
                    strata_counts[name] += float(selected.sum().detach().cpu())
                add_reconstruction_sums(recon_sums, name, reconstruction_metric_sums(pred, signal, selected_time_mask(target_mask, selected)))
            add_sums(coherence_sums, mask_coherence_batch_sums(target_mask, hidden_mask, event_mask))
            maybe_collect_examples(examples, batch, pred, args.max_plot_examples)

    metrics: dict[str, float | None] = {}
    metrics.update(finalize_reconstruction_metric_sums(recon_sums.get("all", {})))
    for name in ("event_region", "background_region", *STRATA_ORDER):
        metrics.update(finalize_reconstruction_metric_sums(recon_sums.get(name, {}), prefix=name))
    metrics.update(finalize_mask_coherence_sums(coherence_sums))
    for name, count in strata_counts.items():
        metrics[f"{name}_samples"] = count

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    plot_examples(args.output_dir / "reconstruction_examples.pdf", examples, args.max_plot_examples)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
