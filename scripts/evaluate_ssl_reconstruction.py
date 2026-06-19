#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p3_ssl.config import load_config
from p3_ssl.data import SSLManifestDataset
from p3_ssl.metrics import reconstruction_metrics
from p3_ssl.models import MomentLikeConfig, MomentLikeReconstructor


def make_model(config: dict) -> MomentLikeReconstructor:
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
        seed=int(config["experiment"].get("seed", 42)),
    )


def plot_examples(pdf_path: Path, batches: list[dict], preds: list[torch.Tensor], max_examples: int) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        shown = 0
        for batch, pred in zip(batches, preds):
            signal = batch["signal"].cpu()
            mask = batch["target_time_mask"].cpu()
            event = batch["event_mask"].cpu()
            pred = pred.cpu()
            for i in range(signal.shape[0]):
                if shown >= max_examples:
                    return
                fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
                x = range(signal.shape[-1])
                ax.plot(x, signal[i, 0].numpy(), color="black", linewidth=0.8, label="target")
                ax.plot(x, pred[i, 0].numpy(), color="tab:red", linewidth=0.8, label="reconstruction")
                ax.fill_between(x, signal[i, 0].min().item(), signal[i, 0].max().item(), where=mask[i].numpy(), color="tab:orange", alpha=0.18)
                ax.fill_between(x, signal[i, 0].min().item(), signal[i, 0].max().item(), where=event[i].numpy(), color="tab:green", alpha=0.12)
                ax.set_title(str(batch["sample_id"][i]))
                ax.legend(loc="upper right")
                pdf.savefig(fig)
                plt.close(fig)
                shown += 1


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

    sums: dict[str, float] = {}
    count = 0
    example_batches = []
    example_preds = []
    with torch.no_grad():
        for batch in loader:
            signal = batch["signal"].to(device)
            token_mask = batch["token_mask"].to(device)
            time_mask = batch["target_time_mask"].to(device)
            pred = model(signal, token_mask=token_mask)
            metrics = reconstruction_metrics(pred, signal, time_mask)
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value
            count += 1
            if len(example_batches) * args.batch_size < args.max_plot_examples:
                example_batches.append(batch)
                example_preds.append(pred.detach().cpu())

    metrics = {key: value / max(count, 1) for key, value in sums.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    plot_examples(args.output_dir / "reconstruction_examples.pdf", example_batches, example_preds, args.max_plot_examples)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()

