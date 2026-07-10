#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

from p3_ssl.models import MomentLikeConfig, MomentLikeReconstructor


def main() -> None:
    parser = argparse.ArgumentParser(description="Export only the P3_SSL encoder weights and metadata.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    data_cfg = config["data"]
    patch_cfg = config["patching"]
    model_cfg = config["model"]
    model = MomentLikeReconstructor(
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
    model.load_state_dict(checkpoint["model_state_dict"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": model.encoder_state_dict(),
            "config": config,
            "source_checkpoint": str(args.checkpoint),
        },
        args.output,
    )
    print(f"Wrote encoder export to {args.output}")


if __name__ == "__main__":
    main()

