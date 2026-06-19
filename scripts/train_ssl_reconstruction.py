#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p3_ssl.config import load_config
from p3_ssl.data import SSLManifestDataset, read_manifest
from p3_ssl.losses import composite_reconstruction_loss
from p3_ssl.metrics import reconstruction_metrics
from p3_ssl.models import MomentLikeConfig, MomentLikeReconstructor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(config: dict, manifest: Path, split: str) -> SSLManifestDataset:
    data = config["data"]
    patching = config["patching"]
    masking = config["masking"]
    return SSLManifestDataset(
        manifest_csv=manifest,
        split=split,
        input_length_raw=int(data["input_length_raw"]),
        decimation_factor=int(data["decimation_factor"]),
        input_length_ssl=int(data["input_length_ssl"]),
        normalization=str(data.get("normalization", "window_zscore")),
        patch_size=int(patching["patch_size"]),
        patch_stride=int(patching["patch_stride"]),
        guard_points=int(patching.get("guard_points", 8)),
        mask_ratio=float(masking.get("mask_ratio", 0.25)),
        min_block_length=int(masking.get("min_block_length", 24)),
        max_block_length=int(masking.get("max_block_length", 128)),
        high_derivative_probability=float(masking.get("high_derivative_probability", 0.25)),
        seed=int(config["experiment"].get("seed", 42)),
    )


def make_model(config: dict) -> MomentLikeReconstructor:
    model_cfg = config["model"]
    data_cfg = config["data"]
    patch_cfg = config["patching"]
    cfg = MomentLikeConfig(
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
    return MomentLikeReconstructor(cfg)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        signal = batch["signal"].to(device)
        token_mask = batch["token_mask"].to(device)
        time_mask = batch["target_time_mask"].to(device)
        pred = model(signal, token_mask=token_mask)
        metrics = reconstruction_metrics(pred, signal, time_mask)
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the P3_SSL MOMENT-like masked reconstructor.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["experiment"].get("seed", 42)))
    output_dir = args.output_dir or Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_split = str(config["data"].get("split_train", "train"))
    val_split = str(config["data"].get("split_val", "val"))
    train_ds = make_dataset(config, args.manifest, train_split)
    available_val = bool(read_manifest(args.manifest, split=val_split))
    val_ds = make_dataset(config, args.manifest, val_split) if available_val else train_ds

    train_cfg = config["training"]
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 16))
    num_workers = args.num_workers if args.num_workers is not None else int(train_cfg.get("num_workers", 2))
    epochs = args.epochs or int(train_cfg.get("epochs", 20))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device(args.device)
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    loss_cfg = config["loss"]
    best_mse = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for batch in train_loader:
            signal = batch["signal"].to(device)
            token_mask = batch["token_mask"].to(device)
            time_mask = batch["target_time_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                pred = model(signal, token_mask=token_mask)
                loss, _ = composite_reconstruction_loss(
                    pred,
                    signal,
                    time_mask,
                    lambda_signal=float(loss_cfg.get("lambda_signal", 1.0)),
                    lambda_derivative=float(loss_cfg.get("lambda_derivative", 0.2)),
                    lambda_energy=float(loss_cfg.get("lambda_energy", 0.05)),
                    huber_delta=float(loss_cfg.get("huber_delta", 1.0)),
                )
            scaler.scale(loss).backward()
            if float(train_cfg.get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
            batches += 1
        val_metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": running / max(batches, 1), **val_metrics}
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        metric = val_metrics.get("masked_mse", record["train_loss"])
        state = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": record,
        }
        torch.save(state, ckpt_dir / "latest.pt")
        if metric < best_mse:
            best_mse = metric
            torch.save(state, ckpt_dir / "best.pt")

    with (output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

