#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p3_ssl.config import load_config, validate_ssl_config
from p3_ssl.augmentations import cosine_distance_loss, positive_signal_augmentation
from p3_ssl.data import SSLManifestDataset, read_manifest
from p3_ssl.decimation import crop_or_pad, decimate_signal, ensure_1d_signal, normalize_signal
from p3_ssl.embedding import CLASS_NAMES, collect_events, event_records_to_metadata, pool_token_embeddings, token_indices_for_interval
from p3_ssl.hybrid_physics import build_hybrid_manifest, generate_synthetic_manifest, summarize_hybrid_manifest
from p3_ssl.hybrid_training import build_training_stages, fixed_ratio_hybrid_batch_sampler, profile_value, synthetic_only_physics_params
from p3_ssl.losses import composite_reconstruction_loss
from p3_ssl.metrics import (
    finalize_mask_coherence_sums,
    finalize_reconstruction_metric_sums,
    mask_coherence_batch_sums,
    reconstruction_metric_sums,
    reconstruction_metrics,
)
from p3_ssl.models import MomentLikeConfig, MomentLikeReconstructor
from p3_ssl.physics import evaluate_physical_latent_space, physical_contrastive_loss
from p3_ssl.physical_eval import (
    evaluate_encoder_on_sweep_directory,
    evaluate_sweep_directory,
    merge_reference_and_candidate_rankings,
    write_physical_evaluation_report,
)
from p3_ssl.robustness import embedding_robustness_metrics
from p3_ssl.run_assessment import assess_hybrid_run, write_run_assessment
from p3_ssl.run_comparison import compare_reconstruction_metrics, load_reference_reconstruction_metrics
from p3_ssl.serialization import json_safe
from p3_ssl.masking import PatchSpec
from scripts.run_ssl_assessment_figures import (
    EmbeddingBundle,
    plot_manifold_figure,
    plot_retrieval_sheet,
    run_label_efficiency,
    summarize_probe_rows,
    write_assessment_dashboard,
    write_robustness_placeholder,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(config: dict[str, Any], manifest: Path, split: str) -> SSLManifestDataset:
    validate_ssl_config(config)
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
        event_biased_probability=float(masking.get("event_biased_probability", 0.0)),
        avoid_fully_hidden_events=bool(masking.get("avoid_fully_hidden_events", False)),
        max_event_hidden_fraction=(
            None
            if masking.get("max_event_hidden_fraction") is None
            else float(masking["max_event_hidden_fraction"])
        ),
        max_mask_attempts=int(masking.get("max_mask_attempts", 1)),
        seed=int(config["experiment"].get("seed", 42)),
    )


def make_model(config: dict[str, Any]) -> MomentLikeReconstructor:
    validate_ssl_config(config)
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


def load_preprocessed_signal(row, config: dict[str, Any]) -> np.ndarray:
    data_cfg = config["data"]
    signal = ensure_1d_signal(np.load(row.signal_path))
    signal = crop_or_pad(signal, int(data_cfg["input_length_raw"]), mode="center")
    signal = decimate_signal(signal, int(data_cfg["decimation_factor"]), method="mean")
    signal = crop_or_pad(signal, int(data_cfg["input_length_ssl"]), mode="center")
    return normalize_signal(signal, mode=str(data_cfg.get("normalization", "window_zscore")))


def load_reconstruction_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[MomentLikeReconstructor, int, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_config = checkpoint["config"]
    model = make_model(ckpt_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    input_length = int(ckpt_config["data"]["input_length_ssl"])
    pool = str(ckpt_config.get("model", {}).get("embedding_pool", "mean"))
    return model, input_length, pool


def _row_key(row) -> tuple[str, str, str]:
    return (row.split, row.sample_id, str(row.signal_path))


def _event_key(event) -> tuple[str, str, str]:
    return (event.split, event.sample_id, str(event.signal_path))


def select_rows_deterministic(rows: list[Any], max_rows: int | None, seed: int) -> list[Any]:
    if max_rows is None or len(rows) <= max_rows:
        return rows
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(np.arange(len(rows)), size=max_rows, replace=False))
    return [rows[int(index)] for index in indices]


def make_batch_sampler(dataset: SSLManifestDataset, synthetic_fraction: float, batch_size: int, seed: int):
    return fixed_ratio_hybrid_batch_sampler(
        [row.source_kind for row in dataset.rows],
        synthetic_fraction=synthetic_fraction,
        batch_size=batch_size,
        seed=seed,
    )


def reconstruction_selection_pass(
    recon_metrics: dict[str, float],
    reference: dict[str, Any],
    max_regression_fraction: float,
    keys: tuple[str, ...] = ("masked_mse", "derivative_mse"),
) -> bool:
    """Keep physical checkpoint selection from choosing a reconstruction-regressed epoch."""
    if reference.get("status") != "ok":
        return True
    ref_metrics = reference.get("metrics", {})
    checked = 0
    for key in keys:
        current = recon_metrics.get(key)
        baseline = ref_metrics.get(key)
        if current is None or baseline is None:
            continue
        checked += 1
        if float(baseline) != 0.0 and (float(current) - float(baseline)) / float(baseline) > max_regression_fraction:
            return False
    return checked > 0


def add_sums(target: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        target[key] = target.get(key, 0.0) + float(value)


def real_row_mask(source_kind: list[str] | tuple[str, ...] | Any) -> torch.Tensor:
    return torch.as_tensor([str(item) != "synthetic" for item in source_kind], dtype=torch.bool)


@torch.no_grad()
def evaluate_reconstruction(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
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
            sums[key] = sums.get(key, 0.0) + float(value)
        count += 1
    return {key: value / max(count, 1) for key, value in sums.items()}


@torch.no_grad()
def evaluate_reconstruction_detailed(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    real_only: bool = False,
) -> dict[str, float | None]:
    model.eval()
    recon_sums: dict[str, dict[str, float]] = {}
    coherence_sums: dict[str, float] = {}
    n_samples = 0
    n_real_seen = 0
    for batch in loader:
        signal = batch["signal"]
        target_mask = batch["target_time_mask"]
        hidden_mask = batch["token_time_mask"]
        token_mask = batch["token_mask"]
        event_mask = batch["event_mask"]
        if real_only:
            mask = real_row_mask(batch.get("source_kind", []))
            n_real_seen += int(mask.sum().item())
            if not bool(mask.any()):
                continue
            signal = signal[mask]
            target_mask = target_mask[mask]
            hidden_mask = hidden_mask[mask]
            token_mask = token_mask[mask]
            event_mask = event_mask[mask]
        n_samples += int(signal.shape[0])
        signal = signal.to(device)
        target_mask = target_mask.to(device)
        hidden_mask = hidden_mask.to(device)
        token_mask = token_mask.to(device)
        event_mask = event_mask.to(device)
        pred = model(signal, token_mask=token_mask)
        for name, mask_value in {
            "all": target_mask,
            "event_region": target_mask & event_mask,
            "background_region": target_mask & ~event_mask,
        }.items():
            recon_sums.setdefault(name, {})
            add_sums(recon_sums[name], reconstruction_metric_sums(pred, signal, mask_value))
        add_sums(coherence_sums, mask_coherence_batch_sums(target_mask, hidden_mask, event_mask))

    metrics: dict[str, float | None] = {
        "samples": float(n_samples),
        "real_only": float(real_only),
        "real_seen": float(n_real_seen),
    }
    metrics.update(finalize_reconstruction_metric_sums(recon_sums.get("all", {})))
    for name in ("event_region", "background_region"):
        metrics.update(finalize_reconstruction_metric_sums(recon_sums.get(name, {}), prefix=name))
    metrics.update(finalize_mask_coherence_sums(coherence_sums))
    return metrics


@torch.no_grad()
def collect_physical_embeddings(
    model: MomentLikeReconstructor,
    loader: DataLoader,
    device: torch.device,
    pool: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    embeddings: list[np.ndarray] = []
    params: list[np.ndarray] = []
    for batch in loader:
        signal = batch["signal"].to(device)
        emb = model.global_embedding(signal, token_mask=None, pool=pool)
        embeddings.append(emb.detach().cpu().numpy())
        params.append(batch["physics_params"].numpy())
    return np.concatenate(embeddings, axis=0), np.concatenate(params, axis=0)


@torch.no_grad()
def evaluate_real_estimated_physics(
    model: MomentLikeReconstructor,
    loaders: list[DataLoader | None],
    device: torch.device,
    pool: str,
    k_neighbors: int,
    pass_threshold: float,
) -> dict[str, Any]:
    """Evaluate latent correlations against partial real physics estimates when available."""
    model.eval()
    embeddings: list[np.ndarray] = []
    params: list[np.ndarray] = []
    total_real_seen = 0
    total_with_estimates = 0
    for loader in loaders:
        if loader is None:
            continue
        for batch in loader:
            source_kind = batch.get("source_kind", [])
            real_mask = real_row_mask(source_kind)
            total_real_seen += int(real_mask.sum().item())
            has_physics = batch["has_physics_params"].bool()
            mask = real_mask & has_physics
            total_with_estimates += int(mask.sum().item())
            if not bool(mask.any()):
                continue
            signal = batch["signal"][mask].to(device)
            emb = model.global_embedding(signal, token_mask=None, pool=pool)
            embeddings.append(emb.detach().cpu().numpy())
            params.append(batch["physics_params"][mask].numpy())
    if not embeddings:
        return {
            "status": "not_run",
            "reason": "no_real_rows_with_estimated_physics",
            "real_rows_seen": total_real_seen,
            "real_rows_with_estimated_physics": total_with_estimates,
        }
    metrics = evaluate_physical_latent_space(
        np.concatenate(embeddings, axis=0),
        np.concatenate(params, axis=0),
        k_neighbors=k_neighbors,
        pass_threshold=pass_threshold,
    )
    return {
        "status": "ok",
        "real_rows_seen": total_real_seen,
        "real_rows_with_estimated_physics": total_with_estimates,
        "note": "Uses partial particles2SNR-derived estimates when present; missing parameters are ignored per metric.",
        "metrics": metrics,
    }


def plot_physical_dashboard(metrics: dict[str, Any], output_base: Path) -> None:
    per_param = metrics.get("per_parameter", {})
    names = list(per_param)
    r2 = [float(per_param[name].get("linear_probe_r2", 0.0) or 0.0) for name in names]
    spearman = [float(per_param[name].get("spearman", 0.0) or 0.0) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))
    axes[0].bar(x, r2, color="#0072B2")
    axes[0].set_xticks(x, names, rotation=35, ha="right")
    axes[0].set_ylim(-0.1, 1.05)
    axes[0].set_title("Linear probe R2")
    axes[1].bar(x, spearman, color="#009E73")
    axes[1].set_xticks(x, names, rotation=35, ha="right")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_title("Spearman / circular score")
    fig.suptitle(f"Physical score: {float(metrics.get('physical_score', 0.0)):.3f}")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def train_hybrid(
    config: dict[str, Any],
    manifest: Path,
    output_dir: Path,
    profile: str,
    device: torch.device,
) -> tuple[MomentLikeReconstructor, list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    train_split = str(config["data"].get("split_train", "train"))
    val_split = str(config["data"].get("split_val", "val"))
    train_ds = make_dataset(config, manifest, train_split)
    val_ds = make_dataset(config, manifest, val_split) if read_manifest(manifest, split=val_split) else train_ds

    train_cfg = config["training"]
    batch_size = int(profile_value(train_cfg.get("batch_size", 16), profile))
    num_workers = int(train_cfg.get("num_workers", 0))
    stages = build_training_stages(config, profile)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
    recon_cfg = config["reconstruction_loss"]
    contrast_cfg = config["contrastive_loss"]
    invariance_cfg = config.get("invariance_loss", {})
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    best_physical_score = -float("inf")
    best_physical_metrics: dict[str, Any] = {}
    best_recon_metrics: dict[str, float] = {}
    history: list[dict[str, Any]] = []
    pool = str(config["model"].get("embedding_pool", "mean"))
    physics_cfg = config.get("physics_metrics", {})
    pass_threshold = float(physics_cfg.get("pass_threshold", 0.05))
    real_adaptation_cfg = config.get("real_adaptation", {})
    reference_root = real_adaptation_cfg.get("reconstruction_reference_root")
    val_reconstruction_reference = (
        load_reference_reconstruction_metrics(reference_root, "val")
        if reference_root
        else {"status": "missing"}
    )
    max_reconstruction_regression = float(real_adaptation_cfg.get("max_reconstruction_regression_fraction", 0.25))
    global_epoch = 0

    for stage in stages:
        for stage_epoch in range(1, stage.epochs + 1):
            global_epoch += 1
            batch_sampler = make_batch_sampler(
                train_ds,
                stage.synthetic_fraction,
                batch_size=batch_size,
                seed=int(config["experiment"].get("seed", 42)) + global_epoch * 1009,
            )
            if batch_sampler is None:
                train_loader = DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                )
            else:
                train_loader = DataLoader(
                    train_ds,
                    batch_sampler=batch_sampler,
                    num_workers=num_workers,
                )
            model.train()
            running: dict[str, float] = {
                "loss": 0.0,
                "reconstruction_loss": 0.0,
                "contrastive_loss": 0.0,
                "invariance_loss": 0.0,
            }
            batches = 0
            for batch in train_loader:
                signal = batch["signal"].to(device)
                token_mask = batch["token_mask"].to(device)
                time_mask = batch["target_time_mask"].to(device)
                physics_params = batch["physics_params"].to(device)
                contrastive_params = synthetic_only_physics_params(physics_params, batch.get("source_kind", []))
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    pred = model(signal, token_mask=token_mask)
                    recon_loss, _ = composite_reconstruction_loss(
                        pred,
                        signal,
                        time_mask,
                        lambda_signal=float(recon_cfg.get("lambda_signal", 1.0)),
                        lambda_derivative=float(recon_cfg.get("lambda_derivative", 0.2)),
                        lambda_energy=float(recon_cfg.get("lambda_energy", 0.05)),
                        huber_delta=float(recon_cfg.get("huber_delta", 1.0)),
                    )
                    embeddings = model.global_embedding(signal, token_mask=None, pool=pool)
                    contrast_loss = physical_contrastive_loss(
                        embeddings,
                        contrastive_params,
                        positive_distance=float(contrast_cfg.get("positive_distance", 0.18)),
                        negative_distance=float(contrast_cfg.get("negative_distance", 0.55)),
                        margin=float(contrast_cfg.get("margin", 1.0)),
                    )
                    invariance_weight = float(invariance_cfg.get("weight", 0.0))
                    if invariance_weight > 0.0:
                        augmented = positive_signal_augmentation(
                            signal,
                            noise_std_fraction=float(invariance_cfg.get("noise_std_fraction", 0.05)),
                            max_shift_points=int(invariance_cfg.get("max_shift_points", 8)),
                            amplitude_scale_min=float(invariance_cfg.get("amplitude_scale_min", 0.90)),
                            amplitude_scale_max=float(invariance_cfg.get("amplitude_scale_max", 1.10)),
                            phase_jitter_rad=float(invariance_cfg.get("phase_jitter_rad", 0.05)),
                        )
                        augmented_embeddings = model.global_embedding(augmented, token_mask=None, pool=pool)
                        invariance_loss = cosine_distance_loss(embeddings, augmented_embeddings)
                    else:
                        invariance_loss = embeddings.sum() * 0.0
                    loss = (
                        recon_loss
                        + float(contrast_cfg.get("weight", 0.1)) * contrast_loss
                        + invariance_weight * invariance_loss
                    )
                scaler.scale(loss).backward()
                if float(train_cfg.get("grad_clip_norm", 0.0)) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                running["loss"] += float(loss.detach().cpu())
                running["reconstruction_loss"] += float(recon_loss.detach().cpu())
                running["contrastive_loss"] += float(contrast_loss.detach().cpu())
                running["invariance_loss"] += float(invariance_loss.detach().cpu())
                batches += 1

            recon_metrics = evaluate_reconstruction(model, val_loader, device)
            emb, params = collect_physical_embeddings(model, val_loader, device, pool=pool)
            physical_metrics = evaluate_physical_latent_space(
                emb,
                params,
                k_neighbors=int(physics_cfg.get("k_neighbors", 5)),
                pass_threshold=pass_threshold,
            )
            recon_checkpoint_pass = reconstruction_selection_pass(
                recon_metrics,
                val_reconstruction_reference,
                max_regression_fraction=max_reconstruction_regression,
            )
            record: dict[str, Any] = {
                "epoch": global_epoch,
                "stage": stage.name,
                "stage_epoch": stage_epoch,
                "synthetic_fraction": stage.synthetic_fraction,
                **{key: value / max(batches, 1) for key, value in running.items()},
                **{f"val_{key}": value for key, value in recon_metrics.items()},
                "physical_score": physical_metrics.get("physical_score", 0.0),
                "physical_validation_pass": physical_metrics.get("physical_validation_pass", False),
                "checkpoint_reconstruction_pass": recon_checkpoint_pass,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True))
            state = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": global_epoch,
                "stage": stage.name,
                "metrics": record,
                "physical_metrics": physical_metrics,
                "reconstruction_metrics": recon_metrics,
            }
            torch.save(state, ckpt_dir / "latest.pt")
            score = float(physical_metrics.get("physical_score", 0.0))
            if score > best_physical_score:
                best_physical_score = score
                torch.save(state, ckpt_dir / "best_physical.pt")
            eligible = physical_metrics.get("physical_validation_pass", False) is True and recon_checkpoint_pass
            if eligible and score > best_score:
                best_score = score
                best_physical_metrics = physical_metrics
                best_recon_metrics = recon_metrics
                torch.save(state, ckpt_dir / "best.pt")
    best_checkpoint = ckpt_dir / "best.pt"
    if not best_checkpoint.is_file() and (ckpt_dir / "best_physical.pt").is_file():
        fallback = torch.load(ckpt_dir / "best_physical.pt", map_location="cpu")
        torch.save(fallback, best_checkpoint)
    if best_checkpoint.is_file():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        best_physical_metrics = checkpoint.get("physical_metrics", best_physical_metrics)
        best_recon_metrics = checkpoint.get("reconstruction_metrics", best_recon_metrics)
    return model, history, best_physical_metrics, best_recon_metrics


def write_summary(
    output_dir: Path,
    metrics: dict[str, Any],
    recon: dict[str, float],
    history: list[dict[str, Any]],
    manifest_summary: dict[str, Any],
) -> None:
    summary = {
        "physical_validation_pass": metrics.get("physical_validation_pass", False),
        "physical_score": metrics.get("physical_score", 0.0),
        "physical_metrics": metrics,
        "reconstruction_metrics_val": recon,
        "epochs": len(history),
        "hybrid_manifest_summary": manifest_summary,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False))
    lines = [
        "# Hybrid Physics Run Summary",
        "",
        f"- physical_validation_pass: {summary['physical_validation_pass']}",
        f"- physical_score: {float(summary['physical_score']):.6f}",
        f"- val_masked_mse: {recon.get('masked_mse')}",
        f"- val_event_masked_mse: {recon.get('event_region_masked_mse')}",
        f"- val_background_masked_mse: {recon.get('background_region_masked_mse')}",
        f"- epochs: {len(history)}",
        f"- hybrid_rows: {manifest_summary.get('total_rows')}",
        f"- hybrid_sources: {manifest_summary.get('by_source')}",
        f"- rows_with_any_physics_param: {manifest_summary.get('rows_with_any_physics_param')}",
        f"- rows_with_all_physics_params: {manifest_summary.get('rows_with_all_physics_params')}",
        "",
        "Classic retrieval/classification metrics are intentionally secondary and live under `classic_assessment/`.",
    ]
    (output_dir / "run_summary.md").write_text("\n".join(lines) + "\n")


@torch.no_grad()
def collect_classic_event_bundle(
    model: MomentLikeReconstructor,
    config: dict[str, Any],
    manifest: Path,
    device: torch.device,
    batch_size: int,
    max_real_rows: int | None,
    seed: int,
) -> tuple[EmbeddingBundle | None, np.ndarray | None, dict[str, Any]]:
    rows = [
        row
        for row in read_manifest(manifest)
        if (row.metadata or {}).get("source") == "real" and row.label_path is not None
    ]
    rows = select_rows_deterministic(rows, max_real_rows, seed=seed)
    input_length = int(config["data"]["input_length_ssl"])
    patch_cfg = config["patching"]
    spec = PatchSpec(
        input_length=input_length,
        patch_size=int(patch_cfg["patch_size"]),
        patch_stride=int(patch_cfg["patch_stride"]),
    )
    events = collect_events(rows, input_length=input_length, class_names=CLASS_NAMES)
    if not events:
        return None, None, {"status": "not_run", "reason": "no_labeled_real_events", "n_real_rows": len(rows)}
    classes = sorted(set(event.class_id for event in events))
    splits = sorted(set(event.split for event in events))
    if len(classes) < 2 or "train" not in splits or not ({"test", "val"} & set(splits)):
        return None, None, {
            "status": "not_run",
            "reason": "insufficient_class_or_split_coverage",
            "n_real_rows": len(rows),
            "n_events": len(events),
            "classes": classes,
            "splits": splits,
        }

    events_by_row: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for event in events:
        events_by_row[_event_key(event)].append(event)
    rows_with_events = [row for row in rows if _row_key(row) in events_by_row]
    embeddings: list[np.ndarray] = []
    event_signals: list[np.ndarray] = []
    ordered_events: list[Any] = []
    model.eval()
    for start in range(0, len(rows_with_events), batch_size):
        batch_rows = rows_with_events[start : start + batch_size]
        signals = [load_preprocessed_signal(row, config) for row in batch_rows]
        signal_tensor = torch.from_numpy(np.stack(signals)).float().unsqueeze(1).to(device)
        tokens = model.encode(signal_tensor, token_mask=None)
        for batch_i, row in enumerate(batch_rows):
            for event in events_by_row[_row_key(row)]:
                token_idx = token_indices_for_interval(event.start, event.end, spec)
                vector = pool_token_embeddings(tokens[batch_i], token_idx)
                embeddings.append(vector.detach().cpu().numpy().astype(np.float32))
                event_signals.append(signals[batch_i].astype(np.float32))
                ordered_events.append(event)

    metadata = event_records_to_metadata(ordered_events)
    bundle = EmbeddingBundle(
        key="hybrid_p3_ssl",
        display_name="Hybrid P3 SSL",
        embeddings=np.stack(embeddings).astype(np.float32),
        labels=metadata["class_id"].astype(np.int64),
        split=metadata["split"].astype(str),
        event_id=metadata["event_id"].astype(str),
        class_name=metadata["class_name"].astype(str),
    )
    status = {
        "status": "ok",
        "n_real_rows": len(rows),
        "n_events": len(ordered_events),
        "classes": {
            CLASS_NAMES.get(int(class_id), str(class_id)): int(np.sum(bundle.labels == int(class_id)))
            for class_id in sorted(set(bundle.labels.tolist()))
        },
        "splits": {
            split: int(np.sum(bundle.split == split))
            for split in sorted(set(bundle.split.tolist()))
        },
    }
    return bundle, np.stack(event_signals).astype(np.float32), status


def write_event_metadata_csv(output_dir: Path, bundle: EmbeddingBundle) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["event_id", "split", "class_id", "class_name"]
    with (output_dir / "hybrid_event_metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for event_id, split, class_id, class_name in zip(bundle.event_id, bundle.split, bundle.labels, bundle.class_name):
            writer.writerow(
                {
                    "event_id": str(event_id),
                    "split": str(split),
                    "class_id": int(class_id),
                    "class_name": str(class_name),
                }
            )


def run_classic_secondary_assessment(
    model: MomentLikeReconstructor,
    config: dict[str, Any],
    profile: str,
    manifest: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    classic_cfg = config.get("classic_assessment", {})
    if not bool(classic_cfg.get("enabled", True)):
        return {"status": "disabled"}
    seed = int(config["experiment"].get("seed", 42))
    max_real_rows_value = profile_value(classic_cfg.get("max_real_rows"), profile)
    max_real_rows = None if max_real_rows_value is None else int(max_real_rows_value)
    bundle, event_signals, status = collect_classic_event_bundle(
        model=model,
        config=config,
        manifest=manifest,
        device=device,
        batch_size=batch_size,
        max_real_rows=max_real_rows,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if bundle is None:
        (output_dir / "classic_assessment_summary.json").write_text(json.dumps(json_safe(status), indent=2, allow_nan=False))
        return status

    np.savez_compressed(
        output_dir / "hybrid_event_embeddings.npz",
        embeddings=bundle.embeddings,
        labels=bundle.labels,
        split=bundle.split,
        event_id=bundle.event_id,
        class_name=bundle.class_name,
    )
    write_event_metadata_csv(output_dir, bundle)
    bundles = [bundle]
    max_events_per_class = int(profile_value(classic_cfg.get("max_events_per_class", 500), profile))
    manifold = plot_manifold_figure(
        bundles,
        output_dir=output_dir,
        max_events_per_class=max_events_per_class,
        seed=seed,
        run_tsne=bool(profile_value(classic_cfg.get("run_tsne", True), profile)),
    )
    label_rows = run_label_efficiency(
        bundles,
        output_dir=output_dir,
        fractions=[float(value) for value in profile_value(classic_cfg.get("label_fractions", [0.1, 1.0]), profile)],
        repeats=int(profile_value(classic_cfg.get("probe_repeats", 3), profile)),
        seed=seed,
    )
    label_summary = summarize_probe_rows(label_rows)
    retrieval = plot_retrieval_sheet(
        bundles,
        output_dir=output_dir,
        signals=event_signals,
        metadata={},
        queries_per_class=int(classic_cfg.get("retrieval_queries_per_class", 2)),
        neighbors=int(classic_cfg.get("retrieval_neighbors", 5)),
        metric_max_per_class=int(profile_value(classic_cfg.get("retrieval_metric_max_per_class", 500), profile)),
        seed=seed,
    )
    robustness_placeholder = write_robustness_placeholder(
        output_dir,
        "Hybrid perturbation robustness is reported in the run-level robustness_metrics.json artifact.",
    )
    dashboard = write_assessment_dashboard(
        output_dir,
        bundles=bundles,
        manifold=manifold,
        label_summary=label_summary,
        retrieval=retrieval,
        robustness=robustness_placeholder,
    )
    summary = {
        **status,
        "manifold": manifold,
        "label_efficiency": label_summary,
        "retrieval": retrieval,
        "dashboard": dashboard,
    }
    (output_dir / "classic_assessment_summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False))
    return summary


def make_eval_loader(config: dict[str, Any], manifest: Path, split: str, batch_size: int, num_workers: int) -> DataLoader | None:
    if not read_manifest(manifest, split=split):
        return None
    return DataLoader(
        make_dataset(config, manifest, split),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


@torch.no_grad()
def evaluate_hybrid_physical_baselines(
    model: MomentLikeReconstructor,
    config: dict[str, Any],
    profile: str,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    baseline_cfg = config.get("baseline_assessment", {})
    if not bool(baseline_cfg.get("enabled", True)):
        return {"status": "disabled"}
    sweep_dir_value = profile_value(baseline_cfg.get("sweep_dir"), profile)
    if sweep_dir_value is None:
        return {"status": "not_run", "reason": "missing_sweep_dir"}
    sweep_dir = Path(str(sweep_dir_value))
    if not (sweep_dir / "synthetic_metadata.csv").is_file():
        return {"status": "not_run", "reason": "missing_sweep_artifacts", "sweep_dir": str(sweep_dir)}
    max_combined = profile_value(baseline_cfg.get("max_combined_samples"), profile)
    max_combined_samples = None if max_combined is None else int(max_combined)
    physics_cfg = config.get("physics_metrics", {})
    k_neighbors = int(physics_cfg.get("k_neighbors", 5))
    pass_threshold = float(physics_cfg.get("pass_threshold", 0.05))
    seed = int(baseline_cfg.get("random_seed", 123))
    reference = evaluate_sweep_directory(
        sweep_dir=sweep_dir,
        include_raw=True,
        include_random=True,
        k_neighbors=k_neighbors,
        random_seed=seed,
        max_combined_samples=max_combined_samples,
        pass_threshold=pass_threshold,
    )
    reconstruction_checkpoint_value = profile_value(baseline_cfg.get("reconstruction_only_checkpoint"), profile)
    reconstruction_model_name = str(baseline_cfg.get("reconstruction_only_model_name", "p3_ssl_reconstruction_only"))
    reconstruction_only_summary: dict[str, Any] = {"status": "not_run", "reason": "missing_checkpoint_config"}
    if reconstruction_checkpoint_value:
        reconstruction_checkpoint = Path(str(reconstruction_checkpoint_value))
        if reconstruction_checkpoint.is_file():
            old_model, old_input_length, old_pool = load_reconstruction_checkpoint_model(reconstruction_checkpoint, device)

            def encode_reconstruction_only_panel(signals: np.ndarray) -> np.ndarray:
                chunks: list[np.ndarray] = []
                old_model.eval()
                for start in range(0, signals.shape[0], batch_size):
                    batch = torch.from_numpy(signals[start : start + batch_size]).float().unsqueeze(1).to(device)
                    emb = old_model.global_embedding(batch, token_mask=None, pool=old_pool)
                    chunks.append(emb.detach().cpu().numpy().astype(np.float32))
                return np.concatenate(chunks, axis=0)

            reconstruction_only = evaluate_encoder_on_sweep_directory(
                sweep_dir=sweep_dir,
                encode_panel=encode_reconstruction_only_panel,
                model_name=reconstruction_model_name,
                input_length=old_input_length,
                k_neighbors=k_neighbors,
                max_combined_samples=max_combined_samples,
                seed=seed,
                pass_threshold=pass_threshold,
            )
            reference = merge_reference_and_candidate_rankings(reference, reconstruction_only)
            reconstruction_only_summary = {
                "status": "ok",
                "checkpoint": str(reconstruction_checkpoint),
                "model": reconstruction_model_name,
                "score": float(reconstruction_only["ranking_row"]["combined_physical_score"]),
            }
        else:
            reconstruction_only_summary = {
                "status": "not_run",
                "reason": "missing_checkpoint",
                "checkpoint": str(reconstruction_checkpoint),
                "model": reconstruction_model_name,
            }
    pool = str(config["model"].get("embedding_pool", "mean"))
    input_length = int(config["data"]["input_length_ssl"])

    def encode_panel(signals: np.ndarray) -> np.ndarray:
        chunks: list[np.ndarray] = []
        model.eval()
        for start in range(0, signals.shape[0], batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size]).float().unsqueeze(1).to(device)
            emb = model.global_embedding(batch, token_mask=None, pool=pool)
            chunks.append(emb.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)

    candidate = evaluate_encoder_on_sweep_directory(
        sweep_dir=sweep_dir,
        encode_panel=encode_panel,
        model_name="hybrid_p3_ssl",
        input_length=input_length,
        k_neighbors=k_neighbors,
        max_combined_samples=max_combined_samples,
        seed=seed,
        pass_threshold=pass_threshold,
    )
    merged = merge_reference_and_candidate_rankings(reference, candidate)
    candidate_score = float(candidate["ranking_row"]["combined_physical_score"])
    random_score = next((float(row["combined_physical_score"]) for row in merged["ranking"] if row["model"] == "random_embedding"), None)
    raw_score = next((float(row["combined_physical_score"]) for row in merged["ranking"] if row["model"] == "raw_signal"), None)
    reconstruction_only_score = next(
        (float(row["combined_physical_score"]) for row in merged["ranking"] if row["model"] == reconstruction_model_name),
        None,
    )
    merged["candidate_comparison"] = {
        "candidate_score": candidate_score,
        "random_score": random_score,
        "raw_score": raw_score,
        "reconstruction_only_score": reconstruction_only_score,
        "reconstruction_only_status": reconstruction_only_summary.get("status"),
        "beats_random": random_score is not None and candidate_score > random_score,
        "beats_raw": raw_score is not None and candidate_score > raw_score,
        "beats_reconstruction_only": reconstruction_only_score is not None and candidate_score > reconstruction_only_score,
    }
    baseline_dir = output_dir / "classic_assessment" / "physical_baselines"
    write_physical_evaluation_report(merged, baseline_dir)
    return {
        "status": "ok",
        "sweep_dir": str(sweep_dir),
        "report_dir": str(baseline_dir),
        "reconstruction_only_baseline": reconstruction_only_summary,
        "candidate_comparison": merged["candidate_comparison"],
        "ranking": merged["ranking"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P3 hybrid physical masked-learning pipeline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--real-manifest", type=Path, default=None)
    parser.add_argument("--simulation-source", default="internal")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--profile", choices=["smoke", "full", "long"], default="smoke")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["experiment"].get("seed", 42)))
    output_root = args.output_root or Path(config["paths"].get("output_root", "P3_SSL/outputs/runs"))
    run_name = f"hybrid_physics_{args.profile}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = config["simulation"]
    requested_sources = {item.strip() for item in str(args.simulation_source or sim_cfg.get("source", "internal")).split(",") if item.strip()}
    particles2snr_event_manifest = (
        sim_cfg.get("particles2snr_event_manifest")
        if "particles2snr_pipeline" in requested_sources or "particles2snr" in requested_sources
        else None
    )
    synthetic_manifest = generate_synthetic_manifest(
        output_dir=output_dir,
        n_samples=int(profile_value(sim_cfg["n_synthetic"], args.profile)),
        input_length=int(config["data"]["input_length_raw"]),
        seed=int(config["experiment"].get("seed", 42)),
        normalization=str(sim_cfg.get("normalization", "none")),
        include_two_particle=bool(sim_cfg.get("include_two_particle", True)),
    )
    hybrid_manifest = build_hybrid_manifest(
        synthetic_manifest=synthetic_manifest,
        real_manifest=args.real_manifest,
        output_path=output_dir / "hybrid_manifest.csv",
        max_real_rows=(
            None
            if profile_value(config["hybrid_sampling"].get("max_real_rows"), args.profile) is None
            else int(profile_value(config["hybrid_sampling"].get("max_real_rows"), args.profile))
        ),
        particles2snr_event_manifest=particles2snr_event_manifest,
    )

    device = torch.device(args.device)
    manifest_summary = summarize_hybrid_manifest(hybrid_manifest)

    model, history, physical_metrics, recon_metrics = train_hybrid(
        config=config,
        manifest=hybrid_manifest,
        output_dir=output_dir,
        profile=args.profile,
        device=device,
    )
    train_cfg = config["training"]
    eval_batch_size = int(profile_value(train_cfg.get("batch_size", 16), args.profile))
    eval_workers = int(train_cfg.get("num_workers", 0))
    val_loader = make_eval_loader(config, hybrid_manifest, str(config["data"].get("split_val", "val")), eval_batch_size, eval_workers)
    test_loader = make_eval_loader(config, hybrid_manifest, str(config["data"].get("split_test", "test")), eval_batch_size, eval_workers)
    if val_loader is not None:
        recon_metrics = evaluate_reconstruction_detailed(model, val_loader, device, real_only=False)
        real_val_metrics = evaluate_reconstruction_detailed(model, val_loader, device, real_only=True)
    else:
        real_val_metrics = {"samples": 0.0, "real_only": 1.0}
    if test_loader is not None:
        test_recon_metrics = evaluate_reconstruction_detailed(model, test_loader, device, real_only=False)
        real_test_metrics = evaluate_reconstruction_detailed(model, test_loader, device, real_only=True)
    else:
        test_recon_metrics = recon_metrics
        real_test_metrics = {"samples": 0.0, "real_only": 1.0}

    robustness_cfg = config.get("robustness_metrics", {})
    robustness_loader = val_loader or test_loader
    robustness_metrics = (
        embedding_robustness_metrics(
            model,
            robustness_loader,
            device=device,
            pool=str(config["model"].get("embedding_pool", "mean")),
            perturbations=list(robustness_cfg.get("perturbations", [])) or None,
            max_samples=(
                None
                if profile_value(robustness_cfg.get("max_real_samples"), args.profile) is None
                else int(profile_value(robustness_cfg.get("max_real_samples"), args.profile))
            ),
            real_only=True,
        )
        if robustness_loader is not None
        else {"status": "not_run", "reason": "no evaluation split"}
    )
    real_estimated_physics_metrics = evaluate_real_estimated_physics(
        model=model,
        loaders=[val_loader, test_loader],
        device=device,
        pool=str(config["model"].get("embedding_pool", "mean")),
        k_neighbors=int(config["physics_metrics"].get("k_neighbors", 5)),
        pass_threshold=float(config["physics_metrics"].get("pass_threshold", 0.05)),
    )
    real_adaptation_cfg = config.get("real_adaptation", {})
    reference_root = real_adaptation_cfg.get("reconstruction_reference_root")
    reconstruction_comparison = {
        "val": compare_reconstruction_metrics(
            real_val_metrics,
            load_reference_reconstruction_metrics(reference_root, "val") if reference_root else {"status": "missing"},
            max_regression_fraction=float(real_adaptation_cfg.get("max_reconstruction_regression_fraction", 0.25)),
        ),
        "test": compare_reconstruction_metrics(
            real_test_metrics,
            load_reference_reconstruction_metrics(reference_root, "test") if reference_root else {"status": "missing"},
            max_regression_fraction=float(real_adaptation_cfg.get("max_reconstruction_regression_fraction", 0.25)),
        ),
    }

    classic_dir = output_dir / "classic_assessment"
    classic_dir.mkdir(exist_ok=True)
    classic_secondary_summary = run_classic_secondary_assessment(
        model=model,
        config=config,
        profile=args.profile,
        manifest=hybrid_manifest,
        output_dir=classic_dir,
        device=device,
        batch_size=eval_batch_size,
    )
    physical_baseline_summary = evaluate_hybrid_physical_baselines(
        model=model,
        config=config,
        profile=args.profile,
        output_dir=output_dir,
        device=device,
        batch_size=eval_batch_size,
    )

    (output_dir / "training_history.json").write_text(json.dumps(json_safe(history), indent=2, allow_nan=False))
    (output_dir / "physical_metrics.json").write_text(json.dumps(json_safe(physical_metrics), indent=2, allow_nan=False))
    (output_dir / "reconstruction_metrics_val.json").write_text(
        json.dumps(json_safe({"all": recon_metrics, "real": real_val_metrics}), indent=2, allow_nan=False)
    )
    (output_dir / "reconstruction_metrics_test.json").write_text(
        json.dumps(json_safe({"all": test_recon_metrics, "real": real_test_metrics}), indent=2, allow_nan=False)
    )
    (output_dir / "robustness_metrics.json").write_text(json.dumps(json_safe(robustness_metrics), indent=2, allow_nan=False))
    (output_dir / "real_estimated_physics_metrics.json").write_text(
        json.dumps(json_safe(real_estimated_physics_metrics), indent=2, allow_nan=False)
    )
    (output_dir / "reconstruction_reference_comparison.json").write_text(
        json.dumps(json_safe(reconstruction_comparison), indent=2, allow_nan=False)
    )
    (classic_dir / "README.md").write_text(
        "Classic label-efficiency, retrieval, and manifold metrics are secondary diagnostics for this pipeline.\n"
        "Physical baseline ranking is included here as a reference comparison, not as a class-retrieval pass gate.\n"
    )
    plot_physical_dashboard(physical_metrics, output_dir / "physical_dashboard")
    write_summary(output_dir, physical_metrics, recon_metrics, history, manifest_summary)
    summary_path = output_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["physical_baseline_summary"] = physical_baseline_summary
    summary["classic_secondary_summary"] = classic_secondary_summary
    summary["real_estimated_physics_summary"] = real_estimated_physics_metrics
    summary["reconstruction_reference_comparison"] = reconstruction_comparison
    summary_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False))
    md_path = output_dir / "run_summary.md"
    with md_path.open("a") as f:
        f.write("\n## Physical Baselines\n\n")
        f.write(f"- status: {physical_baseline_summary.get('status')}\n")
        comparison = physical_baseline_summary.get("candidate_comparison", {})
        if comparison:
            f.write(f"- hybrid_p3_ssl_score: {comparison.get('candidate_score')}\n")
            f.write(f"- beats_random: {comparison.get('beats_random')}\n")
            f.write(f"- beats_raw: {comparison.get('beats_raw')}\n")
            f.write(f"- reconstruction_only_score: {comparison.get('reconstruction_only_score')}\n")
            f.write(f"- beats_reconstruction_only: {comparison.get('beats_reconstruction_only')}\n")
        f.write("\n## Secondary Classic Diagnostics\n\n")
        f.write(f"- status: {classic_secondary_summary.get('status')}\n")
        if classic_secondary_summary.get("status") == "ok":
            f.write(f"- n_events: {classic_secondary_summary.get('n_events')}\n")
            f.write(f"- classes: {classic_secondary_summary.get('classes')}\n")
        f.write("\n## Real Estimated Physics\n\n")
        f.write(f"- status: {real_estimated_physics_metrics.get('status')}\n")
        f.write(f"- real_rows_with_estimated_physics: {real_estimated_physics_metrics.get('real_rows_with_estimated_physics')}\n")
        metrics = real_estimated_physics_metrics.get("metrics", {})
        if metrics:
            f.write(f"- physical_score: {metrics.get('physical_score')}\n")
            f.write(f"- mean_spearman: {metrics.get('mean_spearman')}\n")
        f.write("\n## Reconstruction Reference\n\n")
        for split, result in reconstruction_comparison.items():
            f.write(f"- {split}_status: {result.get('status')}\n")
            if result.get("status") == "ok":
                f.write(f"- {split}_regression_pass: {result.get('reconstruction_regression_pass')}\n")
    assessment = assess_hybrid_run(output_dir)
    write_run_assessment(assessment, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "physical_score": physical_metrics.get("physical_score", 0.0)}))


if __name__ == "__main__":
    main()
