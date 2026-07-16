from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import validate_study_config
from .losses import masked_mse
from .masking import PatchSpec, build_patch_aligned_isolated_masks, build_ssl_masks
from .study_data import RealEventDataset, SimulatedLatentDataset, validate_study_dataset_contracts
from .study_data import CONTINUOUS_FACTORS
from .study_model import (
    YeastStudyModel,
    YeastStudyModelConfig,
    paired_nuisance_consistency,
    physics_supervision_loss,
)


VALID_CELLS = {"A1", "A2", "A3", "A4"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config_from_study(config: dict[str, Any]) -> YeastStudyModelConfig:
    model = config["model"]
    return YeastStudyModelConfig(
        input_length=int(config["data"]["input_length"]),
        patch_size=int(model["patch_size"]),
        patch_stride=int(model["patch_stride"]),
        d_model=int(model["d_model"]),
        n_heads=int(model["n_heads"]),
        n_layers=int(model["n_layers"]),
        dim_feedforward=int(model["dim_feedforward"]),
        dropout=float(model["dropout"]),
        activation=str(model["activation"]),
        max_tokens=int(model["max_tokens"]),
    )


def build_mask_batch(
    signals: torch.Tensor,
    event_masks: torch.Tensor,
    config: dict[str, Any],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if signals.ndim != 3 or signals.shape[1] != 1:
        raise ValueError("signals must have shape (batch, 1, length)")
    sampling_frequency = float(config["data"]["sampling_frequency_hz"])
    masking = config["masking"]
    model = config["model"]
    spec = PatchSpec(
        input_length=signals.shape[-1],
        patch_size=int(model["patch_size"]),
        patch_stride=int(model["patch_stride"]),
    )
    samples_per_ms = sampling_frequency / 1000.0
    target_masks = []
    token_masks = []
    hidden_masks = []
    for index in range(signals.shape[0]):
        common = {
            "signal": signals[index, 0].detach().cpu().numpy(),
            "spec": spec,
            "rng": np.random.default_rng(seed + index * 7919),
            "mask_ratio": float(masking["mask_ratio"]),
            "high_derivative_probability": float(
                masking.get("high_derivative_probability", 0.25)
            ),
            "event_mask": event_masks[index].detach().cpu().numpy(),
            "event_biased_probability": float(masking.get("event_biased_probability", 0.0)),
            "avoid_fully_hidden_events": bool(masking["avoid_fully_hidden_events"]),
            "max_event_hidden_fraction": float(masking["max_event_hidden_fraction"]),
            "max_mask_attempts": int(masking["max_mask_attempts"]),
        }
        strategy = str(masking.get("strategy", "time_blocks"))
        if strategy == "patch_aligned_isolated":
            result = build_patch_aligned_isolated_masks(
                **common,
                minimum_visible_tokens_between_masks=int(
                    masking.get("minimum_visible_tokens_between_masks", 1)
                ),
            )
        elif strategy == "time_blocks":
            result = build_ssl_masks(
                **common,
                min_block_length=int(
                    round(float(masking["min_block_ms"]) * samples_per_ms)
                ),
                max_block_length=int(
                    round(float(masking["max_block_ms"]) * samples_per_ms)
                ),
                guard_points=int(round(float(masking["guard_ms"]) * samples_per_ms)),
            )
        else:
            raise ValueError(f"Unsupported masking strategy: {strategy}")
        target_masks.append(result["target_time_mask"])
        token_masks.append(result["token_mask"])
        hidden_masks.append(result["token_time_mask"])
    return (
        torch.from_numpy(np.stack(target_masks)).bool(),
        torch.from_numpy(np.stack(token_masks)).bool(),
        torch.from_numpy(np.stack(hidden_masks)).bool(),
    )


def interpolation_baseline(signals: torch.Tensor, hidden_masks: torch.Tensor) -> torch.Tensor:
    outputs = []
    for signal, hidden in zip(signals[:, 0].cpu().numpy(), hidden_masks.cpu().numpy()):
        indices = np.arange(signal.size)
        visible = ~hidden
        if int(np.count_nonzero(visible)) < 2:
            reconstructed = np.zeros_like(signal)
        else:
            reconstructed = signal.copy()
            reconstructed[hidden] = np.interp(indices[hidden], indices[visible], signal[visible])
        outputs.append(reconstructed)
    return torch.from_numpy(np.stack(outputs)).to(signals.device).unsqueeze(1)


def nearest_baseline(signals: torch.Tensor, hidden_masks: torch.Tensor) -> torch.Tensor:
    outputs = []
    for signal, hidden in zip(signals[:, 0].cpu().numpy(), hidden_masks.cpu().numpy()):
        indices = np.arange(signal.size)
        visible_indices = indices[~hidden]
        if visible_indices.size == 0:
            outputs.append(np.zeros_like(signal))
            continue
        insertion = np.searchsorted(visible_indices, indices)
        left = visible_indices[np.clip(insertion - 1, 0, visible_indices.size - 1)]
        right = visible_indices[np.clip(insertion, 0, visible_indices.size - 1)]
        nearest = np.where(indices - left <= right - indices, left, right)
        reconstructed = signal.copy()
        reconstructed[hidden] = signal[nearest[hidden]]
        outputs.append(reconstructed)
    return torch.from_numpy(np.stack(outputs)).to(signals.device).unsqueeze(1)


def visible_mean_baseline(signals: torch.Tensor, hidden_masks: torch.Tensor) -> torch.Tensor:
    """Predict every sample with the per-signal mean of the visible context."""
    if signals.ndim != 3 or signals.shape[1] != 1:
        raise ValueError("signals must have shape (batch, 1, length)")
    if hidden_masks.shape != signals.shape[:1] + signals.shape[2:]:
        raise ValueError("hidden_masks must have shape (batch, length)")
    visible = (~hidden_masks.to(signals.device)).unsqueeze(1)
    count = visible.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (signals * visible).sum(dim=-1, keepdim=True) / count
    return mean.expand_as(signals)


def reconstruction_error_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    event_mask: torch.Tensor,
) -> dict[str, float | int]:
    """Return additive reconstruction statistics for exact dataset aggregation."""
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[1] != 1:
        raise ValueError("prediction and target must have shape (batch, 1, length)")
    expected_mask_shape = target.shape[:1] + target.shape[2:]
    if target_mask.shape != expected_mask_shape or event_mask.shape != expected_mask_shape:
        raise ValueError("target_mask and event_mask must have shape (batch, length)")
    target_mask = target_mask.to(device=target.device, dtype=torch.bool)
    event_mask = event_mask.to(device=target.device, dtype=torch.bool)
    squared_error = torch.square(prediction - target)[:, 0]
    prediction_squared = torch.square(prediction[:, 0])
    target_squared = torch.square(target[:, 0])
    event_target = target_mask & event_mask
    background_target = target_mask & ~event_mask

    def total(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float(values[mask].sum().detach().cpu())

    return {
        "squared_error_sum": total(squared_error, target_mask),
        "event_squared_error_sum": total(squared_error, event_target),
        "background_squared_error_sum": total(squared_error, background_target),
        "prediction_squared_sum": total(prediction_squared, target_mask),
        "target_squared_sum": total(target_squared, target_mask),
        "target_count": int(target_mask.sum().detach().cpu()),
        "event_target_count": int(event_target.sum().detach().cpu()),
        "background_target_count": int(background_target.sum().detach().cpu()),
    }


def _merge_reconstruction_components(
    destination: dict[str, float | int], source: dict[str, float | int]
) -> None:
    for key, value in source.items():
        destination[key] = destination.get(key, 0) + value


def _region_mse(components: dict[str, float | int], region: str) -> float | None:
    prefix = "" if region == "all" else f"{region}_"
    count = int(components[f"{prefix}target_count"])
    if count == 0:
        return None
    return float(components[f"{prefix}squared_error_sum"]) / count


def _simulation_loss(
    model: YeastStudyModel,
    batch: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    include_physics: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    signals = batch["signals"]
    event_masks = batch["event_masks"]
    batch_size, views, channels, length = signals.shape
    flat_signals = signals.reshape(batch_size * views, channels, length)
    flat_events = event_masks.reshape(batch_size * views, length)
    target_mask, token_mask, _ = build_mask_batch(flat_signals, flat_events, config, seed)
    output = model(flat_signals, token_mask.to(flat_signals.device))
    reconstruction = masked_mse(
        output["reconstruction"], flat_signals, target_mask.to(flat_signals.device)
    )
    total = reconstruction
    metrics = {"reconstruction": float(reconstruction.detach())}
    if include_physics:
        targets = batch["continuous_targets"].repeat_interleave(views, dim=0)
        valid = batch["continuous_valid"].repeat_interleave(views, dim=0)
        component = batch["component_target"].repeat_interleave(views, dim=0)
        continuous_loss, component_loss = physics_supervision_loss(output, targets, valid, component)
        consistency = paired_nuisance_consistency(
            output["embedding"].reshape(batch_size, views, -1)
        )
        weights = config["loss"]
        total = (
            total
            + float(weights["physics_continuous_weight"]) * continuous_loss
            + float(weights["component_count_weight"]) * component_loss
            + float(weights["nuisance_consistency_weight"]) * consistency
        )
        metrics.update(
            {
                "physics_continuous": float(continuous_loss.detach()),
                "component_count": float(component_loss.detach()),
                "nuisance_consistency": float(consistency.detach()),
            }
        )
    metrics["loss"] = float(total.detach())
    return total, metrics


def _real_loss(
    model: YeastStudyModel,
    batch: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    signals = batch["signal"]
    target_mask, token_mask, _ = build_mask_batch(signals, batch["event_mask"], config, seed)
    output = model(signals, token_mask.to(signals.device))
    loss = masked_mse(output["reconstruction"], signals, target_mask.to(signals.device))
    return loss, {"loss": float(loss.detach()), "reconstruction": float(loss.detach())}


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


@torch.no_grad()
def evaluate_reconstruction_controls(
    model: YeastStudyModel,
    loader: DataLoader,
    config: dict[str, Any],
    seed: int,
    simulation: bool,
    device: torch.device,
) -> dict[str, float | int | None]:
    model.eval()
    totals: dict[str, dict[str, float | int]] = {
        name: {} for name in ("model", "zero", "visible_mean", "interpolation", "nearest")
    }
    for batch_index, batch in enumerate(loader):
        if simulation:
            batch_size, views, channels, length = batch["signals"].shape
            signals = batch["signals"].reshape(batch_size * views, channels, length).to(device)
            events = batch["event_masks"].reshape(batch_size * views, length)
        else:
            signals = batch["signal"].to(device)
            events = batch["event_mask"]
        target_mask, token_mask, hidden_mask = build_mask_batch(
            signals.cpu(), events, config, seed + batch_index * 100_003
        )
        output = model(signals, token_mask.to(device))
        target_device = target_mask.to(device)
        interpolation = interpolation_baseline(signals, hidden_mask)
        nearest = nearest_baseline(signals, hidden_mask)
        predictions = {
            "model": output["reconstruction"],
            "zero": torch.zeros_like(signals),
            "visible_mean": visible_mean_baseline(signals, hidden_mask),
            "interpolation": interpolation,
            "nearest": nearest,
        }
        for name, prediction in predictions.items():
            _merge_reconstruction_components(
                totals[name],
                reconstruction_error_components(
                    prediction, signals, target_device, events.to(device)
                ),
            )

    result: dict[str, float | int | None] = {}
    for name, components in totals.items():
        result[f"{name}_masked_mse"] = _region_mse(components, "all")
        result[f"{name}_event_region_masked_mse"] = _region_mse(components, "event")
        result[f"{name}_background_region_masked_mse"] = _region_mse(
            components, "background"
        )
    model_components = totals["model"]
    target_count = int(model_components["target_count"])
    zero_mse = result["zero_masked_mse"]
    model_mse = result["model_masked_mse"]
    result.update(
        {
            "model_relative_improvement_vs_zero": (
                (zero_mse - model_mse) / zero_mse
                if zero_mse is not None and zero_mse > 0.0 and model_mse is not None
                else None
            ),
            "model_output_rms_on_mask": math.sqrt(
                float(model_components["prediction_squared_sum"]) / target_count
            ),
            "target_rms_on_mask": math.sqrt(
                float(model_components["target_squared_sum"]) / target_count
            ),
            "target_event_fraction": (
                int(model_components["event_target_count"]) / target_count
            ),
            "target_count": target_count,
        }
    )
    return result


@torch.no_grad()
def evaluate_physics_predictions(
    model: YeastStudyModel,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    squared_sums = np.zeros(len(CONTINUOUS_FACTORS), dtype=np.float64)
    prior_squared_sums = np.zeros(len(CONTINUOUS_FACTORS), dtype=np.float64)
    valid_counts = np.zeros(len(CONTINUOUS_FACTORS), dtype=np.int64)
    correct = 0
    total = 0
    component_counts = np.zeros(2, dtype=np.int64)
    consistencies: list[float] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        batch_size, views, channels, length = batch["signals"].shape
        signals = batch["signals"].reshape(batch_size * views, channels, length)
        output = model(signals, token_mask=None)
        targets = batch["continuous_targets"].repeat_interleave(views, dim=0)
        valid = batch["continuous_valid"].repeat_interleave(views, dim=0)
        squared = torch.square(output["continuous"] - targets)
        prior_squared = torch.square(torch.full_like(targets, 0.5) - targets)
        squared_sums += (squared * valid).sum(dim=0).cpu().numpy()
        prior_squared_sums += (prior_squared * valid).sum(dim=0).cpu().numpy()
        valid_counts += valid.sum(dim=0).cpu().numpy()
        component_target = batch["component_target"].repeat_interleave(views, dim=0)
        correct += int((output["component_logits"].argmax(dim=1) == component_target).sum())
        total += int(component_target.numel())
        component_counts += np.bincount(
            component_target.cpu().numpy(), minlength=component_counts.size
        )
        consistencies.append(
            float(
                paired_nuisance_consistency(
                    output["embedding"].reshape(batch_size, views, -1)
                )
            )
        )
    mse = {
        name: float(squared_sums[index] / max(valid_counts[index], 1))
        for index, name in enumerate(CONTINUOUS_FACTORS)
    }
    prior_mse = {
        name: float(prior_squared_sums[index] / max(valid_counts[index], 1))
        for index, name in enumerate(CONTINUOUS_FACTORS)
    }
    relative_reduction = {
        name: (
            float(1.0 - mse[name] / prior_mse[name])
            if prior_mse[name] > 0.0
            else None
        )
        for name in CONTINUOUS_FACTORS
    }
    model_component_accuracy = correct / max(total, 1)
    majority_component_accuracy = float(component_counts.max() / max(total, 1))
    return {
        "normalized_mse_by_factor": mse,
        "mean_normalized_mse": float(np.mean(list(mse.values()))),
        "constant_prior_normalized_mse_by_factor": prior_mse,
        "constant_prior_mean_normalized_mse": float(np.mean(list(prior_mse.values()))),
        "relative_mse_reduction_vs_constant_by_factor": relative_reduction,
        "component_count_accuracy": model_component_accuracy,
        "majority_component_count_accuracy": majority_component_accuracy,
        "component_count_accuracy_gain": (
            model_component_accuracy - majority_component_accuracy
        ),
        "component_count_support": component_counts.tolist(),
        "paired_nuisance_cosine_loss": float(np.mean(consistencies)),
        "n_view_predictions": total,
    }


@torch.no_grad()
def evaluate_embedding_health(
    model: YeastStudyModel,
    loader: DataLoader,
    *,
    simulation: bool,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    embeddings = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        if simulation:
            batch_size, views, channels, length = batch["signals"].shape
            signals = batch["signals"].reshape(batch_size * views, channels, length)
        else:
            signals = batch["signal"]
        embeddings.append(model(signals, token_mask=None)["embedding"].cpu().numpy())
    return embedding_health_statistics(np.concatenate(embeddings))


def embedding_health_statistics(embeddings: np.ndarray) -> dict[str, Any]:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("embeddings must have shape (n_samples, n_features)")
    centered = values - values.mean(axis=0, keepdims=True)
    standard_deviations = centered.std(axis=0)
    covariance = centered.T @ centered / max(len(values) - 1, 1)
    off_diagonal = covariance - np.diag(np.diag(covariance))
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variances = np.square(singular_values)
    proportions = variances / max(float(variances.sum()), 1.0e-12)
    positive = proportions[proportions > 0.0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-12)
    similarity = normalized @ normalized.T
    n = len(values)
    mean_off_diagonal = (
        float((similarity.sum() - np.trace(similarity)) / (n * (n - 1)))
        if n > 1
        else 1.0
    )
    return {
        "n_embeddings": n,
        "embedding_dimension": int(values.shape[1]),
        "mean_dimension_std": float(standard_deviations.mean()),
        "minimum_dimension_std": float(standard_deviations.min()),
        "mean_absolute_off_diagonal_covariance": float(np.mean(np.abs(off_diagonal))),
        "rms_off_diagonal_covariance": float(np.sqrt(np.mean(np.square(off_diagonal)))),
        "active_dimensions_std_gt_1e_3": int(np.count_nonzero(standard_deviations > 1.0e-3)),
        "effective_rank": effective_rank,
        "mean_off_diagonal_cosine_similarity": mean_off_diagonal,
        "mean_embedding_norm": float(np.linalg.norm(values, axis=1).mean()),
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _endless(loader: DataLoader) -> Iterable[dict[str, Any]]:
    while True:
        yield from loader


def train_study_cell(
    *,
    cell: str,
    config: dict[str, Any],
    real_root: Path,
    simulation_root: Path,
    output_dir: Path,
    profile: str,
    device: torch.device,
    init_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if cell not in VALID_CELLS:
        raise ValueError(f"Unsupported study cell: {cell}")
    validate_study_config(config)
    contract = validate_study_dataset_contracts(real_root, simulation_root)
    if not contract["valid"]:
        raise ValueError(f"Dataset contract failed: {contract['errors']}")
    training = config["training"]
    profile_config = training["profiles"][profile]
    seed = int(training["seed"])
    seed_everything(seed)
    model = YeastStudyModel(model_config_from_study(config)).to(device)
    if cell == "A4":
        if init_checkpoint is None:
            raise ValueError("A4 requires --init-checkpoint from A3")
        checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("cell") != "A3":
            raise ValueError("A4 initialization checkpoint must come from A3")
        if int(checkpoint.get("seed", -1)) != seed:
            raise ValueError("A4 and its A3 initialization checkpoint must use the same seed")
        model.load_state_dict(checkpoint["model_state"])
    batch_size = int(profile_config["batch_size"])
    max_real = profile_config.get("max_real_events")
    max_simulation = profile_config.get("max_simulation_latents")
    workers = int(profile_config["num_workers"])
    real_train = RealEventDataset(
        real_root, config["data"]["real_train_split"], max_events=max_real
    )
    real_validation = RealEventDataset(
        real_root, config["data"]["real_validation_split"], max_events=max_real
    )
    simulation_train = SimulatedLatentDataset(
        simulation_root, config["data"]["simulation_train_split"], max_latents=max_simulation
    )
    simulation_validation = SimulatedLatentDataset(
        simulation_root,
        config["data"]["simulation_validation_split"],
        max_latents=max_simulation,
    )
    generator = torch.Generator().manual_seed(seed)
    real_loader = DataLoader(
        real_train, batch_size=batch_size, shuffle=True, num_workers=workers, generator=generator
    )
    real_validation_loader = DataLoader(real_validation, batch_size=batch_size, shuffle=False)
    simulation_loader = DataLoader(
        simulation_train, batch_size=batch_size, shuffle=True, num_workers=workers, generator=generator
    )
    simulation_validation_loader = DataLoader(
        simulation_validation, batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    epochs = int(
        profile_config["adaptation_epochs"] if cell == "A4" else profile_config["epochs"]
    )
    for epoch in range(epochs):
        model.train()
        epoch_metrics: list[dict[str, float]] = []
        if cell in {"A1"}:
            for batch_index, raw_batch in enumerate(real_loader):
                batch = _move_batch(raw_batch, device)
                loss, metrics = _real_loss(model, batch, config, seed + epoch * 1_000_003 + batch_index)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip_norm"]))
                optimizer.step()
                epoch_metrics.append(metrics)
        elif cell in {"A2", "A3"}:
            for batch_index, raw_batch in enumerate(simulation_loader):
                batch = _move_batch(raw_batch, device)
                loss, metrics = _simulation_loss(
                    model,
                    batch,
                    config,
                    seed + epoch * 1_000_003 + batch_index,
                    include_physics=cell == "A3",
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip_norm"]))
                optimizer.step()
                epoch_metrics.append(metrics)
        else:
            real_iterator = _endless(real_loader)
            simulation_iterator = _endless(simulation_loader)
            for batch_index in range(max(len(real_loader), len(simulation_loader))):
                real_batch = _move_batch(next(real_iterator), device)
                simulation_batch = _move_batch(next(simulation_iterator), device)
                real_loss, real_metrics = _real_loss(
                    model, real_batch, config, seed + epoch * 1_000_003 + batch_index
                )
                simulation_loss, simulation_metrics = _simulation_loss(
                    model,
                    simulation_batch,
                    config,
                    seed + epoch * 1_000_003 + batch_index + 500_000,
                    include_physics=True,
                )
                replay = training["adaptation_replay"]
                progress = epoch / max(epochs - 1, 1)
                synthetic_weight = float(replay["synthetic_weight_start"]) + progress * (
                    float(replay["synthetic_weight_end"])
                    - float(replay["synthetic_weight_start"])
                )
                loss = (1.0 - synthetic_weight) * real_loss + synthetic_weight * simulation_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip_norm"]))
                optimizer.step()
                epoch_metrics.append(
                    {
                        "loss": float(loss.detach()),
                        "real_reconstruction": real_metrics["reconstruction"],
                        "simulation_reconstruction": simulation_metrics["reconstruction"],
                        "synthetic_replay_weight": synthetic_weight,
                    }
                )
        history.append({"epoch": epoch + 1, **_mean_metrics(epoch_metrics)})

    controls: dict[str, Any] = {}
    if cell in {"A1", "A4"}:
        controls["real"] = evaluate_reconstruction_controls(
            model,
            real_validation_loader,
            config,
            seed + 9_000_000,
            simulation=False,
            device=device,
        )
    if cell in {"A2", "A3", "A4"}:
        controls["simulation"] = evaluate_reconstruction_controls(
            model,
            simulation_validation_loader,
            config,
            seed + 9_500_000,
            simulation=True,
            device=device,
        )
    physics_validation = (
        evaluate_physics_predictions(model, simulation_validation_loader, device)
        if cell in {"A3", "A4"}
        else None
    )
    embedding_health: dict[str, Any] = {}
    if cell in {"A1", "A4"}:
        embedding_health["real"] = evaluate_embedding_health(
            model, real_validation_loader, simulation=False, device=device
        )
    if cell in {"A2", "A3", "A4"}:
        embedding_health["simulation"] = evaluate_embedding_health(
            model, simulation_validation_loader, simulation=True, device=device
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "cell": cell,
            "protocol": config["study"]["protocol"],
            "model_config": asdict(model.config),
            "masking": dict(config["masking"]),
            "model_state": model.state_dict(),
            "profile": profile,
            "seed": seed,
        },
        checkpoint_path,
    )
    result = {
        "cell": cell,
        "profile": profile,
        "seed": seed,
        "device": str(device),
        "masking": dict(config["masking"]),
        "contract": contract,
        "n_real_train": len(real_train),
        "n_real_validation": len(real_validation),
        "n_simulation_train_latents": len(simulation_train),
        "n_simulation_validation_latents": len(simulation_validation),
        "history": history,
        "validation_reconstruction_controls": controls,
        "validation_physics": physics_validation,
        "validation_embedding_health": embedding_health,
        "checkpoint": checkpoint_path.name,
        "sealed_splits_used": [],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
