from __future__ import annotations

import csv
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from particles2snr.ssl_realism_audit import load_particle_population

from .decimation import normalize_signal
from .losses import composite_reconstruction_loss
from .masking import (
    PatchSpec,
    build_balanced_event_mask_cycle,
    build_patch_aligned_isolated_masks,
)
from .models import MomentLikeConfig, MomentLikeReconstructor
from .study_training import (
    interpolation_baseline,
    nearest_baseline,
    visible_mean_baseline,
)


@dataclass(frozen=True)
class DatasetLimits:
    simulation_train: int | None
    simulation_validation: int | None
    real_validation_per_class: int | None


class SingleBeadSimulationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: Path,
        *,
        split: str,
        normalization: str,
        sampling_frequency_hz: float = 1_000_000.0,
        max_samples: int | None = None,
    ) -> None:
        self.signals = np.load(root / "signals.npy", mmap_mode="r")
        with (root / "simulation_metadata.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            self.rows = [
                row
                for row in csv.DictReader(handle)
                if row["split"] == split and int(row["component_count"]) == 1
            ]
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"No single-bead simulation rows for split={split}")
        self.normalization = normalization
        self.sampling_frequency_hz = float(sampling_frequency_hz)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        signal = np.asarray(
            self.signals[int(row["signal_row"])], dtype=np.float32
        ).copy()
        signal = normalize_signal(signal, mode=self.normalization)
        event_mask = np.zeros(signal.size, dtype=bool)
        center = float(row["event_position_fraction"]) * (signal.size - 1)
        half_width = (
            float(row["duration_ms"])
            / 1000.0
            * self.sampling_frequency_hz
            / 2.0
        )
        event_start = max(0, int(round(center - half_width)))
        event_end = min(signal.size, int(round(center + half_width)))
        event_mask[event_start:event_end] = True
        return {
            "signal": torch.from_numpy(signal).unsqueeze(0),
            "event_mask": torch.from_numpy(event_mask),
            "sample_index": index,
            "sample_id": f"{row['latent_id']}:view-{row['view_index']}",
            "class_name": "simulation",
        }


class RealBeadValidationDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: Path,
        *,
        split: str,
        normalization: str,
        max_per_class: int | None = None,
    ) -> None:
        populations = load_particle_population(root, split=split)
        self.rows = []
        for class_name in ("2um", "4um", "10um"):
            rows = populations[class_name]
            if max_per_class is not None:
                rows = rows[:max_per_class]
            self.rows.extend((class_name, row) for row in rows)
        if not self.rows:
            raise ValueError(f"No eligible real bead events for split={split}")
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        class_name, row = self.rows[index]
        signal = normalize_signal(row.signal, mode=self.normalization)
        event_mask = np.zeros(signal.size, dtype=bool)
        event_start = max(
            0,
            min(
                signal.size,
                int(round(float(row.metadata["event_start_index"]))),
            ),
        )
        event_end = max(
            0,
            min(
                signal.size,
                int(round(float(row.metadata["event_end_index"]))),
            ),
        )
        if event_end <= event_start:
            raise ValueError(
                f"Invalid real bead event bounds for {row.identifier}: "
                f"[{event_start}, {event_end})"
            )
        event_mask[event_start:event_end] = True
        return {
            "signal": torch.from_numpy(signal).unsqueeze(0),
            "event_mask": torch.from_numpy(event_mask),
            "sample_index": index,
            "sample_id": row.identifier,
            "class_name": class_name,
        }


def load_bead_ssl_config(path: Path) -> dict[str, Any]:
    from .config import load_config

    config = load_config(path)
    if config["study"]["protocol"] != "bead-ssl-comparison-v1":
        raise ValueError(
            "bead SSL is frozen on protocol bead-ssl-comparison-v1"
        )
    if config["study"]["simulation_dataset"] != "yeast-passage-simulations@v1":
        raise ValueError("bead SSL is frozen on yeast-passage-simulations@v1")
    if config["study"]["training_stage"] != "synthetic_only":
        raise ValueError("bead SSL training must remain synthetic-only")
    if config["model"]["mask_encoding"] != "sample_visibility_v1":
        raise ValueError("bead SSL comparison requires sample_visibility_v1")
    if config["masking"]["evaluation_policy"] != "P25":
        raise ValueError("all cells must use fixed P25 evaluation masks")
    if set(config["loss"]["cells"]) != {"B0", "B1", "B2", "B3"}:
        raise ValueError("loss cells must define B0, B1, B2, and B3")
    if "test" not in config["study"]["forbidden_splits"]:
        raise ValueError("test split must remain forbidden")
    return config


def configure_experiment(
    base_config: dict[str, Any],
    *,
    loss_cell: str,
    mask_policy: str,
    seed: int,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    if loss_cell not in config["loss"]["cells"]:
        raise ValueError(f"Unknown loss cell: {loss_cell}")
    if mask_policy not in {"P25", "CYCLIC25"}:
        raise ValueError(f"Unknown mask policy: {mask_policy}")
    config["loss"]["selected_cell"] = loss_cell
    config["masking"]["training_policy"] = mask_policy
    config["training"]["seed"] = int(seed)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(config: dict[str, Any]) -> MomentLikeReconstructor:
    data = config["data"]
    model = config["model"]
    return MomentLikeReconstructor(
        MomentLikeConfig(
            input_length=int(data["input_length"]),
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
    )


def build_p25_mask_batch(
    signals: torch.Tensor,
    sample_indices: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = config["model"]
    masking = config["masking"]
    spec = PatchSpec(
        input_length=signals.shape[-1],
        patch_size=int(model["patch_size"]),
        patch_stride=int(model["patch_stride"]),
    )
    target_masks = []
    token_masks = []
    for signal, sample_index in zip(
        signals[:, 0].detach().cpu().numpy(),
        sample_indices.detach().cpu().numpy(),
        strict=True,
    ):
        result = build_patch_aligned_isolated_masks(
            signal,
            spec,
            np.random.default_rng(seed + int(sample_index) * 7919),
            mask_ratio=float(masking["mask_ratio"]),
            event_mask=None,
            event_biased_probability=0.0,
            high_derivative_probability=float(
                masking["high_derivative_probability"]
            ),
            minimum_visible_tokens_between_masks=int(
                masking["minimum_visible_tokens_between_masks"]
            ),
        )
        target_masks.append(result["target_time_mask"])
        token_masks.append(result["token_mask"])
    return (
        torch.from_numpy(np.stack(target_masks)).bool(),
        torch.from_numpy(np.stack(token_masks)).bool(),
    )


def _expanded_event_bounds(
    event_mask: np.ndarray,
    *,
    minimum_points: int,
) -> tuple[int, int]:
    active = np.flatnonzero(np.asarray(event_mask, dtype=bool))
    if active.size == 0:
        raise ValueError("CYCLIC25 requires a non-empty simulated event")
    start = int(active[0])
    end = int(active[-1]) + 1
    missing = max(0, minimum_points - (end - start))
    start = max(0, start - missing // 2)
    end = min(event_mask.size, end + missing - missing // 2)
    if end - start < minimum_points:
        if start == 0:
            end = min(event_mask.size, minimum_points)
        elif end == event_mask.size:
            start = max(0, event_mask.size - minimum_points)
    return start, end


@lru_cache(maxsize=16_384)
def _cached_cyclic25_masks(
    *,
    input_length: int,
    event_start: int,
    event_end: int,
    seed: int,
    candidate_size: int,
    candidate_stride: int,
    event_windows_per_pass: int,
    background_windows_per_pass: int,
    require_context_each_side: bool,
) -> np.ndarray:
    event = np.zeros(input_length, dtype=bool)
    event[event_start:event_end] = True
    result = build_balanced_event_mask_cycle(
        event,
        PatchSpec(
            input_length=input_length,
            patch_size=candidate_size,
            patch_stride=candidate_stride,
        ),
        np.random.default_rng(seed),
        event_windows_per_pass=event_windows_per_pass,
        background_windows_per_pass=background_windows_per_pass,
        require_context_each_side=require_context_each_side,
    )
    masks = np.asarray(result["target_time_masks"], dtype=bool)
    masks.setflags(write=False)
    return masks


def build_cyclic25_mask_batch(
    event_masks: torch.Tensor,
    sample_indices: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
    cycle_step: int,
) -> torch.Tensor:
    selected: list[np.ndarray] = []
    for event_tensor, sample_index in zip(
        event_masks.detach().cpu().numpy(),
        sample_indices.detach().cpu().numpy(),
        strict=True,
    ):
        masks = build_cyclic25_masks_for_sample(
            event_tensor,
            int(sample_index),
            config,
            seed=seed,
        )
        selected.append(masks[int(cycle_step) % masks.shape[0]])
    return torch.from_numpy(np.stack(selected)).bool()


def build_cyclic25_masks_for_sample(
    event_mask: np.ndarray,
    sample_index: int,
    config: dict[str, Any],
    *,
    seed: int,
) -> np.ndarray:
    """Return every deterministic CYCLIC25 pass for one annotated event."""
    cyclic = config["masking"]["cyclic25"]
    event_windows = int(cyclic["event_windows_per_pass"])
    candidate_size = int(cyclic["candidate_size"])
    event_start, event_end = _expanded_event_bounds(
        np.asarray(event_mask, dtype=bool),
        minimum_points=event_windows * candidate_size,
    )
    return _cached_cyclic25_masks(
        input_length=np.asarray(event_mask).size,
        event_start=event_start,
        event_end=event_end,
        seed=seed + int(sample_index) * 7919,
        candidate_size=candidate_size,
        candidate_stride=int(cyclic["candidate_stride"]),
        event_windows_per_pass=event_windows,
        background_windows_per_pass=int(
            cyclic["background_windows_per_pass"]
        ),
        require_context_each_side=bool(cyclic["require_context_each_side"]),
    )


def build_training_mask_batch(
    batch: dict[str, Any],
    config: dict[str, Any],
    *,
    seed: int,
    cycle_step: int,
) -> torch.Tensor:
    policy = config["masking"]["training_policy"]
    if policy == "P25":
        target_mask, _ = build_p25_mask_batch(
            batch["signal"],
            batch["sample_index"],
            config,
            seed=seed,
        )
        return target_mask
    if policy == "CYCLIC25":
        return build_cyclic25_mask_batch(
            batch["event_mask"],
            batch["sample_index"],
            config,
            seed=seed,
            cycle_step=cycle_step,
        )
    raise ValueError(f"Unsupported training mask policy: {policy}")


def training_mask_seed(
    config: dict[str, Any],
    *,
    seed: int,
    epoch: int,
    batch_index: int,
) -> int:
    if config["masking"]["training_policy"] == "CYCLIC25":
        return int(seed)
    return int(seed + epoch * 100_003 + batch_index * 997)


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    selected = config["loss"]["selected_cell"]
    cell = config["loss"]["cells"][selected]
    return composite_reconstruction_loss(
        prediction,
        target,
        mask,
        lambda_signal=float(cell["lambda_signal"]),
        lambda_derivative=float(cell["lambda_derivative"]),
        lambda_energy=float(cell["lambda_energy"]),
        huber_delta=float(config["loss"]["huber_delta"]),
        normalize_energy_by_points=bool(
            config["loss"]["normalize_energy_by_points"]
        ),
    )


def _add_sums(destination: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        destination[key] = destination.get(key, 0.0) + float(value)


def _predictor_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    active = mask.to(device=target.device).unsqueeze(1)
    diff_active = (mask[..., 1:] | mask[..., :-1]).to(
        device=target.device
    ).unsqueeze(1)
    active_float = active.to(dtype=target.dtype)
    prediction_energy = (
        torch.square(prediction) * active_float
    ).sum(dim=-1)
    target_energy = (torch.square(target) * active_float).sum(dim=-1)
    relative_energy_error = torch.abs(
        prediction_energy - target_energy
    ) / (target_energy.abs() + 1.0e-12)
    return {
        "squared_error": float(
            torch.square(prediction - target)[active].sum().detach().cpu()
        ),
        "prediction_squared": float(
            torch.square(prediction)[active].sum().detach().cpu()
        ),
        "target_squared": float(
            torch.square(target)[active].sum().detach().cpu()
        ),
        "derivative_squared_error": float(
            torch.square(
                (prediction[..., 1:] - prediction[..., :-1])
                - (target[..., 1:] - target[..., :-1])
            )[diff_active]
            .sum()
            .detach()
            .cpu()
        ),
        "derivative_points": float(diff_active.sum().detach().cpu()),
        "relative_energy_error": float(
            relative_energy_error.sum().detach().cpu()
        ),
        "energy_samples": float(relative_energy_error.numel()),
        "points": float(active.sum().detach().cpu()),
    }


def _finalize_predictor(sums: dict[str, float]) -> dict[str, float]:
    points = max(float(sums.get("points", 0.0)), 1.0)
    target_squared = max(float(sums.get("target_squared", 0.0)), 1.0e-12)
    return {
        "masked_mse": float(sums.get("squared_error", 0.0)) / points,
        "masked_derivative_mse": float(
            sums.get("derivative_squared_error", 0.0)
        )
        / max(float(sums.get("derivative_points", 0.0)), 1.0),
        "relative_energy_error": float(
            sums.get("relative_energy_error", 0.0)
        )
        / max(float(sums.get("energy_samples", 0.0)), 1.0),
        "output_rms_fraction": float(
            np.sqrt(float(sums.get("prediction_squared", 0.0)) / target_squared)
        ),
        "masked_points": points,
    }


@torch.no_grad()
def evaluate_reconstruction(
    model: MomentLikeReconstructor,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    *,
    mask_seed: int,
    evaluation_policy: str = "P25",
    max_examples: int = 6,
    include_regions: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    if evaluation_policy not in {"P25", "CYCLIC25"}:
        raise ValueError(f"Unsupported evaluation policy: {evaluation_policy}")
    model.eval()
    sums: dict[str, dict[str, float]] = {
        name: {}
        for name in ("model", "zero", "visible_mean", "nearest", "interpolation")
    }
    region_sums: dict[str, dict[str, dict[str, float]]] = {
        name: {"event_support": {}, "background": {}}
        for name in sums
    }
    examples: dict[str, list[np.ndarray]] = {
        "signal": [],
        "mask": [],
        "model": [],
        "interpolation": [],
    }
    example_ids: list[str] = []
    def evaluate_batch(
        signal: torch.Tensor,
        target_mask: torch.Tensor,
        sample_ids: list[str],
        *,
        example_count: int | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> None:
        prediction = model(signal, time_mask=target_mask)
        hidden = target_mask
        predictors = {
            "model": prediction,
            "zero": torch.zeros_like(signal),
            "visible_mean": visible_mean_baseline(signal, hidden),
            "nearest": nearest_baseline(signal, hidden),
            "interpolation": interpolation_baseline(signal, hidden),
        }
        for name, values in predictors.items():
            _add_sums(
                sums[name],
                _predictor_sums(values, signal, target_mask),
            )
            if include_regions:
                if event_mask is None:
                    raise ValueError(
                        "regional reconstruction metrics require event masks"
                    )
                for region, region_mask in (
                    ("event_support", target_mask & event_mask),
                    ("background", target_mask & ~event_mask),
                ):
                    _add_sums(
                        region_sums[name][region],
                        _predictor_sums(values, signal, region_mask),
                    )
        remaining = max_examples - len(example_ids)
        if remaining > 0:
            take = min(
                remaining,
                signal.shape[0] if example_count is None else example_count,
            )
            examples["signal"].extend(signal[:take, 0].cpu().numpy())
            examples["mask"].extend(target_mask[:take].cpu().numpy())
            examples["model"].extend(prediction[:take, 0].cpu().numpy())
            examples["interpolation"].extend(
                predictors["interpolation"][:take, 0].cpu().numpy()
            )
            example_ids.extend(sample_ids[:take])

    for batch in loader:
        signal = batch["signal"].to(device)
        if evaluation_policy == "P25":
            target_mask, _ = build_p25_mask_batch(
                signal,
                batch["sample_index"],
                config,
                seed=mask_seed,
            )
            evaluate_batch(
                signal,
                target_mask.to(device),
                list(batch["sample_id"]),
                event_mask=(
                    batch["event_mask"].to(device)
                    if include_regions and "event_mask" in batch
                    else None
                ),
            )
            continue
        if "event_mask" not in batch:
            raise ValueError(
                "CYCLIC25 evaluation requires annotated event masks"
            )
        for item_index in range(signal.shape[0]):
            cycle = build_cyclic25_masks_for_sample(
                batch["event_mask"][item_index].detach().cpu().numpy(),
                int(batch["sample_index"][item_index]),
                config,
                seed=mask_seed,
            )
            cycle_mask = torch.from_numpy(np.array(cycle, copy=True)).to(
                device=device,
                dtype=torch.bool,
            )
            repeated_signal = signal[item_index : item_index + 1].expand(
                cycle_mask.shape[0], -1, -1
            )
            evaluate_batch(
                repeated_signal,
                cycle_mask,
                [str(batch["sample_id"][item_index])],
                example_count=1,
                event_mask=(
                    batch["event_mask"][item_index : item_index + 1]
                    .to(device)
                    .expand(cycle_mask.shape[0], -1)
                    if include_regions
                    else None
                ),
            )
    finalized = {name: _finalize_predictor(value) for name, value in sums.items()}
    if include_regions:
        for name in finalized:
            for region in ("event_support", "background"):
                finalized[name].update(
                    {
                        f"{region}_{metric}": value
                        for metric, value in _finalize_predictor(
                            region_sums[name][region]
                        ).items()
                    }
                )
    public_examples = {
        key: (
            np.stack(value)
            if value
            else np.empty((0,), dtype=np.float32)
        )
        for key, value in examples.items()
    }
    public_examples["sample_id"] = np.asarray(example_ids)
    return finalized, public_examples


def _profile(config: dict[str, Any], name: str) -> tuple[dict[str, Any], DatasetLimits]:
    profile = config["training"]["profiles"][name]
    return profile, DatasetLimits(
        simulation_train=profile["max_simulation_train"],
        simulation_validation=profile["max_simulation_validation"],
        real_validation_per_class=profile["max_real_validation_per_class"],
    )


def train_bead_ssl(
    config: dict[str, Any],
    *,
    simulation_root: Path | None,
    real_root: Path | None,
    output_dir: Path,
    profile_name: str,
    device_name: str,
    prepared_datasets: tuple[Dataset, Dataset, Dataset] | None = None,
    monitoring_datasets: tuple[Dataset, Dataset] | None = None,
    monitoring_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile, limits = _profile(config, profile_name)
    seed = int(config["training"]["seed"])
    seed_everything(seed)
    if prepared_datasets is None:
        if simulation_root is None or real_root is None:
            raise ValueError("legacy bead SSL requires simulation and real roots")
        train_dataset = SingleBeadSimulationDataset(
            simulation_root,
            split=config["data"]["simulation_train_split"],
            normalization=config["data"]["normalization"],
            sampling_frequency_hz=float(config["data"]["sampling_frequency_hz"]),
            max_samples=limits.simulation_train,
        )
        validation_dataset = SingleBeadSimulationDataset(
            simulation_root,
            split=config["data"]["simulation_validation_split"],
            normalization=config["data"]["normalization"],
            sampling_frequency_hz=float(config["data"]["sampling_frequency_hz"]),
            max_samples=limits.simulation_validation,
        )
        real_validation_dataset = RealBeadValidationDataset(
            real_root,
            split=config["data"]["real_validation_split"],
            normalization=config["data"]["normalization"],
            max_per_class=limits.real_validation_per_class,
        )
    else:
        train_dataset, validation_dataset, real_validation_dataset = prepared_datasets
    batch_size = int(profile["batch_size"])
    workers = int(profile["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    real_loader = DataLoader(
        real_validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    monitor_loaders: tuple[DataLoader, DataLoader] | None = None
    if monitoring_datasets is not None:
        monitor_loaders = tuple(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
            )
            for dataset in monitoring_datasets
        )
    device = torch.device(device_name)
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(config["training"]["amp"]) and device.type == "cuda",
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    initial_metrics, _ = evaluate_reconstruction(
        model,
        validation_loader,
        config,
        device,
        mask_seed=seed,
        max_examples=0,
    )
    matched_policy = str(config["masking"]["training_policy"])

    def matched_monitor() -> dict[str, Any] | None:
        if monitor_loaders is None:
            return None
        train_monitor, validation_monitor = monitor_loaders
        train_metrics, _ = evaluate_reconstruction(
            model,
            train_monitor,
            config,
            device,
            mask_seed=seed,
            evaluation_policy=matched_policy,
            max_examples=0,
        )
        validation_metrics, _ = evaluate_reconstruction(
            model,
            validation_monitor,
            config,
            device,
            mask_seed=seed,
            evaluation_policy=matched_policy,
            max_examples=0,
        )
        return {
            "evaluation_policy": matched_policy,
            "train_eval": train_metrics,
            "validation": validation_metrics,
        }

    initial_record: dict[str, Any] = {"epoch": 0, "validation": initial_metrics}
    initial_monitor = matched_monitor()
    if initial_monitor is not None:
        initial_record["matched_monitor"] = initial_monitor
    history: list[dict[str, Any]] = [initial_record]
    best_mse = float("inf")
    best_epoch = 0
    epochs = int(profile["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        component_sums: dict[str, float] = {}
        batches = 0
        for batch_index, batch in enumerate(train_loader):
            signal = batch["signal"].to(device)
            target_mask = build_training_mask_batch(
                batch,
                config,
                seed=training_mask_seed(
                    config,
                    seed=seed,
                    epoch=epoch,
                    batch_index=batch_index,
                ),
                cycle_step=epoch - 1,
            )
            target_mask = target_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=scaler.is_enabled()
            ):
                prediction = model(signal, time_mask=target_mask)
                loss, components = reconstruction_loss(
                    prediction,
                    signal,
                    target_mask,
                    config,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["grad_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            for name, value in components.items():
                component_sums[name] = component_sums.get(
                    name, 0.0
                ) + float(value.detach().cpu())
            batches += 1
        validation_metrics, _ = evaluate_reconstruction(
            model,
            validation_loader,
            config,
            device,
            mask_seed=seed,
            max_examples=0,
        )
        record = {
            "epoch": epoch,
            "train": {
                name: value / max(batches, 1)
                for name, value in component_sums.items()
            },
            "validation": validation_metrics,
        }
        epoch_monitor = matched_monitor()
        if epoch_monitor is not None:
            record["matched_monitor"] = epoch_monitor
        history.append(record)
        state = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "validation": validation_metrics,
        }
        torch.save(state, checkpoint_dir / "latest.pt")
        model_mse = validation_metrics["model"]["masked_mse"]
        if model_mse < best_mse:
            best_mse = model_mse
            best_epoch = epoch
            torch.save(state, checkpoint_dir / "best.pt")
        print(json.dumps(record, sort_keys=True))

    checkpoint_selection = str(
        config["training"].get("checkpoint_selection", "best_validation")
    )
    if checkpoint_selection not in {"best_validation", "fixed_final"}:
        raise ValueError(
            f"unsupported checkpoint selection: {checkpoint_selection}"
        )
    selected_checkpoint = (
        checkpoint_dir / "latest.pt"
        if checkpoint_selection == "fixed_final"
        else checkpoint_dir / "best.pt"
    )
    best = torch.load(
        selected_checkpoint,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best["model_state_dict"])
    simulation_metrics, simulation_examples = evaluate_reconstruction(
        model,
        validation_loader,
        config,
        device,
        mask_seed=seed,
    )
    real_metrics, real_examples = evaluate_reconstruction(
        model,
        real_loader,
        config,
        device,
        mask_seed=seed,
    )
    initial_mse = initial_metrics["model"]["masked_mse"]
    relative_improvement = (initial_mse - best_mse) / max(initial_mse, 1.0e-12)
    gate_cfg = config["promotion_gates"]
    optimization_gates = {
        "relative_improvement": (
            relative_improvement
            >= float(gate_cfg["fixed_validation_relative_improvement_min"])
        ),
        "beats_zero": (
            simulation_metrics["model"]["masked_mse"]
            < simulation_metrics["zero"]["masked_mse"]
        ),
        "beats_visible_mean": (
            simulation_metrics["model"]["masked_mse"]
            < simulation_metrics["visible_mean"]["masked_mse"]
        ),
        "nontrivial_output": (
            simulation_metrics["model"]["output_rms_fraction"]
            >= float(gate_cfg["output_rms_fraction_min"])
        ),
    }
    result = {
        "protocol": config["study"]["protocol"],
        "profile": profile_name,
        "seed": seed,
        "loss_cell": config["loss"]["selected_cell"],
        "training_mask_policy": config["masking"]["training_policy"],
        "evaluation_mask_policy": config["masking"]["evaluation_policy"],
        "mask_encoding": config["model"]["mask_encoding"],
        "best_epoch": best_epoch,
        "selected_epoch": int(best["epoch"]),
        "checkpoint_selection": checkpoint_selection,
        "counts": {
            "simulation_train": len(train_dataset),
            "simulation_validation": len(validation_dataset),
            "real_validation": len(real_validation_dataset),
            **(
                {
                    "matched_monitor_train": len(monitoring_datasets[0]),
                    "matched_monitor_validation": len(monitoring_datasets[1]),
                }
                if monitoring_datasets is not None
                else {}
            ),
        },
        "initial_validation_model_mse": initial_mse,
        "best_validation_model_mse": best_mse,
        "relative_improvement": relative_improvement,
        "simulation_validation": simulation_metrics,
        "real_validation": real_metrics,
        "gates": {
            "optimization": optimization_gates,
            "optimization_pass": all(optimization_gates.values()),
            "simulation_beats_interpolation": (
                simulation_metrics["model"]["masked_mse"]
                < simulation_metrics["interpolation"]["masked_mse"]
            ),
            "real_beats_zero": (
                real_metrics["model"]["masked_mse"]
                < real_metrics["zero"]["masked_mse"]
            ),
            "real_beats_visible_mean": (
                real_metrics["model"]["masked_mse"]
                < real_metrics["visible_mean"]["masked_mse"]
            ),
            "real_beats_interpolation": (
                real_metrics["model"]["masked_mse"]
                < real_metrics["interpolation"]["masked_mse"]
            ),
        },
        **({"monitoring": monitoring_metadata} if monitoring_metadata is not None else {}),
    }
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "simulation_reconstruction_examples.npz",
        **simulation_examples,
    )
    np.savez_compressed(
        output_dir / "real_reconstruction_examples.npz",
        **real_examples,
    )
    return result


def train_b0(
    config: dict[str, Any],
    *,
    simulation_root: Path,
    real_root: Path,
    output_dir: Path,
    profile_name: str,
    device_name: str,
) -> dict[str, Any]:
    """Compatibility wrapper for the waveform-only P25 experiment."""
    configured = configure_experiment(
        config,
        loss_cell="B0",
        mask_policy="P25",
        seed=int(config["training"]["seed"]),
    )
    return train_bead_ssl(
        configured,
        simulation_root=simulation_root,
        real_root=real_root,
        output_dir=output_dir,
        profile_name=profile_name,
        device_name=device_name,
    )
