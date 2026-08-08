from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .budding_synthetic_data import (
    BuddingSimulationDataset,
    budding_event_support_mask,
)
from .masking import PatchSpec, build_balanced_event_mask_cycle
from .models import MomentLikeConfig, MomentLikeReconstructor
from .study_data import RealEventDataset, validate_real_event_dataset_contract
from .study_training import (
    embedding_health_statistics,
    interpolation_baseline,
    nearest_baseline,
    visible_mean_baseline,
)


@dataclass(frozen=True)
class DatasetLimits:
    simulation_train_latents: int | None
    simulation_validation_latents: int | None
    real_validation_events: int | None
    mask_passes_per_source: int | None


@dataclass(frozen=True)
class _Source:
    signal_row: int
    sample_id: str
    packed_event_mask: np.ndarray
    latent_index: int
    view_index: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_budding_ssl_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["study"]["protocol"] != "yeast-budding-masked-learning-v1":
        raise ValueError("unexpected budding masked-learning protocol")
    if config["study"]["simulation_dataset"] != "yeast-budding-simulations-data@v1":
        raise ValueError("primary budding training must use the data-oriented dataset")
    if config["study"]["real_dataset"] != "yeast-events-representation@v3":
        raise ValueError("budding real evaluation is frozen on yeast-events-representation@v3")
    if config["study"]["training_stage"] != "synthetic_only":
        raise ValueError("budding masked-learning training must remain synthetic-only")
    if set(config["study"]["forbidden_splits"]) != {
        "in_session_test",
        "sealed_acquisition_test",
        "followup_test",
        "test",
    }:
        raise ValueError("all sealed/test aliases must remain forbidden")
    if config["data"]["real_source_group"] != "budding":
        raise ValueError("real evaluation must be filtered on source_group=budding")
    if config["data"]["real_validation_split"] != "development_validation":
        raise ValueError("real zero-shot evaluation must use development_validation")
    if config["model"]["mask_encoding"] != "sample_visibility_v1":
        raise ValueError("budding training requires the validated sample visibility encoding")
    masking = config["masking"]
    if (
        int(masking["candidate_size"]),
        int(masking["candidate_stride"]),
        int(masking["event_windows_per_pass"]),
        int(masking["background_windows_per_pass"]),
    ) != (16, 8, 3, 3):
        raise ValueError("budding masking is frozen on 16/8 windows with a balanced 3+3 pass")
    if bool(masking["require_context_each_side"]):
        raise ValueError("edge events must not be excluded for missing one-sided context")
    loss = config["loss"]
    if float(loss["event_weight"]) != 0.8 or float(loss["background_weight"]) != 0.2:
        raise ValueError("budding loss is frozen on 0.8 event + 0.2 background")
    early_stopping = config["training"]["early_stopping"]
    if early_stopping["metric"] != "simulation_validation.model.balanced_loss":
        raise ValueError("budding checkpoint selection must use balanced validation loss")
    if int(early_stopping["patience"]) < 1:
        raise ValueError("early-stopping patience must be positive")
    if float(early_stopping["min_delta"]) < 0.0:
        raise ValueError("early-stopping min_delta must be non-negative")
    return config


def make_model(config: dict[str, Any]) -> MomentLikeReconstructor:
    model = config["model"]
    data = config["data"]
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


class BalancedBuddingMaskDataset(Dataset[dict[str, Any]]):
    """Flatten deterministic complete mask cycles into train/evaluation examples."""

    def __init__(
        self,
        *,
        signals: np.ndarray,
        sources: list[_Source],
        config: dict[str, Any],
        seed: int,
        max_passes_per_source: int | None,
    ) -> None:
        self.signals = signals
        self.sources = sources
        self.input_length = int(config["data"]["input_length"])
        masking = config["masking"]
        self.spec = PatchSpec(
            input_length=self.input_length,
            patch_size=int(masking["candidate_size"]),
            patch_stride=int(masking["candidate_stride"]),
        )
        self.cycles: list[dict[str, np.ndarray]] = []
        self.examples: list[tuple[int, int]] = []
        self.full_mask_pass_examples = 0
        self.missing_left_context_sources = 0
        self.missing_right_context_sources = 0
        for source_index, source in enumerate(sources):
            event = np.unpackbits(
                source.packed_event_mask,
                count=self.input_length,
            ).astype(bool)
            cycle = build_balanced_event_mask_cycle(
                event,
                self.spec,
                np.random.default_rng(seed + source.signal_row * 7_919),
                event_windows_per_pass=int(masking["event_windows_per_pass"]),
                background_windows_per_pass=int(
                    masking["background_windows_per_pass"]
                ),
                require_context_each_side=bool(
                    masking["require_context_each_side"]
                ),
            )
            compact = {
                "event": np.asarray(
                    cycle["pass_event_window_indices"], dtype=np.int16
                ),
                "background": np.asarray(
                    cycle["pass_background_window_indices"], dtype=np.int16
                ),
                "coverage": np.asarray(
                    cycle["cumulative_event_window_coverage"], dtype=np.float32
                ),
                "context": np.asarray(
                    cycle["context_window_indices"], dtype=np.int16
                ),
            }
            self.cycles.append(compact)
            pass_count = int(compact["event"].shape[0])
            self.full_mask_pass_examples += pass_count
            selected_count = (
                pass_count
                if max_passes_per_source is None
                else min(pass_count, max_passes_per_source)
            )
            self.examples.extend(
                (source_index, pass_index)
                for pass_index in range(selected_count)
            )
            self.missing_left_context_sources += int(compact["context"][0] < 0)
            self.missing_right_context_sources += int(compact["context"][1] < 0)
        if not self.examples:
            raise ValueError("balanced budding dataset contains no mask passes")

    @classmethod
    def synthetic(
        cls,
        root: Path,
        split: str,
        *,
        config: dict[str, Any],
        seed: int,
        max_latents: int | None,
        max_passes_per_source: int | None,
    ) -> "BalancedBuddingMaskDataset":
        base = BuddingSimulationDataset(
            root,
            split,
            expected_dataset_id=config["study"]["simulation_dataset"],
            max_latents=max_latents,
            support_threshold=float(config["masking"]["support_relative_threshold"]),
        )
        sources: list[_Source] = []
        for latent_index, rows in enumerate(base.latent_rows):
            for row in rows:
                event = budding_event_support_mask(
                    row,
                    length=int(config["data"]["input_length"]),
                    sampling_frequency_hz=float(
                        config["data"]["sampling_frequency_hz"]
                    ),
                    relative_threshold=float(
                        config["masking"]["support_relative_threshold"]
                    ),
                )
                sources.append(
                    _Source(
                        signal_row=int(row["signal_row"]),
                        sample_id=(
                            f"{row['latent_id']}:view-{int(row['view_index'])}"
                        ),
                        packed_event_mask=np.packbits(event),
                        latent_index=latent_index,
                        view_index=int(row["view_index"]),
                    )
                )
        return cls(
            signals=base.signals,
            sources=sources,
            config=config,
            seed=seed,
            max_passes_per_source=max_passes_per_source,
        )

    @classmethod
    def real(
        cls,
        root: Path,
        split: str,
        *,
        config: dict[str, Any],
        seed: int,
        max_events: int | None,
        max_passes_per_source: int | None,
    ) -> "BalancedBuddingMaskDataset":
        contract = validate_real_event_dataset_contract(root)
        if not contract["valid"]:
            raise ValueError(f"real dataset contract failed: {contract['errors']}")
        base = RealEventDataset(
            root,
            split,
            max_events=max_events,
            source_groups=(str(config["data"]["real_source_group"]),),
        )
        sources: list[_Source] = []
        input_length = int(config["data"]["input_length"])
        for index, row in enumerate(base.rows):
            event = np.zeros(input_length, dtype=bool)
            start = max(
                0,
                min(
                    input_length,
                    int(round(float(row["event_start_input_index"]))),
                ),
            )
            end = max(
                0,
                min(
                    input_length,
                    int(round(float(row["event_end_input_index"]))),
                ),
            )
            event[start:end] = True
            sources.append(
                _Source(
                    signal_row=int(row["signal_row"]),
                    sample_id=str(row["event_id"]),
                    packed_event_mask=np.packbits(event),
                    latent_index=index,
                    view_index=0,
                )
            )
        return cls(
            signals=base.signals,
            sources=sources,
            config=config,
            seed=seed,
            max_passes_per_source=max_passes_per_source,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def source_signal(self, source_index: int) -> torch.Tensor:
        source = self.sources[source_index]
        signal = np.array(
            self.signals[source.signal_row],
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(signal).unsqueeze(0)

    def source_event_mask(self, source_index: int) -> np.ndarray:
        return np.unpackbits(
            self.sources[source_index].packed_event_mask,
            count=self.input_length,
        ).astype(bool)

    def _window_mask(self, indices: np.ndarray) -> np.ndarray:
        result = np.zeros(self.input_length, dtype=bool)
        for index in np.asarray(indices).reshape(-1):
            start, end = self.spec.spans[int(index)]
            result[start:end] = True
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_index, pass_index = self.examples[index]
        source = self.sources[source_index]
        cycle = self.cycles[source_index]
        event_target = self._window_mask(cycle["event"][pass_index])
        background_target = self._window_mask(cycle["background"][pass_index])
        if np.any(event_target & background_target):
            raise RuntimeError("event and background targets overlap")
        target = event_target | background_target
        return {
            "signal": self.source_signal(source_index),
            "event_support_mask": torch.from_numpy(
                self.source_event_mask(source_index)
            ),
            "event_target_mask": torch.from_numpy(event_target),
            "background_target_mask": torch.from_numpy(background_target),
            "target_mask": torch.from_numpy(target),
            "sample_id": source.sample_id,
            "source_index": source_index,
            "latent_index": source.latent_index,
            "view_index": source.view_index,
            "pass_index": pass_index,
        }

    def summary(self) -> dict[str, int]:
        return {
            "sources": len(self.sources),
            "selected_mask_pass_examples": len(self),
            "full_mask_pass_examples": self.full_mask_pass_examples,
            "missing_left_context_sources": self.missing_left_context_sources,
            "missing_right_context_sources": self.missing_right_context_sources,
        }


def balanced_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    event_target_mask: torch.Tensor,
    background_target_mask: torch.Tensor,
    *,
    event_weight: float,
    background_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    squared = torch.square(prediction - target)[:, 0]
    event_mask = event_target_mask.to(device=squared.device, dtype=torch.bool)
    background_mask = background_target_mask.to(
        device=squared.device, dtype=torch.bool
    )
    event_mse = squared[event_mask].mean()
    background_mse = squared[background_mask].mean()
    total = event_weight * event_mse + background_weight * background_mse
    return total, {
        "loss": total,
        "event_mse": event_mse,
        "background_mse": background_mse,
        "raw_masked_mse": squared[event_mask | background_mask].mean(),
    }


def balanced_validation_loss(
    *,
    event_mse: float,
    background_mse: float,
    event_weight: float,
    background_weight: float,
) -> float:
    return (
        event_weight * event_mse
        + background_weight * background_mse
    )


def validation_improved(
    current: float,
    best: float,
    *,
    min_delta: float,
) -> bool:
    return current < best - min_delta


def _merge_region_sums(
    destination: dict[str, float],
    prediction: torch.Tensor,
    target: torch.Tensor,
    event_mask: torch.Tensor,
    background_mask: torch.Tensor,
) -> None:
    squared = torch.square(prediction - target)[:, 0]
    prediction_squared = torch.square(prediction[:, 0])
    target_squared = torch.square(target[:, 0])
    regions = {
        "event": event_mask.to(device=target.device, dtype=torch.bool),
        "background": background_mask.to(device=target.device, dtype=torch.bool),
    }
    regions["all"] = regions["event"] | regions["background"]
    for name, mask in regions.items():
        destination[f"{name}_squared_error"] = destination.get(
            f"{name}_squared_error", 0.0
        ) + float(squared[mask].sum().detach().cpu())
        destination[f"{name}_count"] = destination.get(
            f"{name}_count", 0.0
        ) + float(mask.sum().detach().cpu())
    mask = regions["all"]
    destination["prediction_squared"] = destination.get(
        "prediction_squared", 0.0
    ) + float(prediction_squared[mask].sum().detach().cpu())
    destination["target_squared"] = destination.get(
        "target_squared", 0.0
    ) + float(target_squared[mask].sum().detach().cpu())


def _finalize_regions(
    sums: dict[str, float],
    *,
    event_weight: float,
    background_weight: float,
) -> dict[str, float]:
    result = {
        f"{name}_mse": sums[f"{name}_squared_error"]
        / max(sums[f"{name}_count"], 1.0)
        for name in ("all", "event", "background")
    }
    result["balanced_loss"] = balanced_validation_loss(
        event_mse=result["event_mse"],
        background_mse=result["background_mse"],
        event_weight=event_weight,
        background_weight=background_weight,
    )
    result["output_rms_fraction"] = float(
        np.sqrt(
            sums["prediction_squared"]
            / max(sums["target_squared"], 1.0e-12)
        )
    )
    result["masked_points"] = sums["all_count"]
    return result


@torch.no_grad()
def evaluate_reconstruction(
    model: MomentLikeReconstructor,
    loader: DataLoader,
    device: torch.device,
    *,
    max_examples: int,
    event_weight: float,
    background_weight: float,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    model.eval()
    sums: dict[str, dict[str, float]] = {
        name: {}
        for name in (
            "model",
            "zero",
            "visible_mean",
            "nearest",
            "interpolation",
        )
    }
    examples: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "signal",
            "event_support_mask",
            "event_target_mask",
            "background_target_mask",
            "model",
            "zero",
            "visible_mean",
            "nearest",
            "interpolation",
        )
    }
    sample_ids: list[str] = []
    pass_indices: list[int] = []
    for batch in loader:
        signal = batch["signal"].to(device)
        target_mask = batch["target_mask"].to(device)
        event_target = batch["event_target_mask"].to(device)
        background_target = batch["background_target_mask"].to(device)
        prediction = model(signal, time_mask=target_mask)
        predictors = {
            "model": prediction,
            "zero": torch.zeros_like(signal),
            "visible_mean": visible_mean_baseline(signal, target_mask),
            "nearest": nearest_baseline(signal, target_mask),
            "interpolation": interpolation_baseline(signal, target_mask),
        }
        for name, values in predictors.items():
            _merge_region_sums(
                sums[name],
                values,
                signal,
                event_target,
                background_target,
            )
        remaining = max_examples - len(sample_ids)
        if remaining > 0:
            take = min(remaining, signal.shape[0])
            examples["signal"].extend(signal[:take, 0].cpu().numpy())
            examples["event_support_mask"].extend(
                batch["event_support_mask"][:take].cpu().numpy()
            )
            examples["event_target_mask"].extend(
                event_target[:take].cpu().numpy()
            )
            examples["background_target_mask"].extend(
                background_target[:take].cpu().numpy()
            )
            for name, values in predictors.items():
                examples[name].extend(values[:take, 0].cpu().numpy())
            sample_ids.extend(list(batch["sample_id"][:take]))
            pass_indices.extend(
                [int(value) for value in batch["pass_index"][:take]]
            )
    arrays = {
        key: np.stack(values) if values else np.empty((0,), dtype=np.float32)
        for key, values in examples.items()
    }
    arrays["sample_id"] = np.asarray(sample_ids)
    arrays["pass_index"] = np.asarray(pass_indices, dtype=np.int64)
    return (
        {
            name: _finalize_regions(
                value,
                event_weight=event_weight,
                background_weight=background_weight,
            )
            for name, value in sums.items()
        },
        arrays,
    )


@torch.no_grad()
def evaluate_embedding_health(
    model: MomentLikeReconstructor,
    dataset: BalancedBuddingMaskDataset,
    device: torch.device,
    *,
    batch_size: int,
    paired_views: bool,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    embeddings: list[np.ndarray] = []
    for start in range(0, len(dataset.sources), batch_size):
        values = torch.stack(
            [
                dataset.source_signal(index)
                for index in range(
                    start,
                    min(start + batch_size, len(dataset.sources)),
                )
            ]
        ).to(device)
        embeddings.append(model.global_embedding(values).cpu().numpy())
    matrix = np.concatenate(embeddings)
    result = embedding_health_statistics(matrix)
    if paired_views:
        if len(matrix) % 2:
            raise ValueError("paired synthetic embeddings require an even source count")
        paired = matrix.reshape(-1, 2, matrix.shape[1])

        def cosine(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            numerator = np.sum(first * second, axis=1)
            denominator = np.maximum(
                np.linalg.norm(first, axis=1)
                * np.linalg.norm(second, axis=1),
                1.0e-12,
            )
            return numerator / denominator

        pair_cosine = cosine(paired[:, 0], paired[:, 1])
        shuffled = paired[np.random.default_rng(seed).permutation(len(paired)), 1]
        shuffled_cosine = cosine(paired[:, 0], shuffled)
        result.update(
            {
                "paired_view_mean_cosine_similarity": float(pair_cosine.mean()),
                "shuffled_view_mean_cosine_similarity": float(
                    shuffled_cosine.mean()
                ),
                "paired_view_cosine_gain": float(
                    pair_cosine.mean() - shuffled_cosine.mean()
                ),
            }
        )
    return result


def _profile(
    config: dict[str, Any], name: str
) -> tuple[dict[str, Any], DatasetLimits]:
    profile = config["training"]["profiles"][name]
    return profile, DatasetLimits(
        simulation_train_latents=profile["max_simulation_train_latents"],
        simulation_validation_latents=profile[
            "max_simulation_validation_latents"
        ],
        real_validation_events=profile["max_real_validation_events"],
        mask_passes_per_source=profile["max_mask_passes_per_source"],
    )


def train_budding_ssl(
    config: dict[str, Any],
    *,
    simulation_root: Path,
    real_root: Path,
    output_dir: Path,
    profile_name: str,
    device_name: str,
) -> dict[str, Any]:
    profile, limits = _profile(config, profile_name)
    seed = int(config["training"]["seed"])
    _seed_everything(seed)
    train_dataset = BalancedBuddingMaskDataset.synthetic(
        simulation_root,
        config["data"]["simulation_train_split"],
        config=config,
        seed=seed,
        max_latents=limits.simulation_train_latents,
        max_passes_per_source=limits.mask_passes_per_source,
    )
    simulation_validation = BalancedBuddingMaskDataset.synthetic(
        simulation_root,
        config["data"]["simulation_validation_split"],
        config=config,
        seed=seed + 1_000_003,
        max_latents=limits.simulation_validation_latents,
        max_passes_per_source=limits.mask_passes_per_source,
    )
    real_validation = BalancedBuddingMaskDataset.real(
        real_root,
        config["data"]["real_validation_split"],
        config=config,
        seed=seed + 2_000_003,
        max_events=limits.real_validation_events,
        max_passes_per_source=limits.mask_passes_per_source,
    )
    batch_size = int(profile["batch_size"])
    workers = int(profile["num_workers"])
    generator = torch.Generator().manual_seed(seed)
    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            generator=generator,
        ),
        "simulation_validation": DataLoader(
            simulation_validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        ),
        "real_validation": DataLoader(
            real_validation,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
        ),
    }
    device = torch.device(device_name)
    if profile_name == "smoke" and device.type != "cpu":
        raise ValueError("the integration smoke is intentionally CPU-only")
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
    event_weight = float(config["loss"]["event_weight"])
    background_weight = float(config["loss"]["background_weight"])
    early_stopping = config["training"]["early_stopping"]
    patience = int(early_stopping["patience"])
    min_delta = float(early_stopping["min_delta"])
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    initial_metrics, _ = evaluate_reconstruction(
        model,
        loaders["simulation_validation"],
        device,
        max_examples=0,
        event_weight=event_weight,
        background_weight=background_weight,
    )
    initial_state = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "coverage_cycle": 0,
        "validation": initial_metrics,
        "selection_metric": {
            "name": early_stopping["metric"],
            "value": initial_metrics["model"]["balanced_loss"],
        },
    }
    torch.save(initial_state, checkpoint_dir / "best.pt")
    torch.save(initial_state, checkpoint_dir / "latest.pt")
    history: list[dict[str, Any]] = [
        {
            "coverage_cycle": 0,
            "optimizer_updates": 0,
            "validation": {
                "loss": initial_metrics["model"]["balanced_loss"],
                "raw_masked_mse": initial_metrics["model"]["all_mse"],
                "event_mse": initial_metrics["model"]["event_mse"],
                "background_mse": initial_metrics["model"]["background_mse"],
            },
            "simulation_validation": initial_metrics,
        }
    ]
    best_validation_loss = float(initial_metrics["model"]["balanced_loss"])
    best_cycle = 0
    epochs_without_improvement = 0
    early_stopping_triggered = False
    stop_reason = "maximum_coverage_cycles"
    optimizer_updates = 0
    max_updates = profile["max_optimizer_updates"]
    coverage_cycles = int(profile["coverage_cycles"])
    for coverage_cycle in range(1, coverage_cycles + 1):
        model.train()
        component_sums: dict[str, float] = {}
        batches = 0
        for batch in loaders["train"]:
            signal = batch["signal"].to(device)
            target_mask = batch["target_mask"].to(device)
            event_target = batch["event_target_mask"].to(device)
            background_target = batch["background_target_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=scaler.is_enabled(),
            ):
                prediction = model(signal, time_mask=target_mask)
                loss, components = balanced_reconstruction_loss(
                    prediction,
                    signal,
                    event_target,
                    background_target,
                    event_weight=event_weight,
                    background_weight=background_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(config["training"]["grad_clip_norm"]),
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer_updates += 1
            batches += 1
            for name, value in components.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(
                    value.detach().cpu()
                )
            if max_updates is not None and optimizer_updates >= int(max_updates):
                break
        validation_metrics, _ = evaluate_reconstruction(
            model,
            loaders["simulation_validation"],
            device,
            max_examples=0,
            event_weight=event_weight,
            background_weight=background_weight,
        )
        validation_loss = float(
            validation_metrics["model"]["balanced_loss"]
        )
        improved = validation_improved(
            validation_loss,
            best_validation_loss,
            min_delta=min_delta,
        )
        if improved:
            best_validation_loss = validation_loss
            best_cycle = coverage_cycle
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        record = {
            "coverage_cycle": coverage_cycle,
            "optimizer_updates": optimizer_updates,
            "train": {
                name: value / max(batches, 1)
                for name, value in component_sums.items()
            },
            "validation": {
                "loss": validation_loss,
                "raw_masked_mse": validation_metrics["model"]["all_mse"],
                "event_mse": validation_metrics["model"]["event_mse"],
                "background_mse": validation_metrics["model"][
                    "background_mse"
                ],
            },
            "simulation_validation": validation_metrics,
            "checkpoint_selection": {
                "metric": early_stopping["metric"],
                "improved": improved,
                "best_coverage_cycle": best_cycle,
                "best_value": best_validation_loss,
                "epochs_without_improvement": epochs_without_improvement,
            },
        }
        history.append(record)
        state = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "coverage_cycle": coverage_cycle,
            "optimizer_updates": optimizer_updates,
            "validation": validation_metrics,
            "selection_metric": {
                "name": early_stopping["metric"],
                "value": validation_loss,
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
        torch.save(state, checkpoint_dir / "latest.pt")
        if improved:
            torch.save(state, checkpoint_dir / "best.pt")
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(record, sort_keys=True))
        if max_updates is not None and optimizer_updates >= int(max_updates):
            stop_reason = "maximum_optimizer_updates"
            break
        if epochs_without_improvement >= patience:
            early_stopping_triggered = True
            stop_reason = "early_stopping"
            break

    best = torch.load(
        checkpoint_dir / "best.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best["model_state_dict"])
    simulation_metrics, simulation_examples = evaluate_reconstruction(
        model,
        loaders["simulation_validation"],
        device,
        max_examples=int(profile["max_reconstruction_examples"]),
        event_weight=event_weight,
        background_weight=background_weight,
    )
    real_metrics, real_examples = evaluate_reconstruction(
        model,
        loaders["real_validation"],
        device,
        max_examples=int(profile["max_reconstruction_examples"]),
        event_weight=event_weight,
        background_weight=background_weight,
    )
    embedding_metrics = {
        "simulation_validation": evaluate_embedding_health(
            model,
            simulation_validation,
            device,
            batch_size=batch_size,
            paired_views=True,
            seed=seed,
        ),
        "real_validation": evaluate_embedding_health(
            model,
            real_validation,
            device,
            batch_size=batch_size,
            paired_views=False,
            seed=seed,
        ),
    }
    result = {
        "protocol": config["study"]["protocol"],
        "profile": profile_name,
        "seed": seed,
        "training_stage": "synthetic_only",
        "simulation_dataset": config["study"]["simulation_dataset"],
        "real_dataset": config["study"]["real_dataset"],
        "real_source_group": config["data"]["real_source_group"],
        "sealed_splits_used": [],
        "best_coverage_cycle": best_cycle,
        "completed_coverage_cycles": int(history[-1]["coverage_cycle"]),
        "optimizer_updates": optimizer_updates,
        "training_control": {
            "maximum_coverage_cycles": coverage_cycles,
            "selection_metric": early_stopping["metric"],
            "early_stopping_patience": patience,
            "early_stopping_min_delta": min_delta,
            "early_stopping_triggered": early_stopping_triggered,
            "stop_reason": stop_reason,
        },
        "counts": {
            "simulation_train": train_dataset.summary(),
            "simulation_validation": simulation_validation.summary(),
            "real_validation": real_validation.summary(),
        },
        "initial_simulation_validation_model_mse": float(
            initial_metrics["model"]["all_mse"]
        ),
        "best_simulation_validation_balanced_loss": best_validation_loss,
        "best_simulation_validation_model_mse": float(
            simulation_metrics["model"]["all_mse"]
        ),
        "simulation_validation": simulation_metrics,
        "real_validation": real_metrics,
        "embeddings": embedding_metrics,
    }
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
