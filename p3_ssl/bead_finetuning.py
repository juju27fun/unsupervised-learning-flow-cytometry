from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .bead_ssl import make_model, seed_everything
from .decimation import normalize_signal
from .models import MomentLikeReconstructor


METHODS = ("from_scratch", "P25", "CYCLIC25")
SIMULATION_FRACTIONS = (0.10, 1.0)
REAL_FRACTIONS = (0.25, 1.0)
REAL_CLASS_NAMES = ("2um", "4um", "10um")
REGRESSION_TARGET_NAMES = ("duration_ms", "doppler_khz")


@dataclass(frozen=True)
class FineTuningConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-2
    calibration_fraction: float = 0.20
    gradient_clip_norm: float = 1.0

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0 < self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be in (0, 0.5)")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")


@dataclass(frozen=True)
class FineTuningDataset:
    signals: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    sample_ids: np.ndarray
    splits: np.ndarray
    task: str
    target_names: tuple[str, ...]

    def validate(self, *, input_length: int) -> None:
        n_samples = int(self.signals.shape[0])
        if self.task not in {"simulation", "real"}:
            raise ValueError(f"Unsupported task: {self.task}")
        if self.signals.ndim != 2 or self.signals.shape[1] != input_length:
            raise ValueError(
                f"signals must have shape (N, {input_length}), got "
                f"{self.signals.shape}"
            )
        for name, values in (
            ("targets", self.targets),
            ("groups", self.groups),
            ("sample_ids", self.sample_ids),
            ("splits", self.splits),
        ):
            if len(values) != n_samples:
                raise ValueError(f"{name} length does not match signals")
        if len(set(self.sample_ids.astype(str).tolist())) != n_samples:
            raise ValueError("sample_ids must be unique")
        if not np.isfinite(self.signals).all():
            raise ValueError("signals contain non-finite values")
        if self.task == "simulation":
            if self.targets.shape != (n_samples, 2):
                raise ValueError("simulation targets must have shape (N, 2)")
            if not np.isfinite(self.targets).all():
                raise ValueError("simulation targets contain non-finite values")
        else:
            labels = np.asarray(self.targets)
            if labels.ndim != 1 or not set(np.unique(labels)).issubset({0, 1, 2}):
                raise ValueError("real targets must be integer labels 0, 1, 2")


class BeadDownstreamModel(nn.Module):
    def __init__(
        self,
        encoder: MomentLikeReconstructor,
        *,
        output_dim: int,
        pool: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.head = nn.Linear(encoder.config.d_model, output_dim)

    def forward(self, signals: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder.global_embedding(signals, pool=self.pool)
        return self.head(embedding)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fraction(task: str, fraction: float) -> float:
    if task == "simulation":
        allowed = SIMULATION_FRACTIONS
    elif task == "real":
        allowed = REAL_FRACTIONS
    else:
        raise ValueError(f"Unsupported task: {task}")
    value = float(fraction)
    if value not in allowed:
        raise ValueError(
            f"Unsupported {task} fraction {value}; allowed={allowed}"
        )
    return value


def validate_split_access(
    *,
    task: str,
    fraction: float,
    seed: int,
    fit_splits: Sequence[str],
    evaluation_split: str,
    settings: FineTuningConfig,
    confirmatory_manifest: Path | None,
    config_path: Path,
    checkpoint_paths: Mapping[str, Path],
    dataset_manifest_sha256: str,
    source_paths: Mapping[str, Path],
) -> dict[str, Any] | None:
    fit_split_names = tuple(str(split) for split in fit_splits)
    if "test" not in {*fit_split_names, str(evaluation_split)}:
        return None
    if "test" in fit_split_names:
        raise PermissionError("The sealed test split can never be a fit split")
    if str(evaluation_split) != "test":
        raise PermissionError("The sealed test split must be evaluation-only")
    if confirmatory_manifest is None:
        raise PermissionError(
            "Refusing test split without --confirmatory-manifest"
        )
    payload = json.loads(confirmatory_manifest.read_text(encoding="utf-8"))
    if payload.get("confirmatory_test_authorized") is not True:
        raise PermissionError(
            "Confirmatory manifest does not authorize the test split"
        )
    if payload.get("protocol_frozen") is not True:
        raise PermissionError("Confirmatory manifest does not freeze the protocol")
    if payload.get("test_open_count") != 0:
        raise PermissionError("Confirmatory manifest has already opened the test")
    if payload.get("sealed_split_accessed") is not False:
        raise PermissionError("Confirmatory manifest records prior sealed access")
    if payload.get("config_sha256") != sha256_file(config_path):
        raise PermissionError("Confirmatory config hash mismatch")
    design = payload.get("confirmatory_design")
    if not isinstance(design, dict):
        raise PermissionError("Confirmatory design is missing")
    task_design = design.get("tasks", {}).get(task)
    if not isinstance(task_design, dict):
        raise PermissionError(f"Confirmatory task is not frozen: {task}")
    expected_fit_splits = tuple(task_design.get("fit_splits", ()))
    if fit_split_names != expected_fit_splits:
        raise PermissionError("Confirmatory fit splits do not match the freeze")
    if str(evaluation_split) != task_design.get("evaluation_split"):
        raise PermissionError(
            "Confirmatory evaluation split does not match the freeze"
        )
    if not math.isclose(float(fraction), float(task_design.get("fraction", -1))):
        raise PermissionError("Confirmatory label fraction does not match the freeze")
    if int(seed) not in {int(value) for value in design.get("encoder_seeds", ())}:
        raise PermissionError("Confirmatory seed is not frozen")
    if list(design.get("methods", ())) != list(METHODS):
        raise PermissionError("Confirmatory methods do not match the implementation")
    if task_design.get("settings") != asdict(settings):
        raise PermissionError("Confirmatory fine-tuning settings mismatch")
    expected_checkpoints = payload.get("checkpoint_sha256", {})
    for policy in ("P25", "CYCLIC25"):
        expected = expected_checkpoints.get(policy)
        if not isinstance(expected, dict):
            raise PermissionError(
                f"Confirmatory {policy} checkpoint map is missing"
            )
        expected_hash = expected.get(str(seed))
        if sha256_file(checkpoint_paths[policy]) != expected_hash:
            raise PermissionError(
                f"Confirmatory {policy} checkpoint hash mismatch for seed {seed}"
            )
    expected_datasets = payload.get("dataset_manifest_sha256")
    if (
        not isinstance(expected_datasets, dict)
        or dataset_manifest_sha256 != expected_datasets.get(task)
    ):
        raise PermissionError("Confirmatory dataset manifest hash mismatch")
    expected_sources = payload.get("source_sha256")
    if not isinstance(expected_sources, dict):
        raise PermissionError("Confirmatory source hashes are missing")
    if set(expected_sources) != set(source_paths):
        raise PermissionError("Confirmatory source set mismatch")
    for name, path in source_paths.items():
        if sha256_file(path) != expected_sources.get(name):
            raise PermissionError(f"Confirmatory source hash mismatch: {name}")
    return payload


def load_finetuning_dataset(
    path: Path,
    *,
    task: str,
    input_length: int,
    normalization: str,
) -> FineTuningDataset:
    with np.load(path, allow_pickle=False) as payload:
        required = {"signals", "targets", "groups", "sample_ids", "split"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"Dataset NPZ is missing arrays: {sorted(missing)}")
        signals = np.asarray(payload["signals"], dtype=np.float32)
        targets = np.asarray(payload["targets"])
        groups = np.asarray(payload["groups"]).astype(str)
        sample_ids = np.asarray(payload["sample_ids"]).astype(str)
        splits = np.asarray(payload["split"]).astype(str)
        if "target_names" in payload.files:
            target_names = tuple(
                np.asarray(payload["target_names"]).astype(str).tolist()
            )
        else:
            target_names = (
                REGRESSION_TARGET_NAMES
                if task == "simulation"
                else REAL_CLASS_NAMES
            )
    if signals.ndim == 3 and signals.shape[1] == 1:
        signals = signals[:, 0]
    if signals.ndim != 2:
        raise ValueError("signals must be a two-dimensional array")
    signals = np.stack(
        [normalize_signal(signal, mode=normalization) for signal in signals]
    ).astype(np.float32)
    if task == "simulation":
        targets = np.asarray(targets, dtype=np.float32)
        if target_names != REGRESSION_TARGET_NAMES:
            raise ValueError(
                f"Simulation target_names must be {REGRESSION_TARGET_NAMES}"
            )
    elif task == "real":
        if targets.dtype.kind in {"U", "S"}:
            mapping = {name: index for index, name in enumerate(REAL_CLASS_NAMES)}
            try:
                targets = np.asarray(
                    [mapping[str(value)] for value in targets], dtype=np.int64
                )
            except KeyError as exc:
                raise ValueError(f"Unknown real class label: {exc.args[0]}") from exc
        else:
            targets = np.asarray(targets, dtype=np.int64)
        if target_names != REAL_CLASS_NAMES:
            raise ValueError(f"Real target_names must be {REAL_CLASS_NAMES}")
    else:
        raise ValueError(f"Unsupported task: {task}")
    result = FineTuningDataset(
        signals=signals,
        targets=targets,
        groups=groups,
        sample_ids=sample_ids,
        splits=splits,
        task=task,
        target_names=target_names,
    )
    result.validate(input_length=input_length)
    return result


def _partition_score(
    targets: np.ndarray,
    selected: np.ndarray,
    *,
    task: str,
    target_fraction: float,
) -> float:
    size_penalty = abs(float(selected.mean()) - target_fraction)
    if task == "real":
        overall = np.bincount(targets.astype(int), minlength=3) / len(targets)
        chosen = np.bincount(
            targets[selected].astype(int), minlength=3
        ) / max(int(selected.sum()), 1)
        return size_penalty + float(np.abs(overall - chosen).sum())
    overall_mean = targets.mean(axis=0)
    overall_scale = np.maximum(targets.std(axis=0), 1.0e-8)
    chosen_mean = targets[selected].mean(axis=0)
    return size_penalty + float(
        np.abs((chosen_mean - overall_mean) / overall_scale).mean()
    )


def group_safe_subset(
    indices: np.ndarray,
    *,
    targets: np.ndarray,
    groups: np.ndarray,
    fraction: float,
    task: str,
    seed: int,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if fraction == 1.0:
        return indices.copy()
    local_targets = targets[indices]
    local_groups = groups[indices]
    unique_groups = np.unique(local_groups)
    if unique_groups.size < 2:
        raise ValueError("Group-safe sampling requires at least two groups")
    candidates = GroupShuffleSplit(
        n_splits=64,
        train_size=fraction,
        random_state=seed,
    )
    best: tuple[float, np.ndarray] | None = None
    for selected_local, _ in candidates.split(indices, groups=local_groups):
        selected_mask = np.zeros(indices.size, dtype=bool)
        selected_mask[selected_local] = True
        if task == "real" and set(
            np.unique(local_targets[selected_mask]).tolist()
        ) != {0, 1, 2}:
            continue
        score = _partition_score(
            local_targets,
            selected_mask,
            task=task,
            target_fraction=fraction,
        )
        if best is None or score < best[0]:
            best = (score, indices[selected_mask])
    if best is None:
        raise ValueError(
            "Unable to build a group-safe subset containing every class"
        )
    return np.sort(best[1])


def group_safe_calibration_split(
    indices: np.ndarray,
    *,
    targets: np.ndarray,
    groups: np.ndarray,
    calibration_fraction: float,
    task: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    local_targets = targets[indices]
    local_groups = groups[indices]
    splitter = GroupShuffleSplit(
        n_splits=64,
        test_size=calibration_fraction,
        random_state=seed,
    )
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for fit_local, calibration_local in splitter.split(
        indices, groups=local_groups
    ):
        if task == "real":
            if set(np.unique(local_targets[fit_local]).tolist()) != {0, 1, 2}:
                continue
            if set(
                np.unique(local_targets[calibration_local]).tolist()
            ) != {0, 1, 2}:
                continue
        selected = np.zeros(indices.size, dtype=bool)
        selected[calibration_local] = True
        score = _partition_score(
            local_targets,
            selected,
            task=task,
            target_fraction=calibration_fraction,
        )
        if best is None or score < best[0]:
            best = (
                score,
                indices[fit_local],
                indices[calibration_local],
            )
    if best is None:
        raise ValueError(
            "Unable to build a group-safe internal calibration split"
        )
    fit, calibration = np.sort(best[1]), np.sort(best[2])
    if set(groups[fit]).intersection(set(groups[calibration])):
        raise AssertionError("Internal calibration group leakage")
    return fit, calibration


def validate_external_group_separation(
    data: FineTuningDataset,
    *,
    fit_splits: Sequence[str],
    evaluation_split: str,
) -> tuple[np.ndarray, np.ndarray]:
    fit_candidates = np.flatnonzero(np.isin(data.splits, list(fit_splits)))
    evaluation_indices = np.flatnonzero(data.splits == evaluation_split)
    if not fit_candidates.size or not evaluation_indices.size:
        raise ValueError("Requested fit or evaluation split is empty")
    overlap = set(data.groups[fit_candidates]).intersection(
        set(data.groups[evaluation_indices])
    )
    if overlap:
        raise ValueError(
            f"Group leakage between fit and evaluation splits: "
            f"{sorted(overlap)[:3]}"
        )
    return fit_candidates, evaluation_indices


def _validate_checkpoint(
    payload: dict[str, Any],
    *,
    policy: str,
    seed: int,
    config: dict[str, Any],
) -> None:
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError(f"{policy} checkpoint has no frozen config")
    if "model_state_dict" not in payload or "epoch" not in payload:
        raise ValueError(f"{policy} checkpoint is incomplete")
    if int(payload["epoch"]) != 20:
        raise ValueError(f"{policy} checkpoint must be frozen at epoch 20")
    expected = copy.deepcopy(config)
    expected["training"]["seed"] = int(seed)
    expected["masking"]["training_policy"] = policy
    expected["loss"]["selected_cell"] = "B0"
    for section in ("study", "data", "model", "masking", "loss", "training"):
        if checkpoint_config.get(section) != expected.get(section):
            raise ValueError(
                f"{policy} checkpoint frozen {section} config mismatch"
            )


def initialize_paired_models(
    config: dict[str, Any],
    *,
    seed: int,
    task: str,
    p25_checkpoint: Path,
    cyclic25_checkpoint: Path,
    device: torch.device,
) -> tuple[dict[str, BeadDownstreamModel], dict[str, dict[str, Any]]]:
    output_dim = 2 if task == "simulation" else 3
    checkpoint_paths = {
        "P25": p25_checkpoint,
        "CYCLIC25": cyclic25_checkpoint,
    }
    seed_everything(seed)
    scratch_encoder = make_model(config)
    encoders: dict[str, MomentLikeReconstructor] = {
        "from_scratch": scratch_encoder
    }
    metadata: dict[str, dict[str, Any]] = {
        "from_scratch": {"checkpoint": None, "epoch": 0}
    }
    for policy, path in checkpoint_paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _validate_checkpoint(
            payload, policy=policy, seed=seed, config=config
        )
        encoder = make_model(config)
        encoder.load_state_dict(payload["model_state_dict"], strict=True)
        encoders[policy] = encoder
        metadata[policy] = {
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "epoch": int(payload["epoch"]),
        }

    seed_everything(seed + 10_000)
    reference_head = nn.Linear(config["model"]["d_model"], output_dim)
    head_state = copy.deepcopy(reference_head.state_dict())
    models = {}
    for method in METHODS:
        model = BeadDownstreamModel(
            encoders[method],
            output_dim=output_dim,
            pool=str(config["model"]["embedding_pool"]),
        )
        model.head.load_state_dict(head_state)
        models[method] = model.to(device)
    return models, metadata


def _classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    labels = np.arange(3)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": {
            name: float(value)
            for name, value in zip(
                REAL_CLASS_NAMES,
                recall_score(
                    targets,
                    predictions,
                    labels=labels,
                    average=None,
                    zero_division=0,
                ),
                strict=True,
            )
        },
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=labels
        ).astype(int).tolist(),
    }


def _regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    r2_values = r2_score(
        targets, predictions, multioutput="raw_values"
    )
    mae_values = mean_absolute_error(
        targets, predictions, multioutput="raw_values"
    )
    iqr = np.maximum(
        np.quantile(targets, 0.75, axis=0)
        - np.quantile(targets, 0.25, axis=0),
        1.0e-8,
    )
    return {
        "mean_r2": float(np.mean(r2_values)),
        "r2": {
            name: float(value)
            for name, value in zip(
                REGRESSION_TARGET_NAMES, r2_values, strict=True
            )
        },
        "mae": {
            name: float(value)
            for name, value in zip(
                REGRESSION_TARGET_NAMES, mae_values, strict=True
            )
        },
        "normalized_mae_iqr": {
            name: float(value)
            for name, value in zip(
                REGRESSION_TARGET_NAMES, mae_values / iqr, strict=True
            )
        },
    }


def aggregate_regression_by_group(
    targets: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(np.asarray(groups, dtype=str))
    grouped_targets = []
    grouped_predictions = []
    for group in unique:
        indices = np.flatnonzero(np.asarray(groups, dtype=str) == group)
        selected_targets = np.asarray(targets)[indices]
        if not np.allclose(selected_targets, selected_targets[0]):
            raise ValueError(f"Regression targets differ within group {group}")
        grouped_targets.append(selected_targets[0])
        grouped_predictions.append(np.asarray(predictions)[indices].mean(axis=0))
    return np.asarray(grouped_targets), np.asarray(grouped_predictions)


def _predict(
    model: nn.Module,
    signals: np.ndarray,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(signals[indices]).unsqueeze(1)),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    values = []
    with torch.no_grad():
        for (batch,) in loader:
            values.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(values, axis=0)


def _temperature_scale(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    logits_tensor = torch.from_numpy(logits).double()
    targets_tensor = torch.from_numpy(targets.astype(np.int64))
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = nn.functional.cross_entropy(
            logits_tensor / temperature, targets_tensor
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def fine_tune_one(
    model: BeadDownstreamModel,
    data: FineTuningDataset,
    *,
    fit_indices: np.ndarray,
    calibration_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    settings: FineTuningConfig,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    settings.validate()
    seed_everything(seed)
    fit_targets = data.targets[fit_indices]
    if data.task == "simulation":
        target_mean = fit_targets.mean(axis=0)
        target_std = np.maximum(fit_targets.std(axis=0), 1.0e-8)
        training_targets = (
            (fit_targets - target_mean) / target_std
        ).astype(np.float32)
        criterion: nn.Module = nn.MSELoss()
    else:
        target_mean = np.zeros(3, dtype=np.float32)
        target_std = np.ones(3, dtype=np.float32)
        training_targets = fit_targets.astype(np.int64)
        counts = np.bincount(training_targets, minlength=3)
        if np.any(counts == 0):
            raise ValueError("Fine-tuning fit split must contain every class")
        class_weights = len(training_targets) / (3.0 * counts)
        criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(class_weights.astype(np.float32)).to(device)
        )
    x_tensor = torch.from_numpy(data.signals[fit_indices]).unsqueeze(1)
    y_tensor = torch.from_numpy(training_targets)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=settings.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    optimizer_steps = 0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for signals, targets in loader:
            signals = signals.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(signals)
            loss = criterion(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), settings.gradient_clip_norm
            )
            optimizer.step()
            optimizer_steps += 1
            total += float(loss.detach().cpu()) * len(signals)
            count += len(signals)
        calibration_raw = _predict(
            model,
            data.signals,
            calibration_indices,
            batch_size=settings.batch_size,
            device=device,
        )
        if data.task == "simulation":
            calibration_target = (
                (data.targets[calibration_indices] - target_mean) / target_std
            )
            calibration_loss = float(
                np.mean(np.square(calibration_raw - calibration_target))
            )
        else:
            calibration_logits = torch.from_numpy(calibration_raw)
            calibration_target = torch.from_numpy(
                data.targets[calibration_indices].astype(np.int64)
            )
            calibration_loss = float(
                nn.functional.cross_entropy(
                    calibration_logits, calibration_target
                )
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(count, 1),
                "internal_calibration_loss": calibration_loss,
            }
        )
        if calibration_loss < best_loss:
            best_loss = calibration_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("Fine-tuning did not produce a checkpoint")
    model.load_state_dict(best_state)
    evaluation_raw = _predict(
        model,
        data.signals,
        evaluation_indices,
        batch_size=settings.batch_size,
        device=device,
    )
    result: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_internal_calibration_loss": best_loss,
        "epochs_completed": settings.epochs,
        "optimizer_steps": optimizer_steps,
        "history": history,
        "model_state_dict": best_state,
    }
    if data.task == "simulation":
        predictions = evaluation_raw * target_std + target_mean
        grouped_targets, grouped_predictions = aggregate_regression_by_group(
            data.targets[evaluation_indices],
            predictions,
            data.groups[evaluation_indices],
        )
        result.update(
            {
                "predictions": predictions.astype(np.float32),
                "metrics": _regression_metrics(
                    grouped_targets, grouped_predictions
                ),
                "evaluation_latents": int(grouped_targets.shape[0]),
                "target_standardization": {
                    "mean": target_mean.astype(float).tolist(),
                    "std": target_std.astype(float).tolist(),
                },
            }
        )
    else:
        calibration_logits = _predict(
            model,
            data.signals,
            calibration_indices,
            batch_size=settings.batch_size,
            device=device,
        )
        temperature = _temperature_scale(
            calibration_logits, data.targets[calibration_indices]
        )
        scaled = evaluation_raw / temperature
        probabilities = torch.softmax(
            torch.from_numpy(scaled), dim=1
        ).numpy()
        predictions = probabilities.argmax(axis=1)
        result.update(
            {
                "predictions": predictions.astype(np.int64),
                "probabilities": probabilities.astype(np.float32),
                "metrics": _classification_metrics(
                    data.targets[evaluation_indices], predictions
                ),
                "temperature": temperature,
            }
        )
    return result


def _write_method_outputs(
    output_dir: Path,
    *,
    method: str,
    result: dict[str, Any],
    data: FineTuningDataset,
    evaluation_indices: np.ndarray,
    initialization: dict[str, Any],
) -> None:
    metrics_payload = {
        "method": method,
        "metrics": result["metrics"],
        "best_epoch": result["best_epoch"],
        "best_internal_calibration_loss": result[
            "best_internal_calibration_loss"
        ],
        "epochs_completed": result["epochs_completed"],
        "optimizer_steps": result["optimizer_steps"],
        "evaluation_latents": result.get("evaluation_latents"),
        "history": result["history"],
        "temperature": result.get("temperature"),
        "target_standardization": result.get("target_standardization"),
        "initialization": initialization,
    }
    (output_dir / f"{method}_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "method": method,
            "model_state_dict": result["model_state_dict"],
            "best_epoch": result["best_epoch"],
            "metrics": result["metrics"],
        },
        output_dir / f"{method}_checkpoint.pt",
    )
    with (output_dir / f"{method}_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        if data.task == "simulation":
            fields = [
                "sample_id",
                "group",
                "split",
                "target_duration_ms",
                "prediction_duration_ms",
                "target_doppler_khz",
                "prediction_doppler_khz",
            ]
        else:
            fields = [
                "sample_id",
                "group",
                "split",
                "target",
                "prediction",
                "probability_2um",
                "probability_4um",
                "probability_10um",
            ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for local_index, index in enumerate(evaluation_indices):
            if data.task == "simulation":
                row = {
                    "sample_id": data.sample_ids[index],
                    "group": data.groups[index],
                    "split": data.splits[index],
                    "target_duration_ms": float(data.targets[index, 0]),
                    "prediction_duration_ms": float(
                        result["predictions"][local_index, 0]
                    ),
                    "target_doppler_khz": float(data.targets[index, 1]),
                    "prediction_doppler_khz": float(
                        result["predictions"][local_index, 1]
                    ),
                }
            else:
                row = {
                    "sample_id": data.sample_ids[index],
                    "group": data.groups[index],
                    "split": data.splits[index],
                    "target": REAL_CLASS_NAMES[int(data.targets[index])],
                    "prediction": REAL_CLASS_NAMES[
                        int(result["predictions"][local_index])
                    ],
                    **{
                        f"probability_{name}": float(
                            result["probabilities"][local_index, class_index]
                        )
                        for class_index, name in enumerate(REAL_CLASS_NAMES)
                    },
                }
            writer.writerow(row)


def run_paired_finetuning(
    config: dict[str, Any],
    data: FineTuningDataset,
    *,
    fraction: float,
    fit_splits: Sequence[str],
    evaluation_split: str,
    seed: int,
    p25_checkpoint: Path,
    cyclic25_checkpoint: Path,
    settings: FineTuningConfig,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    fraction = validate_fraction(data.task, fraction)
    settings.validate()
    data.validate(input_length=int(config["data"]["input_length"]))
    if not fit_splits:
        raise ValueError("fit_splits cannot be empty")
    fit_candidates, evaluation_indices = validate_external_group_separation(
        data,
        fit_splits=fit_splits,
        evaluation_split=evaluation_split,
    )
    selected = group_safe_subset(
        fit_candidates,
        targets=data.targets,
        groups=data.groups,
        fraction=fraction,
        task=data.task,
        seed=seed,
    )
    fit_indices, calibration_indices = group_safe_calibration_split(
        selected,
        targets=data.targets,
        groups=data.groups,
        calibration_fraction=settings.calibration_fraction,
        task=data.task,
        seed=seed,
    )
    models, initialization = initialize_paired_models(
        config,
        seed=seed,
        task=data.task,
        p25_checkpoint=p25_checkpoint,
        cyclic25_checkpoint=cyclic25_checkpoint,
        device=device,
    )
    results: dict[str, Any] = {}
    for method in METHODS:
        method_result = fine_tune_one(
            models[method],
            data,
            fit_indices=fit_indices,
            calibration_indices=calibration_indices,
            evaluation_indices=evaluation_indices,
            settings=settings,
            seed=seed,
            device=device,
        )
        _write_method_outputs(
            output_dir,
            method=method,
            result=method_result,
            data=data,
            evaluation_indices=evaluation_indices,
            initialization=initialization[method],
        )
        results[method] = {
            key: value
            for key, value in method_result.items()
            if key
            not in {
                "model_state_dict",
                "predictions",
                "probabilities",
                "history",
            }
        }
    summary = {
        "protocol": "bead-downstream-finetuning-v1",
        "task": data.task,
        "fraction": fraction,
        "seed": seed,
        "fit_splits": list(fit_splits),
        "evaluation_split": evaluation_split,
        "settings": asdict(settings),
        "counts": {
            "fit_candidates": int(fit_candidates.size),
            "selected": int(selected.size),
            "internal_fit": int(fit_indices.size),
            "internal_calibration": int(calibration_indices.size),
            "evaluation": int(evaluation_indices.size),
            "selected_groups": int(np.unique(data.groups[selected]).size),
            "internal_fit_groups": int(
                np.unique(data.groups[fit_indices]).size
            ),
            "internal_calibration_groups": int(
                np.unique(data.groups[calibration_indices]).size
            ),
            "evaluation_groups": int(
                np.unique(data.groups[evaluation_indices]).size
            ),
        },
        "methods": results,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metric_name = "mean_r2" if data.task == "simulation" else "macro_f1"
    with (output_dir / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "task",
                "fraction",
                "seed",
                metric_name,
                "best_epoch",
                "optimizer_steps",
            ],
        )
        writer.writeheader()
        for method in METHODS:
            writer.writerow(
                {
                    "method": method,
                    "task": data.task,
                    "fraction": fraction,
                    "seed": seed,
                    metric_name: results[method]["metrics"][metric_name],
                    "best_epoch": results[method]["best_epoch"],
                    "optimizer_steps": results[method]["optimizer_steps"],
                }
            )
    return summary
