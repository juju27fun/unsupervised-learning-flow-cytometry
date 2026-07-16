from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .followup_objectives import FollowupObjectiveConfig, VALID_FOLLOWUP_CELLS, followup_ssl_objective
from .study_data import RealEventDataset, SimulatedLatentDataset
from .study_model import YeastStudyModel, paired_nuisance_consistency, physics_supervision_loss
from .study_training import build_mask_batch, model_config_from_study, seed_everything


EXPECTED_CELLS = {
    "R0": {"time_reconstruction": True, "spectral_reconstruction": False, "vicreg": False},
    "R1": {"time_reconstruction": True, "spectral_reconstruction": False, "vicreg": True},
    "R2": {"time_reconstruction": True, "spectral_reconstruction": True, "vicreg": False},
    "R3": {"time_reconstruction": True, "spectral_reconstruction": True, "vicreg": True},
}
EXPECTED_FORBIDDEN_SPLITS = {
    "in_session_test",
    "sealed_acquisition_test",
    "followup_test",
    "test",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_followup_config(config: dict[str, Any]) -> None:
    if config["study"]["source_frozen_protocol"] != "yeast-ssl-followup-v2-frozen-20260716":
        raise ValueError("Week 2 must derive from the frozen Week 1 protocol")
    if config["objectives"]["cells"] != EXPECTED_CELLS:
        raise ValueError("R0-R3 objective cells differ from the frozen protocol")
    if set(config["data"]["forbidden_training_splits"]) != EXPECTED_FORBIDDEN_SPLITS:
        raise ValueError("Final-split guard differs from the frozen protocol")
    selected = {
        config["data"]["real_train_split"],
        config["data"]["real_validation_split"],
        config["data"]["simulation_train_split"],
        config["data"]["simulation_validation_split"],
    }
    if selected & EXPECTED_FORBIDDEN_SPLITS:
        raise ValueError("A forbidden split was selected")
    if [int(seed) for seed in config["training"]["representation_seeds"]] != [42, 43, 44]:
        raise ValueError("Representation seeds must remain 42, 43, and 44")
    if config["training"]["checkpoint_selection"] != "fixed_final_epoch_no_early_stopping":
        raise ValueError("Week 2 uses a fixed final checkpoint for equal budgets")
    if int(config["objectives"]["vicreg"]["minimum_independent_latents_per_batch"]) != 2:
        raise ValueError("VICReg requires at least two independent samples")
    windows = [int(value) for value in config["objectives"]["spectral"]["windows_samples"]]
    hops = [int(value) for value in config["objectives"]["spectral"]["hops_samples"]]
    if windows != [128, 256, 512] or hops != [32, 64, 128]:
        raise ValueError("STFT resolutions differ from the frozen protocol")
    if any(window // 4 != hop for window, hop in zip(windows, hops)):
        raise ValueError("Every STFT hop must be one quarter of its window")
    if config["objectives"]["spectral"].get("center") is not False:
        raise ValueError("Week 2 STFT must not use reflection padding")
    for profile in config["training"]["profiles"].values():
        if int(profile["batch_size"]) < 2:
            raise ValueError("Every Week 2 batch must contain at least two independent samples")


def validate_followup_dataset_contracts(
    real_root: Path, simulation_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    real_contract = json.loads((real_root / "input_contract.json").read_text(encoding="utf-8"))
    simulation_summary = json.loads(
        (simulation_root / "dataset_summary.json").read_text(encoding="utf-8")
    )
    development_path = real_root / "development_events.csv"
    sealed_path = real_root / "sealed_followup_test_events.csv"
    if not development_path.is_file() or not sealed_path.is_file():
        raise ValueError("Follow-up development and sealed metadata must be physically separated")
    development_rows = _read_csv(development_path)
    development_splits = {row["development_split"] for row in development_rows}
    errors: list[str] = []
    if real_contract.get("contract_id") != config["study"]["input_contract"]:
        errors.append("unexpected real input contract")
    if int(real_contract.get("output_length", -1)) != int(config["data"]["input_length"]):
        errors.append("real output length differs from the Week 2 contract")
    if development_splits != {
        config["data"]["real_train_split"],
        config["data"]["real_validation_split"],
    }:
        errors.append(f"development metadata has unexpected splits: {sorted(development_splits)}")
    if development_splits & EXPECTED_FORBIDDEN_SPLITS:
        errors.append("development metadata contains a final split")
    real_shape = list(np.load(real_root / "signals.npy", mmap_mode="r").shape)
    simulation_shape = list(np.load(simulation_root / "signals.npy", mmap_mode="r").shape)
    if real_shape[1:] != [int(config["data"]["input_length"])]:
        errors.append(f"incompatible real signal shape: {real_shape}")
    if simulation_shape[1:] != [int(config["data"]["input_length"])]:
        errors.append(f"incompatible simulation signal shape: {simulation_shape}")
    if "4096" not in str(simulation_summary.get("input_contract", "")):
        errors.append("simulation does not declare a compatible 4096-sample contract")
    return {
        "valid": not errors,
        "errors": errors,
        "real_shape": real_shape,
        "simulation_shape": simulation_shape,
        "development_splits": sorted(development_splits),
        "sealed_metadata_opened": False,
        "real_contract": real_contract.get("contract_id"),
        "simulation_generator": simulation_summary.get("generator_id"),
    }


def objective_config_from_followup(config: dict[str, Any]) -> FollowupObjectiveConfig:
    spectral = config["objectives"]["spectral"]
    vicreg = config["objectives"]["vicreg"]
    return FollowupObjectiveConfig(
        spectral_windows=tuple(int(value) for value in spectral["windows_samples"]),
        spectral_hop_divisor=4,
        spectral_center=bool(spectral["center"]),
        spectral_epsilon=float(spectral["epsilon"]),
        time_weight=float(config["objectives"]["time_weight"]),
        spectral_weight=float(spectral["weight"]),
        vicreg_weight=float(vicreg["global_weight"]),
        invariance_weight=float(vicreg["invariance_weight"]),
        variance_weight=float(vicreg["variance_weight"]),
        covariance_weight=float(vicreg["covariance_weight"]),
        variance_floor=float(vicreg["variance_floor"]),
        variance_epsilon=float(vicreg["epsilon"]),
    )


def _float_terms(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in terms.items()}


def followup_real_loss(
    model: YeastStudyModel,
    batch: dict[str, Any],
    config: dict[str, Any],
    cell: str,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    signals = batch["signal"]
    event_masks = batch["event_mask"]
    target_mask, token_mask, _ = build_mask_batch(signals, event_masks, config, seed)
    first = model(signals, token_mask.to(signals.device))
    paired = None
    if cell in {"R1", "R3"}:
        _, second_token_mask, _ = build_mask_batch(
            signals, event_masks, config, seed + 10_000_019
        )
        second = model(signals, second_token_mask.to(signals.device))
        paired = torch.stack((first["embedding"], second["embedding"]), dim=1)
    loss, terms = followup_ssl_objective(
        cell,
        first["reconstruction"],
        signals,
        target_mask.to(signals.device),
        paired_embeddings=paired,
        config=objective_config_from_followup(config),
    )
    return loss, _float_terms(terms)


def followup_simulation_loss(
    model: YeastStudyModel,
    batch: dict[str, Any],
    config: dict[str, Any],
    cell: str,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    signals = batch["signals"]
    event_masks = batch["event_masks"]
    batch_size, views, channels, length = signals.shape
    flat_signals = signals.reshape(batch_size * views, channels, length)
    flat_events = event_masks.reshape(batch_size * views, length)
    target_mask, token_mask, _ = build_mask_batch(flat_signals, flat_events, config, seed)
    output = model(flat_signals, token_mask.to(flat_signals.device))
    paired = output["embedding"].reshape(batch_size, views, -1)
    ssl_loss, terms = followup_ssl_objective(
        cell,
        output["reconstruction"],
        flat_signals,
        target_mask.to(flat_signals.device),
        paired_embeddings=paired if cell in {"R1", "R3"} else None,
        config=objective_config_from_followup(config),
    )
    continuous_targets = batch["continuous_targets"].repeat_interleave(views, dim=0)
    continuous_valid = batch["continuous_valid"].repeat_interleave(views, dim=0)
    component_target = batch["component_target"].repeat_interleave(views, dim=0)
    continuous, component = physics_supervision_loss(
        output, continuous_targets, continuous_valid, component_target
    )
    nuisance = paired_nuisance_consistency(paired)
    weights = config["objectives"]["common_physics_supervision"]
    total = (
        ssl_loss
        + float(weights["continuous_weight"]) * continuous
        + float(weights["component_count_weight"]) * component
        + float(weights["nuisance_consistency_weight"]) * nuisance
    )
    metrics = _float_terms(terms)
    metrics.update(
        {
            "physics_continuous": float(continuous.detach().cpu()),
            "component_count": float(component.detach().cpu()),
            "nuisance_consistency": float(nuisance.detach().cpu()),
            "total_with_common_physics": float(total.detach().cpu()),
        }
    )
    return total, metrics


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    names = sorted({name for row in rows for name in row})
    return {
        name: float(np.mean([row[name] for row in rows if name in row])) for name in names
    }


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def _endless(loader: DataLoader) -> Iterable[dict[str, Any]]:
    while True:
        yield from loader


def _optimizer_step(
    model: YeastStudyModel,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    grad_clip_norm: float,
) -> float:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite Week 2 loss: {float(loss.detach().cpu())}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("Non-finite Week 2 gradient norm")
    optimizer.step()
    return float(gradient_norm.detach().cpu())


def convergence_diagnostics(
    history: list[dict[str, Any]], expected_pretraining: int, expected_adaptation: int
) -> dict[str, Any]:
    phases = {
        phase: [row for row in history if row["phase"] == phase]
        for phase in ("synthetic_pretraining", "real_adaptation")
    }
    expected = {
        "synthetic_pretraining": expected_pretraining,
        "real_adaptation": expected_adaptation,
    }
    phase_results: dict[str, Any] = {}
    for phase, rows in phases.items():
        losses = np.asarray([float(row["epoch_loss"]) for row in rows], dtype=float)
        finite = bool(losses.size and np.isfinite(losses).all())
        if losses.size >= 3:
            first = float(losses[:3].mean())
            final = float(losses[-3:].mean())
            trend = final <= first
            trend_status = "pass" if trend else "fail"
        else:
            first = final = float(losses.mean()) if losses.size else math.nan
            trend = True
            trend_status = "not_assessed_smoke"
        phase_results[phase] = {
            "completed_epochs": len(rows),
            "expected_epochs": expected[phase],
            "all_losses_finite": finite,
            "first_three_mean": first,
            "final_three_mean": final,
            "loss_trend": trend_status,
            "pass": len(rows) == expected[phase] and finite and trend,
        }
    return {
        "definition": "all fixed epochs complete, finite, and final-three mean loss does not exceed first-three mean in each full phase",
        "phases": phase_results,
        "converged": all(result["pass"] for result in phase_results.values()),
    }


def train_followup_cell(
    *,
    cell: str,
    seed: int,
    config: dict[str, Any],
    real_root: Path,
    simulation_root: Path,
    output_dir: Path,
    profile: str,
    device: torch.device,
) -> dict[str, Any]:
    validate_followup_config(config)
    if cell not in VALID_FOLLOWUP_CELLS:
        raise ValueError(f"Unknown Week 2 cell: {cell}")
    if seed not in {int(value) for value in config["training"]["representation_seeds"]}:
        raise ValueError(f"Seed {seed} is not preregistered")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    contract = validate_followup_dataset_contracts(real_root, simulation_root, config)
    if not contract["valid"]:
        raise ValueError(f"Dataset contract failed: {contract['errors']}")
    profile_config = config["training"]["profiles"][profile]
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    model = YeastStudyModel(model_config_from_study(config)).to(device)
    batch_size = int(profile_config["batch_size"])
    workers = int(profile_config["num_workers"])
    real_train = RealEventDataset(
        real_root,
        config["data"]["real_train_split"],
        max_events=profile_config["max_real_events"],
    )
    simulation_train = SimulatedLatentDataset(
        simulation_root,
        config["data"]["simulation_train_split"],
        max_latents=profile_config["max_simulation_latents"],
    )
    real_generator = torch.Generator().manual_seed(seed + 101)
    simulation_generator = torch.Generator().manual_seed(seed + 202)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    real_loader = DataLoader(real_train, generator=real_generator, **loader_kwargs)
    simulation_loader = DataLoader(
        simulation_train, generator=simulation_generator, **loader_kwargs
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    grad_clip = float(config["training"]["grad_clip_norm"])
    pretraining_epochs = int(profile_config["synthetic_pretraining_epochs"])
    adaptation_epochs = int(profile_config["real_adaptation_epochs"])
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_optimizer_steps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(pretraining_epochs):
        model.train()
        rows: list[dict[str, float]] = []
        for batch_index, raw_batch in enumerate(simulation_loader):
            batch = _move_batch(raw_batch, device)
            loss, metrics = followup_simulation_loss(
                model,
                batch,
                config,
                cell,
                seed + epoch * 1_000_003 + batch_index,
            )
            metrics["gradient_norm"] = _optimizer_step(model, optimizer, loss, grad_clip)
            metrics["epoch_loss"] = float(loss.detach().cpu())
            rows.append(metrics)
            total_optimizer_steps += 1
        means = _mean_metrics(rows)
        history.append(
            {
                "phase": "synthetic_pretraining",
                "epoch": epoch + 1,
                "epoch_loss": means["epoch_loss"],
                "optimizer_steps": len(rows),
                **{name: value for name, value in means.items() if name != "epoch_loss"},
            }
        )

    replay = config["training"]["adaptation_replay"]
    for epoch in range(adaptation_epochs):
        model.train()
        rows = []
        real_iterator = _endless(real_loader)
        simulation_iterator = _endless(simulation_loader)
        progress = epoch / max(adaptation_epochs - 1, 1)
        synthetic_weight = float(replay["synthetic_weight_start"]) + progress * (
            float(replay["synthetic_weight_end"])
            - float(replay["synthetic_weight_start"])
        )
        for batch_index in range(max(len(real_loader), len(simulation_loader))):
            real_batch = _move_batch(next(real_iterator), device)
            simulation_batch = _move_batch(next(simulation_iterator), device)
            real_loss, real_metrics = followup_real_loss(
                model,
                real_batch,
                config,
                cell,
                seed + 20_000_000 + epoch * 1_000_003 + batch_index,
            )
            simulation_loss, simulation_metrics = followup_simulation_loss(
                model,
                simulation_batch,
                config,
                cell,
                seed + 30_000_000 + epoch * 1_000_003 + batch_index,
            )
            loss = (1.0 - synthetic_weight) * real_loss + synthetic_weight * simulation_loss
            metrics = {
                "epoch_loss": float(loss.detach().cpu()),
                "real_total": float(real_loss.detach().cpu()),
                "simulation_total": float(simulation_loss.detach().cpu()),
                "synthetic_replay_weight": synthetic_weight,
                **{f"real_{name}": value for name, value in real_metrics.items()},
                **{f"simulation_{name}": value for name, value in simulation_metrics.items()},
            }
            metrics["gradient_norm"] = _optimizer_step(model, optimizer, loss, grad_clip)
            rows.append(metrics)
            total_optimizer_steps += 1
        means = _mean_metrics(rows)
        history.append(
            {
                "phase": "real_adaptation",
                "epoch": epoch + 1,
                "epoch_loss": means["epoch_loss"],
                "optimizer_steps": len(rows),
                **{name: value for name, value in means.items() if name != "epoch_loss"},
            }
        )

    runtime_seconds = time.perf_counter() - started
    convergence = convergence_diagnostics(history, pretraining_epochs, adaptation_epochs)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "cell": cell,
            "protocol": config["study"]["protocol"],
            "source_frozen_protocol": config["study"]["source_frozen_protocol"],
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "profile": profile,
            "seed": seed,
            "checkpoint_policy": config["training"]["checkpoint_selection"],
            "sealed_splits_used": [],
        },
        checkpoint_path,
    )
    result = {
        "cell": cell,
        "seed": seed,
        "profile": profile,
        "device": str(device),
        "contract": contract,
        "n_real_train": len(real_train),
        "n_simulation_train_latents": len(simulation_train),
        "history": history,
        "convergence": convergence,
        "runtime": {
            "wall_seconds": runtime_seconds,
            "optimizer_steps": total_optimizer_steps,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        },
        "checkpoint": checkpoint_path.name,
        "sealed_splits_used": [],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
