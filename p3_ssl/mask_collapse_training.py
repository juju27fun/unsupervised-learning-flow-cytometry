from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import validate_study_config
from .followup_objectives import vicreg_pair_loss
from .losses import masked_mse
from .study_data import RealEventDataset, validate_real_event_dataset_contract
from .study_model import YeastStudyModel
from .study_training import (
    build_mask_batch,
    evaluate_embedding_health,
    evaluate_reconstruction_controls,
    model_config_from_study,
    seed_everything,
)


VALID_MASK_COLLAPSE_CELLS = frozenset({"C0", "C1"})


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(str(tuple(values.shape)).encode("ascii"))
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def mask_collapse_loss(
    model: YeastStudyModel,
    signals: torch.Tensor,
    event_masks: torch.Tensor,
    config: dict[str, Any],
    *,
    cell: str,
    seed: int,
    vicreg_weight: float,
    vicreg_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    if cell not in VALID_MASK_COLLAPSE_CELLS:
        raise ValueError(f"Unknown mask-collapse cell: {cell}")
    target_mask, first_token_mask, _ = build_mask_batch(signals, event_masks, config, seed)
    first = model(signals, first_token_mask.to(signals.device))
    time = masked_mse(first["reconstruction"], signals, target_mask.to(signals.device))
    total = time
    metrics = {"time_reconstruction": float(time.detach().cpu()), "vicreg": 0.0}
    if cell == "C1":
        _, second_token_mask, _ = build_mask_batch(
            signals, event_masks, config, seed + int(vicreg_config["second_view_seed_offset"])
        )
        second = model(signals, second_token_mask.to(signals.device))
        vicreg, terms = vicreg_pair_loss(
            torch.stack((first["embedding"], second["embedding"]), dim=1),
            invariance_weight=float(vicreg_config["invariance_weight"]),
            variance_weight=float(vicreg_config["variance_weight"]),
            covariance_weight=float(vicreg_config["covariance_weight"]),
            variance_floor=float(vicreg_config["variance_floor"]),
            epsilon=float(vicreg_config["epsilon"]),
        )
        total = total + vicreg_weight * vicreg
        metrics.update(
            {
                "vicreg": float(vicreg.detach().cpu()),
                "vicreg_invariance": float(terms["invariance"].detach().cpu()),
                "vicreg_variance": float(terms["variance"].detach().cpu()),
                "vicreg_covariance": float(terms["covariance"].detach().cpu()),
            }
        )
    metrics["loss"] = float(total.detach().cpu())
    return total, metrics


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def train_mask_collapse_cell(
    *,
    cell: str,
    config: dict[str, Any],
    real_root: Path,
    output_dir: Path,
    profile: str,
    device: torch.device,
    vicreg_weight: float,
    vicreg_config: dict[str, Any],
    drop_last_training_batch: bool,
) -> dict[str, Any]:
    if cell not in VALID_MASK_COLLAPSE_CELLS:
        raise ValueError(f"Unknown mask-collapse cell: {cell}")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if vicreg_weight <= 0.0:
        raise ValueError("vicreg_weight must be positive")
    validate_study_config(config)
    if config["masking"].get("strategy") != "patch_aligned_isolated":
        raise ValueError("Mask-collapse training requires a patch-aligned policy")

    contract = validate_real_event_dataset_contract(real_root)
    if not contract["valid"]:
        raise ValueError(f"Real dataset contract failed: {contract['errors']}")
    training = config["training"]
    profile_config = training["profiles"][profile]
    seed = int(training["seed"])
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    model = YeastStudyModel(model_config_from_study(config)).to(device)
    initial_model_sha256 = _state_sha256(model)
    maximum = profile_config.get("max_real_events")
    train_data = RealEventDataset(real_root, config["data"]["real_train_split"], max_events=maximum)
    validation_data = RealEventDataset(
        real_root, config["data"]["real_validation_split"], max_events=maximum
    )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(profile_config["batch_size"])
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(profile_config["num_workers"]),
        generator=generator,
        pin_memory=device.type == "cuda",
        drop_last=drop_last_training_batch,
    )
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history = []
    for epoch in range(int(profile_config["epochs"])):
        model.train()
        rows = []
        for batch_index, batch in enumerate(train_loader):
            signals = batch["signal"].to(device, non_blocking=True)
            events = batch["event_mask"].to(device, non_blocking=True)
            loss, metrics = mask_collapse_loss(
                model,
                signals,
                events,
                config,
                cell=cell,
                seed=seed + epoch * 1_000_003 + batch_index,
                vicreg_weight=vicreg_weight,
                vicreg_config=vicreg_config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["grad_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite mask-collapse gradient")
            optimizer.step()
            metrics["gradient_norm"] = float(gradient_norm.detach().cpu())
            rows.append(metrics)
        history.append({"epoch": epoch + 1, **_mean_metrics(rows)})

    controls = evaluate_reconstruction_controls(
        model,
        validation_loader,
        config,
        seed=seed + 9_000_000,
        simulation=False,
        device=device,
    )
    health = evaluate_embedding_health(
        model, validation_loader, simulation=False, device=device
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
            "vicreg_weight": vicreg_weight if cell == "C1" else 0.0,
            "vicreg_config": dict(vicreg_config),
            "initial_model_sha256": initial_model_sha256,
        },
        checkpoint_path,
    )
    result = {
        "cell": cell,
        "profile": profile,
        "seed": seed,
        "device": str(device),
        "masking": dict(config["masking"]),
        "vicreg_weight": vicreg_weight if cell == "C1" else 0.0,
        "vicreg_config": dict(vicreg_config),
        "contract": contract,
        "training_contract": {
            "model_initialization_seed": seed,
            "initial_model_sha256": initial_model_sha256,
            "data_loader_seed": seed,
            "drop_last_training_batch": drop_last_training_batch,
            "batch_size": batch_size,
            "steps_per_epoch": len(train_loader),
            "examples_per_epoch": len(train_loader) * batch_size,
            "discarded_examples_per_epoch": len(train_data) - len(train_loader) * batch_size,
            "first_view_mask_seed_schedule": "seed + epoch*1000003 + batch_index",
            "second_view_seed_offset": int(vicreg_config["second_view_seed_offset"]),
        },
        "n_real_train": len(train_data),
        "n_real_validation": len(validation_data),
        "history": history,
        "validation_reconstruction_controls": {"real": controls},
        "validation_embedding_health": {"real": health},
        "checkpoint": checkpoint_path.name,
        "sealed_splits_used": [],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
