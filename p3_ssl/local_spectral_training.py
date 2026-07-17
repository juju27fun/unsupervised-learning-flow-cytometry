from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .followup_objectives import vicreg_pair_loss
from .local_spectral_target import (
    LocalSpectralTargetConfig,
    local_spectral_frame_regions,
    local_spectral_target,
    masked_local_spectral_mse,
)
from .study_data import RealEventDataset, validate_real_event_dataset_contract
from .study_model import YeastStudyModel
from .study_training import (
    build_mask_batch,
    evaluate_embedding_health,
    interpolation_baseline,
    model_config_from_study,
    seed_everything,
)


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state.items():
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(str(tuple(values.shape)).encode("ascii"))
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def local_spectral_ssl_loss(
    model: YeastStudyModel,
    signals: torch.Tensor,
    event_masks: torch.Tensor,
    effective_config: dict[str, Any],
    *,
    seed: int,
    target_config: LocalSpectralTargetConfig,
    vicreg_weight: float,
    vicreg_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        target = local_spectral_target(signals, target_config)
    _, first_token_mask, _ = build_mask_batch(signals, event_masks, effective_config, seed)
    first_token_mask = first_token_mask.to(signals.device)
    first = model.forward_local_spectral(signals, first_token_mask)
    spectral = masked_local_spectral_mse(
        first["local_spectral_prediction"], target, first_token_mask, target_config
    )
    _, second_token_mask, _ = build_mask_batch(
        signals,
        event_masks,
        effective_config,
        seed + int(vicreg_config["second_view_seed_offset"]),
    )
    second = model.forward_local_spectral(signals, second_token_mask.to(signals.device))
    vicreg, terms = vicreg_pair_loss(
        torch.stack((first["embedding"], second["embedding"]), dim=1),
        invariance_weight=float(vicreg_config["invariance_weight"]),
        variance_weight=float(vicreg_config["variance_weight"]),
        covariance_weight=float(vicreg_config["covariance_weight"]),
        variance_floor=float(vicreg_config["variance_floor"]),
        epsilon=float(vicreg_config["epsilon"]),
    )
    total = spectral + vicreg_weight * vicreg
    return total, {
        "loss": float(total.detach().cpu()),
        "local_spectral_prediction": float(spectral.detach().cpu()),
        "vicreg": float(vicreg.detach().cpu()),
        "vicreg_invariance": float(terms["invariance"].detach().cpu()),
        "vicreg_variance": float(terms["variance"].detach().cpu()),
        "vicreg_covariance": float(terms["covariance"].detach().cpu()),
    }


def run_local_spectral_fixed_overfit(
    model: YeastStudyModel,
    signals: torch.Tensor,
    event_masks: torch.Tensor,
    train_constant: torch.Tensor,
    effective_config: dict[str, Any],
    *,
    seed: int,
    target_config: LocalSpectralTargetConfig,
    steps: int,
    learning_rate: float,
    grad_clip_norm: float,
    log_every: int,
    device: torch.device,
) -> dict[str, Any]:
    if steps <= 0 or learning_rate <= 0.0 or grad_clip_norm <= 0.0 or log_every <= 0:
        raise ValueError("S1 overfit optimization values must be positive")
    torch.manual_seed(seed)
    signals = signals.to(device)
    event_masks = event_masks.to(device)
    train_constant = train_constant.to(device)
    with torch.no_grad():
        target = local_spectral_target(signals, target_config)
    _, token_mask, _ = build_mask_batch(signals, event_masks, effective_config, seed)
    token_mask = token_mask.to(device)
    zero_prediction = torch.zeros(
        len(signals),
        effective_config["model"]["max_tokens"],
        target_config.feature_count,
        dtype=signals.dtype,
        device=device,
    )
    constant_prediction = zero_prediction.clone()
    constant_prediction[
        :, target_config.first_valid_token : target_config.stop_valid_token
    ] = train_constant
    zero_loss = float(masked_local_spectral_mse(zero_prediction, target, token_mask, target_config))
    constant_loss = float(
        masked_local_spectral_mse(constant_prediction, target, token_mask, target_config)
    )
    baseline_loss = min(zero_loss, constant_loss)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)

    def evaluate() -> tuple[float, torch.Tensor]:
        model.eval()
        with torch.no_grad():
            output = model.forward_local_spectral(signals, token_mask)[
                "local_spectral_prediction"
            ]
            loss = float(masked_local_spectral_mse(output, target, token_mask, target_config))
        return loss, output

    initial_loss, _ = evaluate()
    history = [{"step": 0, "masked_feature_mse": initial_loss}]
    gradient_norms = []
    first_encoder_gradient_norm = None
    first_head_gradient_norm = None
    for step in range(1, steps + 1):
        model.train()
        output = model.forward_local_spectral(signals, token_mask)[
            "local_spectral_prediction"
        ]
        loss = masked_local_spectral_mse(output, target, token_mask, target_config)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 1:
            encoder_gradients = [
                parameter.grad.detach().square().sum()
                for parameter in model.reconstructor.parameters()
                if parameter.grad is not None
            ]
            head_gradients = [
                parameter.grad.detach().square().sum()
                for parameter in model.local_spectral_head.parameters()
                if parameter.grad is not None
            ]
            first_encoder_gradient_norm = float(torch.sqrt(torch.stack(encoder_gradients).sum()))
            first_head_gradient_norm = float(torch.sqrt(torch.stack(head_gradients).sum()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite S1 overfit gradient")
        gradient_norms.append(float(gradient_norm.detach().cpu()))
        optimizer.step()
        if step % log_every == 0 or step == steps:
            current, _ = evaluate()
            history.append({"step": step, "masked_feature_mse": current})

    final_loss, final_prediction = evaluate()
    valid_prediction = final_prediction[
        :, target_config.first_valid_token : target_config.stop_valid_token
    ]
    valid_mask = token_mask[
        :, target_config.first_valid_token : target_config.stop_valid_token
    ].unsqueeze(-1).expand_as(target)
    selected_prediction = valid_prediction[valid_mask]
    selected_target = target[valid_mask]
    output_rms = float(torch.sqrt(selected_prediction.square().mean()))
    target_rms = float(torch.sqrt(selected_target.square().mean()))
    relative_improvement = (
        (baseline_loss - final_loss) / baseline_loss if baseline_loss > 0.0 else None
    )
    return {
        "n_signals": len(signals),
        "steps": steps,
        "learning_rate": learning_rate,
        "zero_masked_feature_mse": zero_loss,
        "train_constant_masked_feature_mse": constant_loss,
        "strongest_baseline_masked_feature_mse": baseline_loss,
        "initial_masked_feature_mse": initial_loss,
        "final_masked_feature_mse": final_loss,
        "relative_improvement_vs_strongest_baseline": relative_improvement,
        "model_output_rms_on_mask": output_rms,
        "target_rms_on_mask": target_rms,
        "output_rms_fraction_of_target": output_rms / target_rms,
        "first_encoder_gradient_norm": first_encoder_gradient_norm,
        "first_head_gradient_norm": first_head_gradient_norm,
        "first_total_gradient_norm": gradient_norms[0],
        "last_total_gradient_norm": gradient_norms[-1],
        "history": history,
        "gates": {
            "finite_gradients": all(torch.isfinite(torch.tensor(gradient_norms)).tolist()),
            "nonzero_encoder_gradient": bool(first_encoder_gradient_norm and first_encoder_gradient_norm > 0),
            "nonzero_head_gradient": bool(first_head_gradient_norm and first_head_gradient_norm > 0),
            "improves_strongest_baseline_by_0p80": (
                relative_improvement is not None and relative_improvement >= 0.80
            ),
            "nontrivial_amplitude_0p10": output_rms / target_rms >= 0.10,
        },
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    names = sorted({name for row in rows for name in row})
    return {
        name: float(np.mean([row[name] for row in rows if name in row]))
        for name in names
    }


def _selected_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor,
) -> dict[str, float | int]:
    expanded = selected.unsqueeze(-1).expand_as(target)
    count = int(expanded.sum())
    if count == 0:
        return {"squared_error_sum": 0.0, "prediction_square_sum": 0.0, "target_square_sum": 0.0, "count": 0}
    return {
        "squared_error_sum": float(((prediction - target).square() * expanded).sum()),
        "prediction_square_sum": float((prediction.square() * expanded).sum()),
        "target_square_sum": float((target.square() * expanded).sum()),
        "count": count,
    }


def _merge_sums(destination: dict[str, float | int], source: dict[str, float | int]) -> None:
    for name, value in source.items():
        destination[name] = destination.get(name, 0) + value


@torch.no_grad()
def compute_train_local_spectral_constant(
    loader: DataLoader,
    target_config: LocalSpectralTargetConfig,
    device: torch.device,
) -> torch.Tensor:
    total = torch.zeros(
        target_config.valid_token_count,
        target_config.feature_count,
        device=device,
    )
    count = 0
    for batch in loader:
        target = local_spectral_target(batch["signal"].to(device), target_config)
        total += target.sum(dim=0)
        count += len(target)
    if count == 0:
        raise ValueError("Cannot compute an S1 constant from an empty loader")
    return total / count


@torch.no_grad()
def evaluate_local_spectral_controls(
    model: YeastStudyModel,
    loader: DataLoader,
    train_constant: torch.Tensor,
    effective_config: dict[str, Any],
    *,
    seed: int,
    target_config: LocalSpectralTargetConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    names = ("model", "zero", "train_constant", "feature_of_interpolation")
    regions = ("all", "event", "background", "boundary")
    totals = {
        name: {region: {} for region in regions}
        for name in names
    }
    for batch_index, batch in enumerate(loader):
        signals = batch["signal"].to(device)
        events = batch["event_mask"].to(device)
        target = local_spectral_target(signals, target_config)
        _, token_mask, hidden_mask = build_mask_batch(
            signals, events, effective_config, seed + batch_index * 100_003
        )
        token_mask = token_mask.to(device)
        valid_mask = token_mask[
            :, target_config.first_valid_token : target_config.stop_valid_token
        ]
        output = model.forward_local_spectral(signals, token_mask)[
            "local_spectral_prediction"
        ][:, target_config.first_valid_token : target_config.stop_valid_token]
        interpolation = interpolation_baseline(signals, hidden_mask)
        predictions = {
            "model": output,
            "zero": torch.zeros_like(target),
            "train_constant": train_constant.unsqueeze(0).expand_as(target),
            "feature_of_interpolation": local_spectral_target(interpolation, target_config),
        }
        frame_regions = local_spectral_frame_regions(events, target_config)
        selections = {"all": valid_mask, **frame_regions}
        for name, prediction in predictions.items():
            for region, region_mask in selections.items():
                selected = valid_mask if region == "all" else valid_mask & region_mask
                _merge_sums(
                    totals[name][region], _selected_sums(prediction, target, selected)
                )
    result: dict[str, Any] = {"regions": {}}
    for name in names:
        result["regions"][name] = {}
        for region in regions:
            values = totals[name][region]
            count = int(values.get("count", 0))
            result["regions"][name][region] = {
                "mse": float(values["squared_error_sum"] / count) if count else None,
                "prediction_rms": (
                    float(np.sqrt(values["prediction_square_sum"] / count)) if count else None
                ),
                "target_rms": (
                    float(np.sqrt(values["target_square_sum"] / count)) if count else None
                ),
                "feature_count": count,
            }
    all_regions = result["regions"]
    model_all = all_regions["model"]["all"]
    interpolation_all = all_regions["feature_of_interpolation"]["all"]
    result.update(
        {
            "model_masked_feature_mse": model_all["mse"],
            "zero_masked_feature_mse": all_regions["zero"]["all"]["mse"],
            "train_constant_masked_feature_mse": all_regions["train_constant"]["all"]["mse"],
            "feature_of_interpolation_masked_feature_mse": interpolation_all["mse"],
            "model_output_rms_fraction_of_target": (
                model_all["prediction_rms"] / model_all["target_rms"]
            ),
        }
    )
    return result


def train_local_spectral_s1(
    *,
    effective_config: dict[str, Any],
    study_config: dict[str, Any],
    real_root: Path,
    output_dir: Path,
    profile: str,
    device: torch.device,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    contract = validate_real_event_dataset_contract(real_root)
    if not contract["valid"]:
        raise ValueError(f"Real dataset contract failed: {contract['errors']}")
    training = effective_config["training"]
    profile_config = training["profiles"][profile]
    seed = int(study_config["training"]["seed"])
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    target_config = LocalSpectralTargetConfig(
        input_length=int(study_config["target"]["input_length"]),
        sampling_frequency_hz=float(study_config["target"]["sampling_frequency_hz"]),
        patch_size=int(study_config["target"]["patch_size"]),
        window_samples=int(study_config["target"]["window_samples"]),
        first_frequency_bin=int(study_config["target"]["first_frequency_bin"]),
        stop_frequency_bin=int(study_config["target"]["stop_frequency_bin"]),
        first_valid_token=int(study_config["target"]["first_valid_token"]),
        stop_valid_token=int(study_config["target"]["stop_valid_token"]),
    )
    base_model_config = model_config_from_study(effective_config)
    control_model = YeastStudyModel(base_model_config)
    control_model_sha256 = _state_sha256(control_model.state_dict())
    control_encoder_sha256 = _state_sha256(control_model.reconstructor.encoder_state_dict())
    seed_everything(seed)
    model = YeastStudyModel(
        replace(base_model_config, local_spectral_features=target_config.feature_count)
    ).to(device)
    treatment_encoder_sha256 = _state_sha256(model.reconstructor.encoder_state_dict())
    if treatment_encoder_sha256 != control_encoder_sha256:
        raise RuntimeError("S1 encoder initialization differs from the matched C1 control")

    maximum = profile_config.get("max_real_events")
    train_data = RealEventDataset(
        real_root, effective_config["data"]["real_train_split"], max_events=maximum
    )
    validation_data = RealEventDataset(
        real_root, effective_config["data"]["real_validation_split"], max_events=maximum
    )
    batch_size = int(profile_config["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(profile_config["num_workers"]),
        generator=generator,
        pin_memory=device.type == "cuda",
        drop_last=bool(study_config["training"]["drop_last_training_batch"]),
    )
    train_evaluation_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    train_constant = compute_train_local_spectral_constant(
        train_evaluation_loader, target_config, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    vicreg_weight = float(study_config["training"]["vicreg_global_weight"])
    vicreg_config = dict(study_config["training"]["vicreg"])
    history = []
    for epoch in range(int(profile_config["epochs"])):
        model.train()
        rows = []
        for batch_index, batch in enumerate(train_loader):
            signals = batch["signal"].to(device, non_blocking=True)
            events = batch["event_mask"].to(device, non_blocking=True)
            loss, metrics = local_spectral_ssl_loss(
                model,
                signals,
                events,
                effective_config,
                seed=seed + epoch * 1_000_003 + batch_index,
                target_config=target_config,
                vicreg_weight=vicreg_weight,
                vicreg_config=vicreg_config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["grad_clip_norm"])
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite S1 training gradient")
            optimizer.step()
            metrics["gradient_norm"] = float(gradient_norm.detach().cpu())
            rows.append(metrics)
        history.append({"epoch": epoch + 1, **_mean_metrics(rows)})

    controls = evaluate_local_spectral_controls(
        model,
        validation_loader,
        train_constant,
        effective_config,
        seed=seed + 9_000_000,
        target_config=target_config,
        device=device,
    )
    health = evaluate_embedding_health(
        model, validation_loader, simulation=False, device=device
    )
    held_out_gates = study_config["gates"]["held_out"]
    model_mse = float(controls["model_masked_feature_mse"])
    gates = {
        "beats_zero": model_mse < float(controls["zero_masked_feature_mse"]),
        "beats_train_constant": model_mse
        < float(controls["train_constant_masked_feature_mse"]),
        "beats_feature_of_interpolation": model_mse
        < float(controls["feature_of_interpolation_masked_feature_mse"]),
        "nontrivial_output_amplitude": float(
            controls["model_output_rms_fraction_of_target"]
        )
        >= float(held_out_gates["output_rms_fraction_of_target_min"]),
        "effective_rank": float(health["effective_rank"])
        >= float(held_out_gates["effective_rank_min"]),
        "mean_pairwise_cosine": float(health["mean_off_diagonal_cosine_similarity"])
        <= float(held_out_gates["mean_pairwise_cosine_max"]),
    }
    scientific_decision_allowed = profile == "full"
    decision = (
        study_config["decision"]["success_action"]
        if all(gates.values()) and scientific_decision_allowed
        else study_config["decision"]["failure_action"]
        if scientific_decision_allowed
        else "smoke_only_no_scientific_decision"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "cell": "S1",
            "protocol": study_config["study"]["protocol"],
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "profile": profile,
            "seed": seed,
            "target": dict(study_config["target"]),
            "masking": dict(effective_config["masking"]),
            "vicreg_weight": vicreg_weight,
            "vicreg_config": vicreg_config,
            "control_model_initial_sha256": control_model_sha256,
            "control_encoder_initial_sha256": control_encoder_sha256,
            "treatment_encoder_initial_sha256": treatment_encoder_sha256,
        },
        checkpoint_path,
    )
    result = {
        "cell": "S1",
        "profile": profile,
        "seed": seed,
        "device": str(device),
        "target": dict(study_config["target"]),
        "masking": dict(effective_config["masking"]),
        "vicreg_weight": vicreg_weight,
        "vicreg_config": vicreg_config,
        "contract": contract,
        "training_contract": {
            "model_initialization_seed": seed,
            "control_model_initial_sha256": control_model_sha256,
            "control_encoder_initial_sha256": control_encoder_sha256,
            "treatment_encoder_initial_sha256": treatment_encoder_sha256,
            "encoder_initialization_matches_c1": True,
            "data_loader_seed": seed,
            "drop_last_training_batch": bool(
                study_config["training"]["drop_last_training_batch"]
            ),
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
        "validation_local_spectral_controls": controls,
        "validation_embedding_health": health,
        "gates": gates,
        "decision": decision,
        "scientific_decision_allowed": scientific_decision_allowed,
        "checkpoint": checkpoint_path.name,
        "sealed_splits_used": [],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
