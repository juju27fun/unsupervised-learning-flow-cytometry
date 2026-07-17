from __future__ import annotations

from typing import Any

import torch

from .followup_objectives import vicreg_pair_loss
from .local_spectral_target import (
    LocalSpectralTargetConfig,
    local_spectral_target,
    masked_local_spectral_mse,
)
from .study_model import YeastStudyModel
from .study_training import build_mask_batch


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
