from __future__ import annotations

from typing import Any

import torch

from .losses import masked_mse
from .study_training import build_mask_batch


def run_fixed_mask_overfit(
    model: torch.nn.Module,
    signals: torch.Tensor,
    event_masks: torch.Tensor,
    config: dict[str, Any],
    *,
    seed: int,
    steps: int,
    learning_rate: float,
    device: torch.device,
    log_every: int = 50,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Train on one fixed batch and mask to test optimization and alignment."""
    if steps <= 0 or learning_rate <= 0.0 or log_every <= 0:
        raise ValueError("steps, learning_rate, and log_every must be positive")
    torch.manual_seed(seed)
    signals = signals.to(device)
    event_masks = event_masks.to(device)
    target_mask, token_mask, _ = build_mask_batch(
        signals.detach().cpu(), event_masks.detach().cpu(), config, seed
    )
    target_mask = target_mask.to(device)
    token_mask = token_mask.to(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    zero_loss = float(masked_mse(torch.zeros_like(signals), signals, target_mask))

    def evaluate() -> tuple[float, torch.Tensor]:
        model.eval()
        with torch.no_grad():
            prediction = model(signals, token_mask)["reconstruction"]
            loss = float(masked_mse(prediction, signals, target_mask))
        return loss, prediction

    initial_loss, _ = evaluate()
    history = [{"step": 0, "masked_mse": initial_loss}]
    gradient_norms: list[float] = []
    for step in range(1, steps + 1):
        model.train()
        prediction = model(signals, token_mask)["reconstruction"]
        loss = masked_mse(prediction, signals, target_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["grad_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite gradient in fixed-mask overfit diagnostic")
        gradient_norms.append(float(gradient_norm.detach().cpu()))
        optimizer.step()
        if step % log_every == 0 or step == steps:
            current_loss, _ = evaluate()
            history.append({"step": step, "masked_mse": current_loss})

    final_loss, final_prediction = evaluate()
    selected_prediction = final_prediction[:, 0][target_mask]
    selected_target = signals[:, 0][target_mask]
    target_rms = float(torch.sqrt(torch.mean(torch.square(selected_target))))
    output_rms = float(torch.sqrt(torch.mean(torch.square(selected_prediction))))
    relative_improvement = (zero_loss - final_loss) / zero_loss if zero_loss > 0.0 else None
    result = {
        "n_signals": int(signals.shape[0]),
        "steps": steps,
        "learning_rate": learning_rate,
        "zero_masked_mse": zero_loss,
        "initial_masked_mse": initial_loss,
        "final_masked_mse": final_loss,
        "relative_improvement_vs_zero": relative_improvement,
        "model_output_rms_on_mask": output_rms,
        "target_rms_on_mask": target_rms,
        "output_rms_fraction_of_target": output_rms / target_rms,
        "first_gradient_norm": gradient_norms[0],
        "last_gradient_norm": gradient_norms[-1],
        "history": history,
        "gates": {
            "finite_gradients": all(torch.isfinite(torch.tensor(gradient_norms)).tolist()),
            "beats_zero": final_loss < zero_loss,
            "reduces_zero_error_by_0p80": (
                relative_improvement is not None and relative_improvement >= 0.80
            ),
            "nontrivial_amplitude_0p10": output_rms / target_rms >= 0.10,
        },
    }
    return result, final_prediction.detach().cpu(), target_mask.detach().cpu()
