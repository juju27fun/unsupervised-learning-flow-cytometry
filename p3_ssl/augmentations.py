from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_distance_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"embedding shapes must match, got {tuple(a.shape)} and {tuple(b.shape)}")
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


def positive_signal_augmentation(
    signals: torch.Tensor,
    noise_std_fraction: float = 0.05,
    max_shift_points: int = 8,
    amplitude_scale_min: float = 0.90,
    amplitude_scale_max: float = 1.10,
    phase_jitter_rad: float = 0.05,
) -> torch.Tensor:
    """Apply small nuisance transforms that should preserve coarse particle identity."""
    if signals.ndim != 3 or signals.shape[1] != 1:
        raise ValueError(f"Expected signals with shape (B, 1, L), got {tuple(signals.shape)}")
    x = signals
    bsz = x.shape[0]
    device = x.device
    dtype = x.dtype

    if amplitude_scale_min > 0.0 or amplitude_scale_max > 0.0:
        scale = torch.empty(bsz, 1, 1, device=device, dtype=dtype).uniform_(amplitude_scale_min, amplitude_scale_max)
        x = x * scale
    if max_shift_points > 0:
        shifts = torch.randint(-max_shift_points, max_shift_points + 1, (bsz,), device=device)
        x = torch.stack([torch.roll(sample, shifts=int(shift.item()), dims=-1) for sample, shift in zip(x, shifts)], dim=0)
    if phase_jitter_rad > 0.0:
        spectrum = torch.fft.rfft(x, dim=-1)
        phase = torch.empty(bsz, 1, spectrum.shape[-1], device=device, dtype=x.real.dtype).uniform_(
            -phase_jitter_rad,
            phase_jitter_rad,
        )
        phase[..., 0] = 0.0
        if x.shape[-1] % 2 == 0:
            phase[..., -1] = 0.0
        spectrum = spectrum * torch.exp(1j * phase)
        x = torch.fft.irfft(spectrum, n=x.shape[-1], dim=-1).to(dtype=dtype)
    if noise_std_fraction > 0.0:
        scale = x.std(dim=-1, keepdim=True).clamp_min(1.0e-6)
        x = x + torch.randn_like(x) * scale * noise_std_fraction
    return x
