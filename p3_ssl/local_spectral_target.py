from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LocalSpectralTargetConfig:
    input_length: int = 4096
    sampling_frequency_hz: float = 1_000_000.0
    patch_size: int = 16
    window_samples: int = 256
    first_frequency_bin: int = 2
    stop_frequency_bin: int = 26
    first_valid_token: int = 8
    stop_valid_token: int = 248
    epsilon: float = 1.0e-12

    @property
    def feature_count(self) -> int:
        return self.stop_frequency_bin - self.first_frequency_bin

    @property
    def valid_token_count(self) -> int:
        return self.stop_valid_token - self.first_valid_token


def validate_local_spectral_config(config: LocalSpectralTargetConfig) -> None:
    if config.input_length != 4096 or config.patch_size != 16:
        raise ValueError("S1 requires the frozen 4096-sample, 16-sample-patch contract")
    if config.window_samples != 256 or config.sampling_frequency_hz != 1_000_000.0:
        raise ValueError("S1 requires one frozen 256-sample window at 1 MHz")
    if (config.first_frequency_bin, config.stop_frequency_bin) != (2, 26):
        raise ValueError("S1 requires the frozen 7.8125-97.65625 kHz bins")
    if (config.first_valid_token, config.stop_valid_token) != (8, 248):
        raise ValueError("S1 requires center tokens [8, 248)")
    if config.feature_count != 24 or config.valid_token_count != 240:
        raise ValueError("S1 target dimensions differ from the frozen contract")
    if config.epsilon <= 0.0:
        raise ValueError("epsilon must be positive")


def local_spectral_frequencies(
    config: LocalSpectralTargetConfig = LocalSpectralTargetConfig(),
) -> torch.Tensor:
    validate_local_spectral_config(config)
    bins = torch.arange(config.first_frequency_bin, config.stop_frequency_bin)
    return bins * (config.sampling_frequency_hz / config.window_samples)


def analytic_signal(signals: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(signals):
        raise TypeError("analytic_signal expects real floating-point input")
    length = signals.shape[-1]
    spectrum = torch.fft.fft(signals, dim=-1)
    multiplier = torch.zeros(length, dtype=signals.dtype, device=signals.device)
    multiplier[0] = 1.0
    if length % 2 == 0:
        multiplier[1 : length // 2] = 2.0
        multiplier[length // 2] = 1.0
    else:
        multiplier[1 : (length + 1) // 2] = 2.0
    return torch.fft.ifft(spectrum * multiplier, dim=-1)


def local_spectral_target(
    signals: torch.Tensor,
    config: LocalSpectralTargetConfig = LocalSpectralTargetConfig(),
) -> torch.Tensor:
    """Compute S1 targets from complete signals before any masking."""
    validate_local_spectral_config(config)
    if signals.ndim == 3:
        if signals.shape[1] != 1:
            raise ValueError("S1 expects one signal channel")
        values = signals[:, 0]
    elif signals.ndim == 2:
        values = signals
    else:
        raise ValueError("signals must have shape (batch, time) or (batch, 1, time)")
    if values.shape[-1] != config.input_length:
        raise ValueError("Signal length differs from the S1 input contract")
    analytic = analytic_signal(values)
    edge_offset = config.patch_size // 2
    framed = analytic[:, edge_offset:-edge_offset].unfold(
        -1, config.window_samples, config.patch_size
    )
    if framed.shape[1] != config.valid_token_count:
        raise RuntimeError("Unexpected number of local spectral frames")
    window = torch.hann_window(
        config.window_samples,
        periodic=True,
        dtype=values.dtype,
        device=values.device,
    )
    spectrum = torch.fft.fft(framed * window, dim=-1)
    power = spectrum[..., config.first_frequency_bin : config.stop_frequency_bin].abs().square()
    reference = power.mean(dim=(1, 2), keepdim=True).clamp_min(config.epsilon)
    return torch.log1p(power / reference).to(dtype=values.dtype)


def masked_local_spectral_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
    config: LocalSpectralTargetConfig = LocalSpectralTargetConfig(),
) -> torch.Tensor:
    validate_local_spectral_config(config)
    expected_prediction = (target.shape[0], config.input_length // config.patch_size, config.feature_count)
    expected_target = (target.shape[0], config.valid_token_count, config.feature_count)
    if tuple(prediction.shape) != expected_prediction or tuple(target.shape) != expected_target:
        raise ValueError("Prediction or target shape differs from the S1 contract")
    if tuple(token_mask.shape) != expected_prediction[:2]:
        raise ValueError("token_mask shape differs from the S1 token sequence")
    valid_prediction = prediction[:, config.first_valid_token : config.stop_valid_token]
    valid_mask = token_mask[:, config.first_valid_token : config.stop_valid_token]
    count = valid_mask.sum()
    if int(count) == 0:
        raise ValueError("S1 loss requires at least one masked valid token")
    squared = (valid_prediction - target).square()
    weighted = squared * valid_mask.to(squared.dtype).unsqueeze(-1)
    return weighted.sum() / (count * config.feature_count)


def local_spectral_frame_regions(
    event_masks: torch.Tensor,
    config: LocalSpectralTargetConfig = LocalSpectralTargetConfig(),
) -> dict[str, torch.Tensor]:
    validate_local_spectral_config(config)
    if tuple(event_masks.shape[1:]) != (config.input_length,):
        raise ValueError("event_masks must have shape (batch, 4096)")
    edge_offset = config.patch_size // 2
    frames = event_masks.bool()[:, edge_offset:-edge_offset].unfold(
        -1, config.window_samples, config.patch_size
    )
    any_event = frames.any(dim=-1)
    all_event = frames.all(dim=-1)
    return {
        "event": all_event,
        "background": ~any_event,
        "boundary": any_event & ~all_event,
    }
