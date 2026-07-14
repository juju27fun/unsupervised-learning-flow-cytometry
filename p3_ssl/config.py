from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read P3_SSL YAML configs") from exc
    with Path(path).open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def ssl_token_count(input_length: int, patch_size: int, patch_stride: int) -> int:
    if input_length <= 0:
        raise ValueError("input_length must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if patch_stride <= 0:
        raise ValueError("patch_stride must be positive")
    if patch_size > input_length:
        raise ValueError("patch_size must be <= input_length")
    return 1 + (input_length - patch_size) // patch_stride


def validate_ssl_config(config: dict[str, Any]) -> None:
    data = config["data"]
    patching = config["patching"]
    model = config["model"]
    input_length_raw = int(data["input_length_raw"])
    input_length_ssl = int(data["input_length_ssl"])
    decimation_factor = int(data["decimation_factor"])
    patch_size = int(patching["patch_size"])
    patch_stride = int(patching["patch_stride"])
    max_tokens = int(model.get("max_tokens", 1024))

    if input_length_raw % input_length_ssl != 0:
        raise ValueError(
            "input_length_raw must be divisible by input_length_ssl: "
            f"{input_length_raw} % {input_length_ssl} != 0"
        )
    expected_decimation = input_length_raw // input_length_ssl
    if decimation_factor != expected_decimation:
        raise ValueError(
            "decimation_factor must match input_length_raw // input_length_ssl: "
            f"got {decimation_factor}, expected {expected_decimation}"
        )
    n_tokens = ssl_token_count(input_length_ssl, patch_size, patch_stride)
    if n_tokens > max_tokens:
        raise ValueError(
            "Computed token count exceeds model.max_tokens: "
            f"{n_tokens} > {max_tokens}"
        )


def validate_study_config(config: dict[str, Any]) -> None:
    data = config["data"]
    model = config["model"]
    masking = config["masking"]
    policy = config["information_policy"]
    input_length = int(data["input_length"])
    tokens = ssl_token_count(
        input_length,
        int(model["patch_size"]),
        int(model["patch_stride"]),
    )
    if tokens != int(model["max_tokens"]):
        raise ValueError(f"Study token count must equal model.max_tokens: {tokens}")
    forbidden = set(data.get("forbidden_training_splits", []))
    selected = {
        data["real_train_split"],
        data["real_validation_split"],
        data["simulation_train_split"],
        data["simulation_validation_split"],
    }
    overlap = forbidden & selected
    if overlap:
        raise ValueError(f"Forbidden split selected for training/validation: {sorted(overlap)}")
    if float(masking["min_block_ms"]) <= 0 or float(masking["max_block_ms"]) < float(
        masking["min_block_ms"]
    ):
        raise ValueError("Invalid masking block duration")
    roles = (
        set(policy["preserve_predict"]),
        set(policy["randomize_invariant"]),
        set(policy["unresolved_excluded"]),
    )
    if any(left & right for index, left in enumerate(roles) for right in roles[index + 1 :]):
        raise ValueError("Information-policy roles must be disjoint")
