from __future__ import annotations

from pathlib import Path
from typing import Any


ACTIVE_SIMULATION_DATASET = "yeast-passage-simulations@v2"
HISTORICAL_SIMULATION_DATASET = "yeast-passage-simulations@v1"
BEAD_SIMULATION_DATASET = "yeast-passage-simulations@v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path,
    *,
    _ancestry: tuple[Path, ...] = (),
) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read P3_SSL YAML configs") from exc
    config_path = Path(path)
    if not config_path.is_absolute() and not config_path.is_file():
        repository_relative = REPOSITORY_ROOT / config_path
        if repository_relative.is_file():
            config_path = repository_relative
    with config_path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    parent = data.pop("extends", None)
    if parent is not None:
        resolved = config_path.resolve()
        if resolved in _ancestry:
            raise ValueError(f"Config inheritance cycle: {config_path}")
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        base = load_config(parent_path, _ancestry=(*_ancestry, resolved))
        data = _deep_merge(base, data)
    return data


def validate_active_simulation_dataset(config: dict[str, Any]) -> None:
    study = config.get("study", {})
    selected = study.get("simulation_dataset")
    if selected is None:
        raise ValueError("Active training config must declare study.simulation_dataset")
    if selected != ACTIVE_SIMULATION_DATASET:
        raise ValueError(
            "Active training is v2-only: expected "
            f"{ACTIVE_SIMULATION_DATASET}, got {selected}. "
            f"{HISTORICAL_SIMULATION_DATASET} is historical/reference only."
        )


def validate_bead_simulation_dataset(config: dict[str, Any]) -> None:
    study = config.get("study", {})
    selected = study.get("simulation_dataset")
    if selected is None:
        raise ValueError("Bead training config must declare study.simulation_dataset")
    if selected != BEAD_SIMULATION_DATASET:
        raise ValueError(
            "Bead training is v1-selected after visual review: expected "
            f"{BEAD_SIMULATION_DATASET}, got {selected}. "
            f"{ACTIVE_SIMULATION_DATASET} remains scoped to non-bead studies."
        )


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
    study = config["study"]
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
    representation_seeds = [int(seed) for seed in config["training"]["representation_seeds"]]
    if len(representation_seeds) < 3 or len(representation_seeds) != len(
        set(representation_seeds)
    ):
        raise ValueError("At least three unique representation seeds are required")
    if int(config["training"]["seed"]) not in representation_seeds:
        raise ValueError("training.seed must be included in representation_seeds")
    if study.get("primary_endpoint_status") == "authorized_single_acquisition_in_session":
        decision = study.get("scope_decision", {})
        if not decision.get("independent_acquisition_waived"):
            raise ValueError("Single-acquisition authorization must record the acquisition waiver")
        if decision.get("acquisition_ood_claim_allowed"):
            raise ValueError("Single-acquisition scope cannot allow an acquisition-OOD claim")
        if decision.get("biological_label_claim_allowed"):
            raise ValueError("Source-group proxies cannot authorize a biological-label claim")
    if float(masking["min_block_ms"]) <= 0 or float(masking["max_block_ms"]) < float(
        masking["min_block_ms"]
    ):
        raise ValueError("Invalid masking block duration")
    if not 0.0 < float(masking["mask_ratio"]) < 1.0:
        raise ValueError("masking.mask_ratio must be in (0, 1)")
    for key, default in (
        ("high_derivative_probability", 0.25),
        ("event_biased_probability", 0.0),
    ):
        if not 0.0 <= float(masking.get(key, default)) <= 1.0:
            raise ValueError(f"masking.{key} must be in [0, 1]")
    strategy = str(masking.get("strategy", "time_blocks"))
    if strategy not in {"time_blocks", "patch_aligned_isolated"}:
        raise ValueError(f"Unsupported masking strategy: {strategy}")
    if strategy == "patch_aligned_isolated" and int(
        masking.get("minimum_visible_tokens_between_masks", 1)
    ) < 0:
        raise ValueError("minimum_visible_tokens_between_masks must be non-negative")
    roles = (
        set(policy["preserve_predict"]),
        set(policy["randomize_invariant"]),
        set(policy["unresolved_excluded"]),
    )
    if any(left & right for index, left in enumerate(roles) for right in roles[index + 1 :]):
        raise ValueError("Information-policy roles must be disjoint")


def validate_mask_ablation_config(config: dict[str, Any]) -> None:
    policies = config.get("policies", {})
    expected = {"L0", "S25", "S10", "SE25", "SE10", "P25", "P10", "PE25", "PE10"}
    if set(policies) != expected:
        raise ValueError(f"Mask ablation policies must be {sorted(expected)}")
    for name, policy in policies.items():
        if not 0.0 < float(policy["mask_ratio"]) < 1.0:
            raise ValueError(f"{name}.mask_ratio must be in (0, 1)")
        if float(policy["min_block_ms"]) <= 0.0 or float(policy["max_block_ms"]) < float(
            policy["min_block_ms"]
        ):
            raise ValueError(f"{name} has invalid block durations")
        for key in ("high_derivative_probability", "event_biased_probability"):
            if not 0.0 <= float(policy[key]) <= 1.0:
                raise ValueError(f"{name}.{key} must be in [0, 1]")
        strategy = str(policy.get("strategy", "time_blocks"))
        if strategy not in {"time_blocks", "patch_aligned_isolated"}:
            raise ValueError(f"{name}.strategy is unsupported")
        if strategy == "patch_aligned_isolated" and int(
            policy.get("minimum_visible_tokens_between_masks", 1)
        ) < 0:
            raise ValueError(f"{name} has a negative visible-token gap")
    if config["training"].get("architecture_change_allowed") is not False:
        raise ValueError("The first mask ablation must hold architecture fixed")
    if config["training"].get("target_change_allowed") is not False:
        raise ValueError("The first mask ablation must hold the reconstruction target fixed")
    candidates = set(config["training"].get("candidate_policies", []))
    diagnostics = set(config["training"].get("diagnostic_only_policies", []))
    if candidates != {"P25", "P10", "PE25", "PE10"}:
        raise ValueError("Only patch-aligned policies may enter candidate training")
    if candidates & diagnostics or candidates | diagnostics != expected:
        raise ValueError("Candidate and diagnostic mask policies must partition all policies")
    next_stage = config.get("conditional_next_stage", {})
    if next_stage.get("frozen_before_seed42_outcomes") is not True:
        raise ValueError("Conditional next-stage branches must be frozen before seed-42 outcomes")
    collapse_branch = next_stage.get("branch_pretext_pass_geometry_fail", {})
    if collapse_branch.get("cells") != {
        "C0": {"time_reconstruction": True, "vicreg": False},
        "C1": {"time_reconstruction": True, "vicreg": True},
    }:
        raise ValueError("The anti-collapse branch must contain only the paired C0/C1 contrast")
    if float(collapse_branch.get("vicreg_global_weight", -1.0)) != 1.0:
        raise ValueError("The conditional anti-collapse weight must remain frozen at 1.0")
    if next_stage.get("multiseed_authorization", {}).get("required_before_seeds_43_44") != (
        "pretext_geometry_and_development_utility_pass_at_seed_42"
    ):
        raise ValueError("Additional mask-ablation seeds require all seed-42 gates")


def validate_mask_collapse_config(config: dict[str, Any]) -> None:
    study = config.get("study", {})
    if study.get("sealed_splits_allowed") is not False:
        raise ValueError("The anti-collapse contrast is development-only")
    training = config.get("training", {})
    if training.get("cells") != {
        "C0": {"time_reconstruction": True, "vicreg": False},
        "C1": {"time_reconstruction": True, "vicreg": True},
    }:
        raise ValueError("The anti-collapse study must contain only C0 and C1")
    if int(training.get("first_stage_seed", -1)) != 42:
        raise ValueError("The anti-collapse first stage must remain frozen at seed 42")
    if float(training.get("vicreg_global_weight", -1.0)) != 1.0:
        raise ValueError("The anti-collapse VICReg weight must remain frozen at 1.0")
    if training.get("drop_last_training_batch") is not True:
        raise ValueError("C0/C1 must share the singleton-safe drop-last batch policy")
    vicreg = training.get("vicreg", {})
    if vicreg != {
        "invariance_weight": 1.0,
        "variance_weight": 1.0,
        "covariance_weight": 0.04,
        "variance_floor": 0.5,
        "epsilon": 0.0001,
        "second_view_seed_offset": 10000019,
    }:
        raise ValueError("VICReg coefficients or second-view seed offset changed")
    gates = config.get("promotion_gates", {})
    if gates.get("comparison") != {
        "require_beats_interpolation": True,
        "require_geometry_improvement_over_c0": True,
    }:
        raise ValueError("C1 must beat interpolation and improve geometry over C0")
    decision = config.get("decision", {})
    if decision.get("success_action") != "run_development_utility_evaluation":
        raise ValueError("Successful C1 must proceed only to development utility")
    if decision.get("failure_action") != "run_preregistered_phase_invariant_target_contrast":
        raise ValueError("Failed C1 must trigger the phase-invariant target branch")
    if decision.get("additional_seed_authorization") != (
        "requires_seed42_pretext_geometry_and_development_utility_pass"
    ):
        raise ValueError("C0/C1 cannot directly authorize additional seeds")


def validate_collapse_utility_config(config: dict[str, Any]) -> None:
    study = config.get("study", {})
    if study.get("sealed_splits_allowed") is not False:
        raise ValueError("Collapse utility must remain development-only")
    if study.get("real_dataset") != "yeast-events-development@v1":
        raise ValueError("Collapse utility requires the physically development-only dataset")
    if study.get("real_manifest_sha256") != (
        "42101c3e12dd15a0c11e26e7f4d337aae7977e1c9fd503b78583ecce4823a1de"
    ):
        raise ValueError("Collapse utility dataset manifest changed")
    methods = config.get("methods", {})
    if list(methods) != ["C1", "C0", "random", "raw", "pca96"]:
        raise ValueError("Collapse utility methods and order are frozen")
    pca = methods["pca96"]
    if pca != {
        "components": 96,
        "fit_split": "development_train",
        "svd_solver": "randomized",
        "random_state": 42,
        "whiten": False,
    }:
        raise ValueError("PCA-96 definition changed")
    primary = config.get("primary_gate", {})
    if primary != {
        "label_fraction": 0.1,
        "metric": "macro_f1",
        "minimum_gain": 0.03,
        "paired_interval_level": 0.95,
        "require_interval_lower_above_zero": True,
        "require_positive_gain_vs_each_baseline": True,
        "group_unit": "capture_block_id_fallback_record_id",
        "success_action": "authorize_representation_seeds_43_44",
        "failure_action": "reject_mask_only_rescue",
    }:
        raise ValueError("Collapse utility primary gate changed")
    full = config.get("profiles", {}).get("full", {})
    if full.get("label_fractions") != [0.01, 0.05, 0.1, 0.25, 1.0]:
        raise ValueError("Full utility label fractions changed")
    if full.get("probe_seeds") != [42, 43, 44]:
        raise ValueError("Full utility probe seeds changed")
    if int(full.get("grouped_bootstrap_repeats", 0)) != 1000:
        raise ValueError("Full utility requires 1000 paired bootstrap repeats")


def validate_local_spectral_study_config(config: dict[str, Any]) -> None:
    study = config.get("study", {})
    if study.get("sealed_splits_allowed") is not False:
        raise ValueError("S1 must remain development-only")
    if study.get("dataset") != "yeast-events-representation@v3":
        raise ValueError("S1 must use the same real dataset as C1")
    if study.get("source_mask_policy") != "PE25":
        raise ValueError("S1 must retain the selected C1 mask policy")
    target = config.get("target", {})
    expected_target = {
        "kind": "analytic_local_log_power",
        "input_length": 4096,
        "sampling_frequency_hz": 1000000.0,
        "patch_size": 16,
        "window_samples": 256,
        "first_frequency_bin": 2,
        "stop_frequency_bin": 26,
        "first_valid_token": 8,
        "stop_valid_token": 248,
        "feature_count": 24,
        "normalization": "per_trace_mean_retained_power",
        "transform": "log1p",
        "compute_before_masking": True,
    }
    if target != expected_target:
        raise ValueError("S1 local spectral target differs from the single frozen definition")
    training = config.get("training", {})
    if int(training.get("seed", -1)) != 42:
        raise ValueError("S1 first stage is restricted to seed 42")
    if training.get("control_cell") != "C1" or training.get("treatment_cell") != "S1":
        raise ValueError("S1 must change only the C1 prediction target")
    if float(training.get("vicreg_global_weight", -1.0)) != 1.0:
        raise ValueError("S1 must retain the C1 VICReg weight")
    if training.get("drop_last_training_batch") is not True:
        raise ValueError("S1 must retain the C1 drop-last policy")
    vicreg = training.get("vicreg", {})
    if vicreg != {
        "invariance_weight": 1.0,
        "variance_weight": 1.0,
        "covariance_weight": 0.04,
        "variance_floor": 0.5,
        "epsilon": 0.0001,
        "second_view_seed_offset": 10000019,
    }:
        raise ValueError("S1 VICReg coefficients differ from C1")
    gates = config.get("gates", {})
    if gates.get("preflight") != {
        "phase_rotation_relative_error_max": 0.002,
        "mask_target_independence_required": True,
        "fixed_one_and_eight_improvement_min": 0.8,
        "finite_nonzero_encoder_and_head_gradients": True,
    }:
        raise ValueError("S1 preflight gates changed")
    if gates.get("held_out") != {
        "require_beats_zero": True,
        "require_beats_train_constant": True,
        "require_beats_feature_of_interpolation": True,
        "output_rms_fraction_of_target_min": 0.1,
        "effective_rank_min": 8.0,
        "mean_pairwise_cosine_max": 0.95,
        "report_event_background_boundary": True,
    }:
        raise ValueError("S1 held-out gates changed")
    if config.get("decision") != {
        "success_action": "run_development_utility_with_handcrafted_controls",
        "failure_action": "end_objective_rescue_negative",
        "additional_seed_authorization": "forbidden_until_all_seed42_gates_and_utility_pass",
    }:
        raise ValueError("S1 terminal decision policy changed")
