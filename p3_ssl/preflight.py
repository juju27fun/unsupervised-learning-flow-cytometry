from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import torch

from .hybrid_training import profile_value
from .run_assessment import REQUIRED_RUN_ARTIFACTS


def _check(name: str, passed: bool, evidence: Any = None, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "required": bool(required),
        "evidence": evidence,
    }


def _path_exists(path: str | Path | None) -> bool:
    return path is not None and Path(path).exists()


def _path_is_file(path: str | Path | None) -> bool:
    return path is not None and Path(path).is_file()


def _requested_sources(raw: str | None, default: str) -> set[str]:
    value = raw if raw is not None else default
    return {item.strip() for item in str(value).split(",") if item.strip()}


def _manifest_split_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "")
            counts[split] = counts.get(split, 0) + 1
    return counts


def _output_root_ready(path: Path) -> dict[str, Any]:
    if path.exists():
        return {"exists": True, "is_dir": path.is_dir(), "writable": os.access(path, os.W_OK)}
    parent = path.parent
    return {
        "exists": False,
        "parent": str(parent),
        "parent_exists": parent.exists(),
        "parent_writable": parent.exists() and os.access(parent, os.W_OK),
    }


def _has_profile_value(config_value: Any, profile: str) -> bool:
    if isinstance(config_value, dict):
        return profile in config_value or "full" in config_value
    return config_value is not None


def preflight_hybrid_run(
    config: dict[str, Any],
    real_manifest: str | Path,
    profile: str = "full",
    device: str = "cuda",
    simulation_source: str | None = None,
    output_root: str | Path | None = None,
    cuda_available: bool | None = None,
) -> dict[str, Any]:
    """Check whether the canonical hybrid physical run can start and be assessed."""
    checks: list[dict[str, Any]] = []
    required_sections = (
        "simulation",
        "hybrid_sampling",
        "contrastive_loss",
        "reconstruction_loss",
        "physics_metrics",
        "real_adaptation",
        "baseline_assessment",
        "classic_assessment",
        "training",
    )
    missing_sections = [section for section in required_sections if section not in config]
    checks.append(_check("required_config_sections", not missing_sections, {"missing": missing_sections}))

    for section, key in (
        ("simulation", "n_synthetic"),
        ("hybrid_sampling", "max_real_rows"),
        ("training", "epochs"),
        ("training", "batch_size"),
        ("real_adaptation", "epochs"),
        ("baseline_assessment", "sweep_dir"),
        ("baseline_assessment", "max_combined_samples"),
        ("classic_assessment", "max_real_rows"),
        ("classic_assessment", "label_fractions"),
    ):
        value = config.get(section, {}).get(key)
        checks.append(
            _check(
                f"profile_value_{section}.{key}",
                _has_profile_value(value, profile),
                {"profile": profile, "value": value},
            )
        )

    real_manifest_path = Path(real_manifest)
    real_manifest_exists = real_manifest_path.is_file()
    split_counts = _manifest_split_counts(real_manifest_path) if real_manifest_exists else {}
    expected_splits = {
        str(config.get("data", {}).get("split_train", "train")),
        str(config.get("data", {}).get("split_val", "val")),
        str(config.get("data", {}).get("split_test", "test")),
    }
    checks.append(
        _check(
            "real_manifest_present_with_expected_splits",
            real_manifest_exists and expected_splits <= set(split_counts),
            {"path": str(real_manifest_path), "split_counts": split_counts, "expected_splits": sorted(expected_splits)},
        )
    )

    sim_cfg = config.get("simulation", {})
    sources = _requested_sources(simulation_source, str(sim_cfg.get("source", "internal")))
    event_manifest = sim_cfg.get("particles2snr_event_manifest")
    checks.append(
        _check(
            "particles2snr_event_manifest_present",
            "particles2snr_pipeline" not in sources and "particles2snr" not in sources or _path_is_file(event_manifest),
            {"requested_sources": sorted(sources), "path": event_manifest},
        )
    )

    baseline_cfg = config.get("baseline_assessment", {})
    sweep_dir_value = profile_value(baseline_cfg.get("sweep_dir"), profile)
    sweep_dir = Path(str(sweep_dir_value)) if sweep_dir_value is not None else None
    sweep_required_files = ("synthetic_metadata.csv", "synthetic_signals_encoded.npz")
    missing_sweep_files = [
        name for name in sweep_required_files
        if sweep_dir is None or not (sweep_dir / name).is_file()
    ]
    expected_model_dirs = ("moment_official", "patchtst_pretrained", "conv1dgap_same_input_3class")
    missing_model_embeddings = [
        model for model in expected_model_dirs
        if sweep_dir is None or not (sweep_dir / model / "embeddings.npz").is_file()
    ]
    checks.append(
        _check(
            "physical_baseline_sweep_artifacts_present",
            sweep_dir is not None and not missing_sweep_files and not missing_model_embeddings,
            {
                "sweep_dir": "" if sweep_dir is None else str(sweep_dir),
                "missing_files": missing_sweep_files,
                "missing_model_embeddings": missing_model_embeddings,
            },
        )
    )

    reconstruction_checkpoint = baseline_cfg.get("reconstruction_only_checkpoint")
    checks.append(
        _check(
            "reconstruction_only_checkpoint_present",
            _path_is_file(reconstruction_checkpoint),
            {"path": reconstruction_checkpoint},
        )
    )

    reference_root = config.get("real_adaptation", {}).get("reconstruction_reference_root")
    reference_files = [
        Path(str(reference_root)) / "eval_val" / "metrics.json" if reference_root else None,
        Path(str(reference_root)) / "eval_test" / "metrics.json" if reference_root else None,
    ]
    checks.append(
        _check(
            "reconstruction_reference_metrics_present",
            all(_path_is_file(path) for path in reference_files),
            {"paths": ["" if path is None else str(path) for path in reference_files]},
        )
    )

    resolved_output_root = Path(output_root or config.get("paths", {}).get("output_root", "artifacts/unsupervised-learning-flow-cytometry/runs"))
    output_evidence = _output_root_ready(resolved_output_root)
    output_ready = (
        bool(output_evidence.get("exists") and output_evidence.get("is_dir") and output_evidence.get("writable"))
        or bool(output_evidence.get("parent_exists") and output_evidence.get("parent_writable"))
    )
    checks.append(_check("output_root_writable", output_ready, {"path": str(resolved_output_root), **output_evidence}))

    cuda_ok = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    device_requires_cuda = str(device).startswith("cuda")
    checks.append(
        _check(
            "requested_device_available",
            not device_requires_cuda or cuda_ok,
            {"device": device, "torch_cuda_available": cuda_ok},
        )
    )

    checks.append(
        _check(
            "assessment_artifact_contract_nonempty",
            bool(REQUIRED_RUN_ARTIFACTS),
            {"required_artifacts": list(REQUIRED_RUN_ARTIFACTS)},
        )
    )

    required_failures = [check for check in checks if check["required"] and not check["pass"]]
    return {
        "profile": profile,
        "device": device,
        "run_ready": not required_failures,
        "checks": checks,
        "required_failures": required_failures,
    }
