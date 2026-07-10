from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .serialization import json_safe


REQUIRED_RUN_ARTIFACTS = (
    "synthetic_manifest.csv",
    "hybrid_manifest.csv",
    "training_history.json",
    "physical_metrics.json",
    "physical_dashboard.png",
    "physical_dashboard.pdf",
    "reconstruction_metrics_val.json",
    "reconstruction_metrics_test.json",
    "robustness_metrics.json",
    "real_estimated_physics_metrics.json",
    "reconstruction_reference_comparison.json",
    "classic_assessment/README.md",
    "classic_assessment/classic_assessment_summary.json",
    "classic_assessment/hybrid_event_embeddings.npz",
    "classic_assessment/representation_manifold_metrics.json",
    "classic_assessment/representation_manifold.pdf",
    "classic_assessment/label_efficiency_summary.json",
    "classic_assessment/label_efficiency_metrics.csv",
    "classic_assessment/retrieval_metrics.json",
    "classic_assessment/retrieval_purity.pdf",
    "classic_assessment/assessment_dashboard.json",
    "classic_assessment/physical_baselines/physical_metrics.json",
    "classic_assessment/physical_baselines/physical_ranking.md",
    "run_summary.json",
    "run_summary.md",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _gate(name: str, passed: bool, evidence: Any = None, required: bool = True) -> dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "required": bool(required),
        "evidence": evidence,
    }


def _positive_param_spearman(metrics: dict[str, Any]) -> dict[str, Any]:
    per_param = metrics.get("per_parameter", {})
    required = ("A", "fD_khz", "t0_fraction", "tau_ms")
    values = {
        name: (per_param.get(name) or {}).get("spearman")
        for name in required
    }
    phi = (per_param.get("phi_rad") or {}).get("spearman")
    passed = all(value is not None and float(value) > 0.0 for value in values.values()) and phi is not None
    return {"pass": passed, "values": values, "phi_circular_score": phi}


def _reconstruction_pass(comparison: dict[str, Any]) -> dict[str, Any]:
    split_status = {
        split: result.get("reconstruction_regression_pass")
        for split, result in comparison.items()
        if isinstance(result, dict)
    }
    passed = bool(split_status) and all(value is True for value in split_status.values())
    return {"pass": passed, "splits": split_status}


def _robustness_pass(robustness: dict[str, Any]) -> dict[str, Any]:
    perturbations = robustness.get("perturbations", {})
    required = {"noise_0p10", "scale_1p25", "shift_8", "center_mask_64"}
    present = set(perturbations)
    passed = robustness.get("status") == "ok" and required <= present
    return {"pass": passed, "status": robustness.get("status"), "present": sorted(present), "required": sorted(required)}


def assess_hybrid_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    physical = _load_json(root / "physical_metrics.json")
    summary = _load_json(root / "run_summary.json")
    baseline = _load_json(root / "classic_assessment" / "physical_baselines" / "physical_metrics.json")
    reconstruction = _load_json(root / "reconstruction_reference_comparison.json")
    robustness = _load_json(root / "robustness_metrics.json")

    missing = [path for path in REQUIRED_RUN_ARTIFACTS if not (root / path).is_file()]
    param_spearman = _positive_param_spearman(physical)
    reconstruction_gate = _reconstruction_pass(reconstruction)
    robustness_gate = _robustness_pass(robustness)
    candidate_comparison = baseline.get("candidate_comparison") or summary.get("physical_baseline_summary", {}).get("candidate_comparison", {})
    neighbor_gain = physical.get("neighbor_gain")

    gates = [
        _gate("required_artifacts_present", not missing, {"missing": missing}),
        _gate("physical_validation_pass", physical.get("physical_validation_pass") is True, {"physical_score": physical.get("physical_score")}),
        _gate("positive_core_parameter_spearman", param_spearman["pass"], param_spearman),
        _gate("latent_neighbors_closer_than_random", neighbor_gain is not None and float(neighbor_gain) > 0.0, {"neighbor_gain": neighbor_gain}),
        _gate("beats_random_baseline", candidate_comparison.get("beats_random") is True, candidate_comparison),
        _gate("beats_raw_signal_baseline", candidate_comparison.get("beats_raw") is True, candidate_comparison),
        _gate("beats_reconstruction_only_baseline", candidate_comparison.get("beats_reconstruction_only") is True, candidate_comparison),
        _gate("real_reconstruction_not_regressed", reconstruction_gate["pass"], reconstruction_gate),
        _gate("robustness_documented", robustness_gate["pass"], robustness_gate),
    ]
    required_pass = all(gate["pass"] for gate in gates if gate["required"])
    return {
        "run_dir": str(root),
        "assessment_pass": bool(required_pass),
        "gates": gates,
        "physical_score": physical.get("physical_score"),
        "summary": {
            "physical_validation_pass": physical.get("physical_validation_pass"),
            "candidate_comparison": candidate_comparison,
            "reconstruction": reconstruction_gate,
            "robustness": robustness_gate,
        },
    }


def write_run_assessment(assessment: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_assessment.json").write_text(json.dumps(json_safe(assessment), indent=2, allow_nan=False))
    lines = ["# Hybrid Physics Run Assessment", ""]
    lines.append(f"- assessment_pass: {assessment['assessment_pass']}")
    lines.append(f"- physical_score: {assessment.get('physical_score')}")
    lines.append("")
    lines.append("| gate | pass | required |")
    lines.append("|---|---:|---:|")
    for gate in assessment["gates"]:
        lines.append(f"| {gate['name']} | {gate['pass']} | {gate['required']} |")
    lines.append("")
    lines.append("This gate treats class retrieval and label efficiency as secondary diagnostics, not pass criteria.")
    (output / "run_assessment.md").write_text("\n".join(lines) + "\n")
