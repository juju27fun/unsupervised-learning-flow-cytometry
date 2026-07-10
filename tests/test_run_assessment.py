from __future__ import annotations

import json

from p3_ssl.run_assessment import REQUIRED_RUN_ARTIFACTS, assess_hybrid_run, write_run_assessment


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_minimal_run(
    root,
    *,
    beats_raw: bool = True,
    beats_reconstruction_only: bool = True,
    reconstruction_pass: bool = True,
) -> None:
    for rel in REQUIRED_RUN_ARTIFACTS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text("{}")
        else:
            path.write_text("ok\n")
    physical = {
        "physical_score": 0.8,
        "physical_validation_pass": True,
        "neighbor_gain": 0.2,
        "per_parameter": {
            "A": {"spearman": 0.5},
            "fD_khz": {"spearman": 0.5},
            "phi_rad": {"spearman": 0.5},
            "t0_fraction": {"spearman": 0.5},
            "tau_ms": {"spearman": 0.5},
        },
    }
    _write_json(root / "physical_metrics.json", physical)
    _write_json(
        root / "classic_assessment" / "physical_baselines" / "physical_metrics.json",
        {
            "candidate_comparison": {
                "beats_random": True,
                "beats_raw": beats_raw,
                "beats_reconstruction_only": beats_reconstruction_only,
            }
        },
    )
    _write_json(
        root / "reconstruction_reference_comparison.json",
        {
            "val": {"reconstruction_regression_pass": reconstruction_pass},
            "test": {"reconstruction_regression_pass": reconstruction_pass},
        },
    )
    _write_json(
        root / "robustness_metrics.json",
        {
            "status": "ok",
            "perturbations": {
                "noise_0p10": {},
                "scale_1p25": {},
                "shift_8": {},
                "center_mask_64": {},
            },
        },
    )


def test_assess_hybrid_run_passes_when_all_required_gates_pass(tmp_path) -> None:
    _write_minimal_run(tmp_path)
    assessment = assess_hybrid_run(tmp_path)
    assert assessment["assessment_pass"] is True
    assert all(gate["pass"] for gate in assessment["gates"])
    write_run_assessment(assessment, tmp_path)
    assert (tmp_path / "run_assessment.json").is_file()
    assert (tmp_path / "run_assessment.md").is_file()


def test_assess_hybrid_run_fails_on_raw_or_reconstruction_regression(tmp_path) -> None:
    _write_minimal_run(tmp_path, beats_raw=False, reconstruction_pass=False)
    assessment = assess_hybrid_run(tmp_path)
    assert assessment["assessment_pass"] is False
    failed = {gate["name"] for gate in assessment["gates"] if not gate["pass"]}
    assert "beats_raw_signal_baseline" in failed
    assert "real_reconstruction_not_regressed" in failed


def test_assess_hybrid_run_fails_when_reconstruction_only_baseline_wins(tmp_path) -> None:
    _write_minimal_run(tmp_path, beats_reconstruction_only=False)
    assessment = assess_hybrid_run(tmp_path)
    assert assessment["assessment_pass"] is False
    failed = {gate["name"] for gate in assessment["gates"] if not gate["pass"]}
    assert "beats_reconstruction_only_baseline" in failed
