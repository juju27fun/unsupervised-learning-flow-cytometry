from __future__ import annotations

import pytest

from p3_ssl.study_reporting import _decision_summary, paired_bootstrap_comparisons


def _row(key: str, name: str, values: list[float]) -> dict:
    return {
        key: name,
        "label_fraction": 0.1,
        "grouped_bootstrap": {"metrics": {"macro_f1": {"replicates": values}}},
    }


def test_paired_bootstrap_comparisons_preserve_pairing() -> None:
    a0 = {"results": [_row("method", "handcrafted", [0.4, 0.5]), _row("method", "moment", [0.3, 0.2])]}
    checkpoints = {
        "results": [
            _row("cell", "A2", [0.1, 0.2]),
            _row("cell", "A3", [0.2, 0.3]),
            _row("cell", "A4", [0.3, 0.4]),
        ]
    }
    rows = paired_bootstrap_comparisons(a0, checkpoints)
    assert len(rows) == 4
    assert rows[0]["comparison"] == "A4 - handcrafted"
    assert rows[0]["mean_difference"] == pytest.approx(-0.1)
    assert rows[2]["mean_difference"] == pytest.approx(0.1)


def test_paired_bootstrap_comparisons_pairs_representation_and_probe_seeds() -> None:
    def row(cell: str, checkpoint: str, probe_seed: int, value: float) -> dict:
        return {
            "cell": cell,
            "checkpoint": checkpoint,
            "seed": probe_seed,
            "label_fraction": 0.1,
            "macro_f1": value,
            "grouped_bootstrap": {
                "metrics": {"macro_f1": {"replicates": [value - 0.01, value + 0.01]}}
            },
        }

    a0_rows = []
    for method, value in (("handcrafted", 0.4), ("moment", 0.3)):
        for probe_seed in (42, 43):
            item = _row("method", method, [value - 0.01, value + 0.01])
            item.update({"seed": probe_seed, "macro_f1": value})
            a0_rows.append(item)
    checkpoint_rows = []
    metadata = {}
    for representation_seed in (42, 43):
        for cell, value in (("A2", 0.2), ("A3", 0.3), ("A4", 0.5)):
            name = f"{cell.lower()}_s{representation_seed}"
            metadata[name] = {"cell": cell, "seed": representation_seed}
            for probe_seed in (42, 43):
                checkpoint_rows.append(row(cell, name, probe_seed, value))

    rows = paired_bootstrap_comparisons(
        {"results": a0_rows},
        {"results": checkpoint_rows, "checkpoint_metadata": metadata},
    )
    assert rows[0]["n_paired_runs"] == 4
    assert rows[0]["mean_difference"] == pytest.approx(0.1)
    assert rows[2]["mean_difference"] == pytest.approx(0.2)


def test_decision_summary_enforces_primary_effect_and_domain_diagnostic() -> None:
    a0 = {
        "results": [
            {"method": method, "label_fraction": 0.1, "macro_f1": value}
            for method, value in (
                ("rms", 0.1),
                ("raw", 0.2),
                ("handcrafted", 0.4),
                ("random", 0.25),
                ("moment", 0.35),
                ("patchtst", 0.3),
            )
        ]
    }

    def metadata(cell: str, domain_auc: float, recovery: float) -> dict:
        return {
            "cell": cell,
            "simulation_real_domain_probe": {"roc_auc": domain_auc},
            "development_physical_fidelity": {
                "retained_factor_linear_probes": {
                    "duration_ms": {
                        "relative_mse_reduction_vs_constant": recovery
                    }
                }
            },
        }

    checkpoints = {
        "checkpoint_metadata": {
            "a2": metadata("A2", 0.8, 0.2),
            "a3": metadata("A3", 0.9, 0.4),
            "a4": metadata("A4", 0.95, 0.5),
        }
    }
    comparisons = [
        {
            "left": "A4",
            "right": "handcrafted",
            "mean_difference": 0.01,
            "ci_95_low": -0.01,
        },
        {
            "left": "A4",
            "right": "A3",
            "mean_difference": 0.02,
            "ci_95_low": 0.001,
        },
    ]
    decision = _decision_summary(a0, checkpoints, comparisons)
    assert decision["strongest_eligible_frozen_baseline"] == "handcrafted"
    assert decision["promotion_decision"] == "do_not_promote_a4"
    assert decision["criteria"]["primary_effect_at_least_minimum"] is False
    assert decision["criteria"]["a4_vs_a3_interval_excludes_zero"] is True
    assert decision["criteria"]["adaptation_reduces_mean_domain_auc"] is False
