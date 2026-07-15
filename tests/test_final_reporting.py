from __future__ import annotations

import json

from p3_ssl.final_reporting import DISPLAY
from p3_ssl.final_reporting import build_final_report


def test_final_reporting_covers_every_confirmatory_method() -> None:
    assert set(DISPLAY) == {"handcrafted", "A3", "A4"}


def test_final_reporting_records_negative_primary_result(tmp_path) -> None:
    class_names = ["budding", "mix", "shmoo", "shmoo2"]
    results = []
    for method, values in {
        "handcrafted": [0.44, 0.42, 0.43],
        "A3": [0.31, 0.33, 0.32],
        "A4": [0.34, 0.36, 0.35],
    }.items():
        for seed, value in enumerate(values):
            results.append(
                {
                    "method": method,
                    "macro_f1": value,
                    "per_class_recall": {name: value for name in class_names},
                    "probe_seed": seed,
                }
            )
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "sealed_splits_used": ["in_session_test"],
                "methods": ["handcrafted", "A3", "A4"],
                "results": results,
                "class_names": class_names,
                "n_test_events": 12,
                "n_test_capture_blocks": 4,
                "paired_comparisons": [
                    {
                        "comparison": "A4 - handcrafted",
                        "mean_difference": -0.08,
                        "ci_95_low": -0.10,
                        "ci_95_high": -0.05,
                    },
                    {
                        "comparison": "A4 - A3",
                        "mean_difference": 0.03,
                        "ci_95_low": 0.01,
                        "ci_95_high": 0.04,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_final_report(metrics, tmp_path / "report")

    assert summary["decision"]["primary_interval_position"] == "entirely_below_zero"
    assert summary["decision"]["a4_is_significantly_worse_than_handcrafted"] is True
    assert summary["decision"]["a4_improves_over_a3"] is True
    assert len(summary["outputs"]) == 9
    assert all((tmp_path / "report" / name).is_file() for name in summary["outputs"])
