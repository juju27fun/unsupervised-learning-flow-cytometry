from __future__ import annotations

import pytest

from p3_ssl.study_reporting import paired_bootstrap_comparisons


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
