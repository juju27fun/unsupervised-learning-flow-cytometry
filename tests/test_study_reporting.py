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
