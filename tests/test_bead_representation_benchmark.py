from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from p3_ssl.bead_representation_benchmark import (
    BenchmarkPopulation,
    _simulation_estimator,
    average_simulation_views,
    nested_real_subsets,
    nested_simulation_subsets,
    nominal_ssl_budget,
    label_efficiency_auc,
    paired_grouped_classification_interval,
    paired_hierarchical_interval,
    verify_nested,
)


def test_simulation_subsets_are_nested_and_view_average_uses_latent() -> None:
    targets = np.asarray(
        [[0.2 + index * 0.01, 10.0 + index] for index in range(100)],
        dtype=np.float64,
    )
    subsets = nested_simulation_subsets(targets, seed=7)
    verify_nested(subsets)
    assert len(subsets[0.01]) == 1
    assert len(subsets[0.10]) == 10
    assert len(subsets[1.0]) == 100

    population = BenchmarkPopulation(
        signals=np.zeros((4, 8), dtype=np.float32),
        ids=np.asarray(["a:0", "a:1", "b:0", "b:1"]),
        groups=np.asarray(["a", "a", "b", "b"]),
        labels=np.asarray([[1, 2], [1, 2], [3, 4], [3, 4]], dtype=float),
        metadata=tuple({} for _ in range(4)),
    )
    values, averaged_targets, ids = average_simulation_views(
        population,
        np.asarray([[0, 2], [2, 4], [10, 12], [12, 14]], dtype=float),
    )
    assert ids.tolist() == ["a", "b"]
    assert values.tolist() == [[1.0, 3.0], [11.0, 13.0]]
    assert averaged_targets.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_real_quarter_folds_are_nested_group_disjoint_and_three_class() -> None:
    labels = np.repeat(np.arange(3), 40)
    groups = np.asarray([f"{label}-{index}" for label in range(3) for index in range(40)])
    subsets = nested_real_subsets(labels, groups, seed=3)
    verify_nested(subsets)
    assert set(labels[subsets[0.25]]) == {0, 1, 2}
    assert len(subsets[1.0]) == len(labels)


def test_paired_interval_keeps_five_training_pairs_visible() -> None:
    rows = []
    for representation_seed in range(42, 47):
        for subset_seed in range(3):
            rows.extend(
                [
                    {
                        "method": "P25",
                        "representation_seed": representation_seed,
                        "subset_seed": subset_seed,
                        "fraction": 0.1,
                        "score": 0.2,
                    },
                    {
                        "method": "CYCLIC25",
                        "representation_seed": representation_seed,
                        "subset_seed": subset_seed,
                        "fraction": 0.1,
                        "score": 0.3,
                    },
                ]
            )
    result = paired_hierarchical_interval(
        rows, metric="score", repeats=100, seed=9
    )
    assert result["n_independent_training_pairs"] == 5
    assert result["mean_difference"] == pytest.approx(0.1)
    assert len(result["differences_by_representation_seed"]) == 5


def test_nominal_budget_is_equalized() -> None:
    budget = nominal_ssl_budget()
    assert budget["optimizer_updates"] == 4380
    assert budget["signals_seen"] == 139640
    assert budget["masked_values_contributing_to_loss"] == 142991360


def test_auc_reuses_single_full_label_endpoint_across_subset_seeds() -> None:
    rows = [
        {
            "method": "P25",
            "representation_seed": 42,
            "subset_seed": 0,
            "fraction": 1.0,
            "mean_r2": 0.8,
        }
    ]
    for subset_seed in (0, 1):
        rows.extend(
            {
                "method": "P25",
                "representation_seed": 42,
                "subset_seed": subset_seed,
                "fraction": fraction,
                "mean_r2": score,
            }
            for fraction, score in ((0.01, 0.1), (0.1, 0.4))
        )
    auc = label_efficiency_auc(rows, score_key="mean_r2")
    assert len(auc) == 2


def test_real_interval_resamples_source_groups_and_preserves_pairing() -> None:
    rows = []
    for representation_seed in range(42, 47):
        for event_index, (group, target) in enumerate(
            (("file-a", 0), ("file-a", 1), ("file-b", 2))
        ):
            rows.extend(
                [
                    {
                        "method": "P25",
                        "representation_seed": representation_seed,
                        "subset_seed": 0,
                        "fraction": 1.0,
                        "event_id": f"event-{event_index}",
                        "source_group": group,
                        "target": target,
                        "prediction": 0,
                    },
                    {
                        "method": "CYCLIC25",
                        "representation_seed": representation_seed,
                        "subset_seed": 0,
                        "fraction": 1.0,
                        "event_id": f"event-{event_index}",
                        "source_group": group,
                        "target": target,
                        "prediction": target,
                    },
                ]
            )
    result = paired_grouped_classification_interval(
        rows, repeats=100, seed=8
    )
    assert result["n_independent_training_pairs"] == 5
    assert result["cluster_unit"] == "source_group"
    assert result["mean_difference"] > 0
    assert len(result["differences_by_representation_seed"]) == 5


def test_raw_pca_dimension_fits_smallest_inner_fold() -> None:
    rng = np.random.default_rng(11)
    estimator = _simulation_estimator("raw_pca", seed=3, n_rows=35)
    estimator.fit(rng.normal(size=(35, 128)), rng.normal(size=(35, 2)))
    assert estimator.best_estimator_["pca"].n_components == 28
