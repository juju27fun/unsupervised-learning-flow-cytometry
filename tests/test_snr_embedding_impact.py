from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from p3_ssl.snr_embedding_impact import (
    class_enrichment,
    impurity_from_neighbors,
    precompute_neighbor_indices,
    process_dataset,
    split_for_probe,
    top1_confusion_from_neighbors,
)
from p3_ssl.snr_threshold_manifolds import DatasetSpec


def test_class_enrichment_reports_low_snr_overrepresentation() -> None:
    labels = np.asarray([0, 0, 1, 2], dtype=np.int64)
    low = np.asarray([True, True, False, False])

    result = class_enrichment(labels, low, ("a", "b", "c"))

    assert result["a_low_snr_n"] == 2
    assert result["a_baseline_fraction"] == 0.5
    assert result["a_low_snr_fraction_of_low"] == 1.0
    assert result["a_low_snr_enrichment"] == 2.0


def test_precomputed_neighbors_support_knn_impurity() -> None:
    x_norm = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.02, 0.98],
        ],
        dtype=np.float32,
    )
    x_norm = x_norm / np.linalg.norm(x_norm, axis=1, keepdims=True)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    nn = precompute_neighbor_indices(x_norm, k=1)
    impurity = impurity_from_neighbors(labels, nn, np.ones(labels.shape[0], dtype=bool))

    assert nn.shape == (4, 1)
    assert impurity == 0.0


def test_top1_confusion_from_neighbors_counts_directional_errors() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    neighbor_idx = np.asarray([[2], [0], [3], [2]], dtype=np.int64)

    matrix, same_rate = top1_confusion_from_neighbors(
        labels,
        neighbor_idx,
        np.ones(labels.shape[0], dtype=bool),
        [0, 1],
    )

    np.testing.assert_array_equal(matrix, np.asarray([[1, 1], [0, 2]], dtype=np.int64))
    assert same_rate == 0.75


def test_split_for_probe_uses_existing_split_when_valid() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    split = np.asarray(["train", "train", "test", "train", "train", "test"])

    train_idx, test_idx, source = split_for_probe(labels, split, seed=0)

    assert source == "existing_split"
    np.testing.assert_array_equal(train_idx, np.asarray([0, 1, 3, 4]))
    np.testing.assert_array_equal(test_idx, np.asarray([2, 5]))


def test_process_dataset_writes_metric_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    labels = np.asarray([0] * 8 + [1] * 8 + [2] * 8, dtype=np.int64)
    event_id = np.asarray([f"event_{i:02d}" for i in range(labels.size)])
    split = np.asarray((["train"] * 5 + ["test"] * 3) * 3)
    centers = np.eye(3, dtype=np.float32)[labels]
    embeddings = centers + rng.normal(0.0, 0.03, size=centers.shape).astype(np.float32)
    root = tmp_path / "embedding_root"
    model_dir = root / "moment_official"
    model_dir.mkdir(parents=True)
    np.savez_compressed(
        model_dir / "all_embeddings.npz",
        embeddings=embeddings,
        labels=labels,
        split=split,
        event_id=event_id,
    )
    metadata = pd.DataFrame(
        {
            "event_id": event_id,
            "plot_class_id": labels,
            "plot_class_name": np.asarray([("a", "b", "c")[int(v)] for v in labels]),
            "snr_value": np.linspace(0.0, 1.0, labels.size),
        }
    )
    dataset = DatasetSpec(
        key="toy",
        display_name="Toy",
        snr_column="snr",
        snr_label="SNR",
        class_names=("a", "b", "c"),
        embedding_root=root,
        metadata=metadata,
    )

    summary = process_dataset(
        dataset=dataset,
        model_keys=["moment_official"],
        output_dir=tmp_path / "out",
        max_events_per_class=4,
        k=2,
        seed=4,
    )

    dataset_dir = tmp_path / "out" / "toy"
    assert summary["n_rows"] == 32
    assert (dataset_dir / "snr_embedding_impact_metrics.csv").is_file()
    assert (dataset_dir / "snr_embedding_impact_metrics.json").is_file()
    assert (dataset_dir / "full_dataset_snr_impact_curves.pdf").is_file()
    assert (dataset_dir / "visual_subset_snr_impact_curves.pdf").is_file()
