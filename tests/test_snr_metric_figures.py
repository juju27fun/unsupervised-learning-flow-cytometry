from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from p3_ssl.snr_metric_figures import (
    compute_low_snr_confused_masks,
    most_enriched_class,
    nearest_metric_row,
    parse_quantiles,
    process_dataset,
    quantile_tag,
)
from p3_ssl.snr_threshold_manifolds import DatasetSpec


def test_parse_quantiles_and_tag() -> None:
    assert parse_quantiles("0.20,0.5,0.80") == [0.2, 0.5, 0.8]
    assert quantile_tag(0.2) == "q20"
    assert quantile_tag(0.05) == "q05"


def test_nearest_metric_row_and_most_enriched_class() -> None:
    df = pd.DataFrame(
        [
            {
                "mode": "visual_subset",
                "model": "m",
                "quantile": 0.2,
                "a_low_snr_enrichment": 0.9,
                "b_low_snr_enrichment": 1.4,
                "c_low_snr_enrichment": 1.1,
            }
        ]
    )

    row = nearest_metric_row(df, "visual_subset", "m", 0.2)
    assert most_enriched_class(row, ("a", "b", "c")) == ("b", 1.4)


def _write_model(root: Path, event_id: np.ndarray, labels: np.ndarray, embeddings: np.ndarray) -> None:
    model_dir = root / "moment_official"
    model_dir.mkdir(parents=True)
    np.savez_compressed(
        model_dir / "all_embeddings.npz",
        embeddings=embeddings.astype(np.float32),
        labels=labels.astype(np.int64),
        split=np.asarray(["train", "train", "test", "test", "train", "test"]),
        event_id=event_id,
    )


def _toy_dataset(tmp_path: Path) -> DatasetSpec:
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    event_id = np.asarray([f"event_{i}" for i in range(labels.size)])
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.95, 0.05],
            [0.0, 0.9],
            [-1.0, 0.0],
            [-0.9, 0.0],
        ],
        dtype=np.float32,
    )
    root = tmp_path / "embeddings"
    _write_model(root, event_id, labels, embeddings)
    metadata = pd.DataFrame(
        {
            "event_id": event_id,
            "plot_class_id": labels,
            "plot_class_name": ["a", "a", "b", "b", "c", "c"],
            "snr_value": np.asarray([0.0, 0.1, 0.2, 0.7, 0.8, 0.9], dtype=np.float32),
        }
    )
    return DatasetSpec(
        key="toy",
        display_name="Toy",
        snr_column="snr",
        snr_label="SNR",
        class_names=("a", "b", "c"),
        embedding_root=root,
        metadata=metadata,
    )


def test_compute_low_snr_confused_masks_marks_cross_class_top1(tmp_path: Path) -> None:
    dataset = _toy_dataset(tmp_path)
    reductions = {
        "selected_index": np.arange(6, dtype=np.int64),
        "event_id": dataset.metadata["event_id"].astype(str).to_numpy(),
        "labels": dataset.metadata["plot_class_id"].to_numpy(dtype=np.int64),
        "snr_value": dataset.metadata["snr_value"].to_numpy(dtype=np.float32),
    }

    masks = compute_low_snr_confused_masks(dataset, reductions, ["moment_official"], [0.5])
    low = masks["moment_official"][0.5]["low"]
    confused = masks["moment_official"][0.5]["confused"]

    assert low.tolist() == [True, True, True, False, False, False]
    assert confused.tolist() == [True, True, True, False, False, False]


def _write_metric_inputs(tmp_path: Path, dataset: DatasetSpec) -> Namespace:
    metrics_root = tmp_path / "metrics"
    manifold_root = tmp_path / "manifolds"
    dataset_metrics = metrics_root / dataset.key
    dataset_manifold = manifold_root / dataset.key
    dataset_metrics.mkdir(parents=True)
    dataset_manifold.mkdir(parents=True)
    rows = []
    for mode in ["visual_subset"]:
        rows.append(
            {
                "dataset": dataset.key,
                "mode": mode,
                "model": "moment_official",
                "display_name": "MOMENT frozen pretrained",
                "quantile": 0.2,
                "threshold": 0.1,
                "snr_label": "SNR",
                "n_total": 6,
                "n_low_snr": 2,
                "knn_impurity_delta": 0.2,
                "probe_error_lift": 0.1,
                "top1_same_class_delta": -0.2,
                "a_low_snr_enrichment": 1.5,
                "b_low_snr_enrichment": 0.8,
                "c_low_snr_enrichment": 0.7,
            }
        )
    pd.DataFrame(rows).to_csv(dataset_metrics / "snr_embedding_impact_metrics.csv", index=False)
    details = {
        "models": {
            "moment_official": {
                "visual_subset": {
                    "thresholds": {
                        "q20": {
                            "low_top1_retrieval_matrix_row_normalized": [
                                [0.8, 0.2, 0.0],
                                [0.1, 0.9, 0.0],
                                [0.0, 0.3, 0.7],
                            ]
                        }
                    }
                }
            }
        }
    }
    (dataset_metrics / "snr_embedding_impact_metrics.json").write_text(json.dumps(details))
    np.savez_compressed(
        dataset_manifold / "fixed_reductions.npz",
        selected_index=np.arange(6, dtype=np.int64),
        event_id=dataset.metadata["event_id"].astype(str).to_numpy(),
        labels=dataset.metadata["plot_class_id"].to_numpy(dtype=np.int64),
        snr_value=dataset.metadata["snr_value"].to_numpy(dtype=np.float32),
        moment_official_pca=np.column_stack([np.arange(6), np.arange(6)]).astype(np.float32),
        moment_official_tsne=np.column_stack([np.arange(6), np.arange(6)[::-1]]).astype(np.float32),
    )
    return Namespace(
        output_dir=tmp_path / "out",
        metrics_root=metrics_root,
        manifold_root=manifold_root,
        overlay_mode="visual_subset",
    )


def test_process_dataset_writes_metric_figures(tmp_path: Path) -> None:
    dataset = _toy_dataset(tmp_path)
    args = _write_metric_inputs(tmp_path, dataset)

    summary = process_dataset(dataset, args, ["moment_official"], ["visual_subset"], [0.2])

    out = Path(summary["output_dir"])
    assert (out / "toy_q20_visual_subset_enriched_pca_tsne.pdf").is_file()
    assert (out / "toy_visual_subset_snr_impact_curves.pdf").is_file()
    assert (out / "toy_visual_subset_class_enrichment_heatmap.pdf").is_file()
    assert (out / "toy_visual_subset_top1_confusion_summary.pdf").is_file()
    assert (out / "toy_compact_metric_table.csv").is_file()
