from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p3_ssl.snr_threshold_manifolds import (
    EmbeddingBundle,
    align_embeddings_to_metadata,
    build_particle_metadata,
    build_yeast_metadata,
    quantile_grid,
    quantile_thresholds,
    threshold_summary_rows,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def test_quantile_grid_has_16_thresholds() -> None:
    grid = quantile_grid()

    assert grid.shape == (16,)
    assert np.isclose(grid[0], 0.05)
    assert np.isclose(grid[-1], 0.80)
    np.testing.assert_allclose(np.diff(grid), np.full(15, 0.05), atol=1.0e-6)


def test_threshold_summary_low_counts_are_monotonic() -> None:
    snr = np.arange(60, dtype=np.float32)
    labels = np.asarray([0, 1, 2] * 20, dtype=np.int64)
    thresholds = quantile_thresholds(snr)
    rows = threshold_summary_rows("synthetic", snr, labels, ("a", "b", "c"), thresholds)

    low_counts = [int(row["n_low_snr"]) for row in rows]
    assert low_counts == sorted(low_counts)
    assert low_counts[0] > 0
    assert low_counts[-1] < snr.size


def test_align_embeddings_to_metadata_uses_event_ids() -> None:
    bundle = EmbeddingBundle(
        model_key="m",
        embeddings=np.asarray([[10, 11], [20, 21], [30, 31]], dtype=np.float32),
        event_id=np.asarray(["b", "a", "c"]),
    )
    metadata = pd.DataFrame({"event_id": ["a", "b", "c"]})

    aligned = align_embeddings_to_metadata(bundle, metadata)

    np.testing.assert_array_equal(aligned, np.asarray([[20, 21], [10, 11], [30, 31]], dtype=np.float32))


def test_real_particle_snr_join_is_complete_when_artifacts_exist() -> None:
    particle_root = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones" / "particles2snr_f_3class_native_params_moment_patchtst_conv1dgap"
    manifest = REPO_ROOT / "artifacts" / "particles2SNR-pipeline" / "runs" / "p0_c1_Particles2SNR_F" / "event_classification_dataset" / "event_manifest.csv"
    if not particle_root.exists() or not manifest.exists():
        pytest.skip("Particles2SNR_F artifacts are not available")

    metadata = build_particle_metadata(particle_root, manifest)

    assert metadata.shape[0] == 4690
    assert metadata["snr_value"].notna().all()
    assert set(metadata["plot_class_name"].unique().tolist()) == {"2um", "4um", "10um"}


def test_real_yeast_metadata_uses_requested_three_classes_when_available() -> None:
    yeast_root = ROOT.parent / "artifacts" / "unsupervised-learning-flow-cytometry" / "pretrained_backbones" / "yeast_passage_events_p3_4096"
    if not yeast_root.exists():
        pytest.skip("Yeast event artifacts are not available")

    metadata = build_yeast_metadata(yeast_root, ("mix", "budding", "shmoo2"))

    assert set(metadata["plot_class_name"].unique().tolist()) == {"mix", "budding", "shmoo2"}
    assert "shmoo" not in set(metadata["plot_class_name"].unique().tolist())
    assert metadata["snr_value"].notna().all()
    assert metadata.shape[0] > 0
