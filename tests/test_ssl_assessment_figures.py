from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.run_ssl_assessment_figures import (
    balanced_indices,
    load_aligned_inputs,
    load_embedding_bundles,
    perturb_signal_batch,
    plot_manifold_figure,
    plot_retrieval_sheet,
    run_label_efficiency,
    summarize_probe_rows,
    write_assessment_dashboard,
)


def _write_bundle(root: Path, key: str, embeddings: np.ndarray, labels: np.ndarray, split: np.ndarray, event_id: np.ndarray) -> None:
    model_dir = root / key
    model_dir.mkdir(parents=True)
    np.savez_compressed(
        model_dir / "all_embeddings.npz",
        embeddings=embeddings.astype(np.float32),
        labels=labels.astype(np.int64),
        split=split,
        event_id=event_id,
    )


def _toy_embedding_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    root = tmp_path / "embeddings"
    labels = np.asarray([0] * 8 + [1] * 8 + [2] * 8, dtype=np.int64)
    split = np.asarray((["train"] * 4 + ["val"] * 2 + ["test"] * 2) * 3)
    event_id = np.asarray([f"event_{i:02d}" for i in range(labels.size)])
    centers = np.stack(
        [
            np.eye(3, dtype=np.float32)[labels],
            np.eye(3, dtype=np.float32)[labels] * 0.5,
        ],
        axis=-1,
    ).reshape(labels.size, 6)
    emb_a = centers + rng.normal(0.0, 0.05, size=centers.shape).astype(np.float32)
    emb_b = rng.normal(0.0, 1.0, size=(labels.size, 5)).astype(np.float32)
    _write_bundle(root, "moment_official", emb_a, labels, split, event_id)
    _write_bundle(root, "patchtst_pretrained", emb_b, labels, split, event_id)
    signals = rng.normal(0.0, 1.0, size=(labels.size, 32)).astype(np.float32)
    np.savez_compressed(root / "aligned_512_inputs.npz", signals=signals, labels=labels, split=split)
    with (root / "events_metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["event_id", "sample_id"])
        writer.writeheader()
        for event in event_id:
            writer.writerow({"event_id": event, "sample_id": event})
    return root


def test_load_embedding_bundles_adds_raw_and_random_baselines(tmp_path: Path) -> None:
    root = _toy_embedding_root(tmp_path)
    aligned = load_aligned_inputs(root / "aligned_512_inputs.npz")
    bundles = load_embedding_bundles(
        embedding_root=root,
        model_keys=["moment_official"],
        aligned_inputs=aligned,
        include_raw_baseline=True,
        include_random_baseline=True,
        random_projection_dim=7,
        seed=1,
    )

    assert [bundle.key for bundle in bundles] == ["moment_official", "raw_signal", "random_projection"]
    assert bundles[1].embeddings.shape == (24, 32)
    assert bundles[2].embeddings.shape == (24, 7)
    assert bundles[1].class_name.tolist()[:3] == ["2um", "2um", "2um"]


def test_load_embedding_bundles_rejects_mismatched_event_order(tmp_path: Path) -> None:
    root = _toy_embedding_root(tmp_path)
    with np.load(root / "patchtst_pretrained" / "all_embeddings.npz", allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["event_id"] = payload["event_id"][::-1]
    np.savez_compressed(root / "patchtst_pretrained" / "all_embeddings.npz", **payload)

    with pytest.raises(ValueError, match="Event order mismatch"):
        load_embedding_bundles(
            embedding_root=root,
            model_keys=["moment_official", "patchtst_pretrained"],
            aligned_inputs=None,
            include_raw_baseline=False,
            include_random_baseline=False,
            random_projection_dim=4,
            seed=2,
        )



def test_load_embedding_bundles_supports_common_alias_layouts(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    split = np.asarray(["train", "train", "train", "test", "test", "test"])
    event_id = np.asarray([f"event_{i}" for i in range(labels.size)])
    class_name = np.asarray(["2um", "4um", "10um", "2um", "4um", "10um"])

    patch_dir = root / "patchtst_pretrained" / "full_finetune"
    patch_dir.mkdir(parents=True)
    np.savez_compressed(
        patch_dir / "embeddings.npz",
        embeddings=np.eye(labels.size, dtype=np.float32),
        labels=labels,
        split=split,
        event_id=event_id,
        class_name=class_name,
    )
    p3_dir = root / "embedding_space"
    p3_dir.mkdir(parents=True)
    np.savez_compressed(
        p3_dir / "embeddings_all.npz",
        moment_embeddings=np.ones((labels.size, 4), dtype=np.float32),
        class_id=labels,
        split=split,
        event_id=event_id,
        class_name=class_name,
    )

    bundles = load_embedding_bundles(
        embedding_root=root,
        model_keys=["patchtst_pretrained_full", "p3_ssl_moment_like"],
        aligned_inputs=None,
        include_raw_baseline=False,
        include_random_baseline=False,
        random_projection_dim=4,
        seed=2,
    )

    assert [bundle.key for bundle in bundles] == ["patchtst_pretrained_full", "p3_ssl_moment_like"]
    assert bundles[0].display_name == "PatchTST full fine-tuned"
    assert bundles[1].embeddings.shape == (6, 4)

def test_balanced_indices_caps_per_class() -> None:
    labels = np.asarray([0] * 5 + [1] * 4 + [2] * 3)
    idx = balanced_indices(labels, max_per_class=2, seed=0)

    assert idx.shape == (6,)
    assert {int(c): int((labels[idx] == c).sum()) for c in set(labels.tolist())} == {0: 2, 1: 2, 2: 2}


def test_assessment_outputs_write_expected_files(tmp_path: Path) -> None:
    root = _toy_embedding_root(tmp_path)
    aligned = load_aligned_inputs(root / "aligned_512_inputs.npz")
    bundles = load_embedding_bundles(
        embedding_root=root,
        model_keys=["moment_official", "patchtst_pretrained"],
        aligned_inputs=aligned,
        include_raw_baseline=False,
        include_random_baseline=False,
        random_projection_dim=4,
        seed=2,
    )
    output_dir = tmp_path / "assessment"

    manifold = plot_manifold_figure(bundles, output_dir=output_dir, max_events_per_class=4, seed=2, run_tsne=False)
    rows = run_label_efficiency(bundles, output_dir=output_dir, fractions=[0.5], repeats=1, seed=2)
    retrieval = plot_retrieval_sheet(
        bundles,
        output_dir=output_dir,
        signals=aligned["signals"],
        metadata={},
        queries_per_class=1,
        neighbors=2,
        metric_max_per_class=4,
        seed=2,
    )

    dashboard = write_assessment_dashboard(
        output_dir,
        bundles=bundles,
        manifold=manifold,
        label_summary=summarize_probe_rows(rows),
        retrieval=retrieval,
        robustness={"status": "not_run"},
    )

    assert manifold["n_events_plotted"] == 12
    assert manifold["run_tsne"] is False
    with np.load(output_dir / "representation_manifold_reductions.npz", allow_pickle=True) as reductions:
        assert "moment_official_pca" in reductions.files
        assert "moment_official_tsne" not in reductions.files
    assert len(rows) == 2
    assert set(retrieval) == {"moment_official", "patchtst_pretrained"}
    assert [row["model"] for row in dashboard] == ["moment_official", "patchtst_pretrained"]
    for name in [
        "representation_manifold.pdf",
        "label_efficiency_curve.pdf",
        "retrieval_purity.pdf",
        "nearest_neighbor_retrieval_sheet.pdf",
        "assessment_dashboard.pdf",
        "assessment_dashboard.png",
        "assessment_dashboard.json",
    ]:
        assert (output_dir / name).is_file()
        assert (output_dir / name).stat().st_size > 0


def test_perturb_signal_batch_changes_expected_properties() -> None:
    signals = np.tile(np.linspace(-1.0, 1.0, 32, dtype=np.float32), (3, 1))

    shifted = perturb_signal_batch(signals, "shift_8", seed=0)
    scaled = perturb_signal_batch(signals, "scale_1p25", seed=0)
    masked = perturb_signal_batch(signals, "center_mask_64", seed=0)

    np.testing.assert_allclose(shifted, np.roll(signals, 8, axis=1))
    np.testing.assert_allclose(scaled, signals * 1.25)
    assert np.count_nonzero(masked) == 0
