from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_yeast_latent_overlap as overlap


def test_validate_overlap_with_cached_embeddings(tmp_path: Path) -> None:
    model = "moment_official"
    rng = np.random.default_rng(7)
    real_root = tmp_path / "real"
    synth_root = tmp_path / "synthetic"
    output_dir = tmp_path / "out"
    (real_root / model).mkdir(parents=True)
    synth_root.mkdir()

    real_ids = [f"real_budding_{i}" for i in range(12)]
    control_ids = [f"real_mix_{i}" for i in range(12)]
    event_ids = np.asarray(real_ids + control_ids)
    real_emb = rng.normal(0, 0.35, size=(12, 6)).astype(np.float32)
    control_emb = rng.normal(4, 0.35, size=(12, 6)).astype(np.float32)
    np.savez_compressed(
        real_root / model / "all_embeddings.npz",
        embeddings=np.vstack([real_emb, control_emb]).astype(np.float32),
        labels=np.zeros(24, dtype=np.int64),
        split=np.asarray(["test"] * 24),
        event_id=event_ids,
    )
    pd.DataFrame(
        {
            "event_id": event_ids,
            "sample_id": event_ids,
            "split": "test",
            "class_name": "yeast",
        }
    ).to_csv(real_root / "events_metadata.csv", index=False)
    yeast_meta = tmp_path / "yeast_metadata.csv"
    pd.DataFrame(
        {
            "event_id": event_ids,
            "source_group": ["budding"] * 12 + ["mix"] * 12,
            "quality": "strict",
        }
    ).to_csv(yeast_meta, index=False)

    synth_emb = rng.normal(0.1, 0.35, size=(18, 6)).astype(np.float32)
    np.savez_compressed(
        synth_root / "synthetic_signals_encoded.npz",
        delta_t0_signals=rng.normal(size=(9, 4096)).astype(np.float32),
        amplitude_ratio_signals=rng.normal(size=(9, 4096)).astype(np.float32),
    )
    pd.DataFrame(
        {
            "scenario": "yeast_budded_two_particles",
            "panel": ["delta_t0"] * 9 + ["amplitude_ratio"] * 9,
            "index": np.arange(18),
            "sweep_param": ["delta_t0"] * 9 + ["amplitude_ratio"] * 9,
            "color_value": np.linspace(0, 1, 18),
        }
    ).to_csv(synth_root / "synthetic_metadata.csv", index=False)
    cache_dir = output_dir / "synthetic_embeddings" / model
    cache_dir.mkdir(parents=True)
    np.savez_compressed(cache_dir / "synthetic_embeddings.npz", embeddings=synth_emb, event_id=np.asarray([f"synth_{i}" for i in range(18)]))

    args = argparse.Namespace(
        output_dir=output_dir,
        models=model,
        control_synthetic_roots="",
        real_embedding_root=real_root,
        real_event_root=tmp_path / "missing_event_root",
        budding_raw_dir=tmp_path / "budding",
        yeast_metadata_csv=yeast_meta,
        synthetic_root=synth_root,
        source_group="budding",
        min_real=8,
        expected_input_length=4096,
        knn_k=3,
        parameter_bins=3,
        nearest_examples=4,
        seed=11,
        batch_size=8,
        device="cpu",
        cache_dir=tmp_path / "hf_cache",
        moment_model_id=overlap.MOMENT_DEFAULT_ID,
        patchtst_model_id=overlap.PATCHTST_DEFAULT_ID,
        conv1dgap_checkpoint=tmp_path / "missing.pt",
        force_encode_synthetic=False,
        force_encode_real=False,
        no_encode_synthetic=True,
        require_cuda=False,
        skip_tsne=True,
    )
    overlap.run(args)

    metrics = pd.read_csv(output_dir / "overlap_metrics.csv")
    assert {"real_budding_vs_template_budding_v1", "upper_bound_real_split", "lower_bound_real_vs_non_budding_yeast"} <= set(metrics["comparison"])
    assert (output_dir / "latent_overlap_pca_tsne.png").is_file()
    assert (output_dir / "parameter_overlap_by_bin.csv").is_file()
    assert (output_dir / "nearest_real_examples.csv").is_file()
    summary = json.loads((output_dir / "overlap_metrics.json").read_text())
    assert summary["config"]["source_group"] == "budding"
    assert summary["decision"]["decision"] in {"pilot_ready", "needs_range_restriction_or_generator_tuning"}


def test_falls_back_to_real_event_root_with_cached_real_event_embeddings(tmp_path: Path) -> None:
    model = "moment_official"
    rng = np.random.default_rng(13)
    tiny_real_root = tmp_path / "tiny_real"
    event_root = tmp_path / "event_root"
    synth_root = tmp_path / "synthetic"
    output_dir = tmp_path / "out"
    (tiny_real_root / model).mkdir(parents=True)
    event_root.mkdir()
    synth_root.mkdir()

    tiny_ids = np.asarray(["tiny_budding_0", "tiny_mix_0"])
    np.savez_compressed(
        tiny_real_root / model / "all_embeddings.npz",
        embeddings=rng.normal(size=(2, 4)).astype(np.float32),
        labels=np.zeros(2, dtype=np.int64),
        split=np.asarray(["test", "test"]),
        event_id=tiny_ids,
    )
    pd.DataFrame({"event_id": tiny_ids, "sample_id": tiny_ids, "split": "test", "class_name": "yeast"}).to_csv(
        tiny_real_root / "events_metadata.csv", index=False
    )
    yeast_meta = tmp_path / "yeast_meta.csv"
    pd.DataFrame({"event_id": tiny_ids, "source_group": ["budding", "mix"], "quality": "strict"}).to_csv(yeast_meta, index=False)

    event_ids = np.asarray([f"bud_{i}" for i in range(10)] + [f"mix_{i}" for i in range(10)])
    pd.DataFrame(
        {
            "event_id": event_ids,
            "sample_id": event_ids,
            "split": "test",
            "class_name": "yeast",
            "source_group": ["budding"] * 10 + ["mix"] * 10,
            "quality": "strict",
        }
    ).to_csv(event_root / "events_metadata.csv", index=False)
    np.savez_compressed(
        event_root / "aligned_inputs.npz",
        signals=rng.normal(size=(20, 4096)).astype(np.float32),
        labels=np.full(20, 3, dtype=np.int64),
        split=np.asarray(["test"] * 20),
        event_id=event_ids,
    )
    real_cache = output_dir / "real_embeddings" / model
    real_cache.mkdir(parents=True)
    np.savez_compressed(real_cache / "real_budding_embeddings.npz", embeddings=rng.normal(size=(10, 4)).astype(np.float32), input_length=np.asarray(4096))
    np.savez_compressed(real_cache / "non_budding_yeast_embeddings.npz", embeddings=rng.normal(4, 1, size=(10, 4)).astype(np.float32), input_length=np.asarray(4096))

    pd.DataFrame({"scenario": "x", "panel": ["delta_t0"] * 12, "index": np.arange(12), "color_value": np.linspace(0, 1, 12)}).to_csv(
        synth_root / "synthetic_metadata.csv", index=False
    )
    np.savez_compressed(synth_root / "synthetic_signals_encoded.npz", delta_t0_signals=rng.normal(size=(12, 4096)).astype(np.float32))
    synth_cache = output_dir / "synthetic_embeddings" / model
    synth_cache.mkdir(parents=True)
    np.savez_compressed(synth_cache / "synthetic_embeddings.npz", embeddings=rng.normal(size=(12, 4)).astype(np.float32), input_length=np.asarray(4096))

    args = argparse.Namespace(
        output_dir=output_dir,
        models=model,
        control_synthetic_roots="",
        real_embedding_root=tiny_real_root,
        real_event_root=event_root,
        budding_raw_dir=tmp_path / "budding",
        yeast_metadata_csv=yeast_meta,
        synthetic_root=synth_root,
        source_group="budding",
        min_real=8,
        expected_input_length=4096,
        knn_k=3,
        parameter_bins=3,
        nearest_examples=4,
        seed=11,
        batch_size=8,
        device="cpu",
        cache_dir=tmp_path / "hf_cache",
        moment_model_id=overlap.MOMENT_DEFAULT_ID,
        patchtst_model_id=overlap.PATCHTST_DEFAULT_ID,
        conv1dgap_checkpoint=tmp_path / "missing.pt",
        force_encode_synthetic=False,
        force_encode_real=False,
        no_encode_synthetic=True,
        require_cuda=False,
        skip_tsne=True,
    )
    overlap.run(args)
    summary = json.loads((output_dir / "overlap_metrics.json").read_text())
    assert summary["provenance"][f"{model}:real"]["source"] == "real_event_root"


def test_encode_signal_group_uses_mocked_model_stack(monkeypatch, tmp_path: Path) -> None:
    class FakeTorch:
        @staticmethod
        def device(value):
            return value

    class FakeBackbone:
        @staticmethod
        def encode_all_events(model_key, encoder, signals, batch_size, device):
            return signals[:, :3] + 1.0

    class FakeLatentSweeps:
        backbone = FakeBackbone()

        @staticmethod
        def load_encoder_for_model(model_key, args, device, model_dir):
            return object(), {"fake_model": True}

    monkeypatch.setattr(overlap, "import_encoding_stack", lambda: (FakeTorch, FakeLatentSweeps))
    signals = np.arange(20, dtype=np.float32).reshape(4, 5)
    metadata = pd.DataFrame({"event_id": [f"e{i}" for i in range(4)]})
    args = argparse.Namespace(
        force_encode_synthetic=False,
        force_encode_real=False,
        no_encode_synthetic=False,
        source_group="budding",
        moment_model_id=overlap.MOMENT_DEFAULT_ID,
        patchtst_model_id=overlap.PATCHTST_DEFAULT_ID,
        cache_dir=tmp_path / "cache",
        conv1dgap_checkpoint=tmp_path / "missing.pt",
        batch_size=2,
        device="cpu",
    )
    group = overlap.encode_signal_group(
        args,
        "moment_official",
        signals,
        metadata,
        tmp_path / "emb.npz",
        tmp_path / "model",
        "template_budding_v1",
        {"source": "synthetic_root"},
    )
    assert group.provenance["encoding_exercised"] is True
    assert group.embeddings.shape == (4, 3)
    assert (tmp_path / "emb.npz").is_file()


def test_rejects_input_length_mismatch() -> None:
    real = overlap.EmbeddingGroup(
        "real_budding",
        np.zeros((4, 3), dtype=np.float32),
        pd.DataFrame({"event_id": [f"r{i}" for i in range(4)]}),
        signals=np.zeros((4, 512), dtype=np.float32),
    )
    synth = overlap.EmbeddingGroup(
        "template_budding_v1",
        np.zeros((4, 3), dtype=np.float32),
        pd.DataFrame({"event_id": [f"s{i}" for i in range(4)]}),
        signals=np.zeros((4, 4096), dtype=np.float32),
    )
    try:
        overlap.validate_matching_input_lengths(real, synth)
    except ValueError as exc:
        assert "512-sample artifacts" in str(exc)
    else:
        raise AssertionError("Expected input length mismatch to raise")


def test_rejects_p1_mode_with_raw_embedding_cache(tmp_path: Path) -> None:
    args = argparse.Namespace(
        preprocess_mode="p1_bandpass_saturation",
        preprocess_sampling_frequency_hz=2_000_000.0,
        preprocess_low_khz=5.0,
        preprocess_high_khz_max=100.0,
        saturation_fmin_hz=7_000.0,
        saturation_fmax_hz=80_000.0,
        saturation_min_flat=500,
        saturation_zero_threshold=1.0e-4,
        saturation_guard_before=0,
        saturation_guard_after=0,
    )
    cache_path = tmp_path / "cache.npz"
    np.savez_compressed(cache_path, embeddings=np.zeros((2, 3), dtype=np.float32), input_length=np.asarray(4096))

    with np.load(cache_path) as data:
        try:
            overlap.validate_cache_preprocessing(cache_path, data, args)
        except ValueError as exc:
            assert "preprocessing_id" in str(exc)
        else:
            raise AssertionError("Expected raw cache to be rejected in p1 preprocessing mode")


def test_p1_validation_skips_already_preprocessed_inputs() -> None:
    args = argparse.Namespace(
        preprocess_mode="p1_bandpass_saturation",
        preprocess_sampling_frequency_hz=2_000_000.0,
        preprocess_low_khz=5.0,
        preprocess_high_khz_max=100.0,
        saturation_fmin_hz=7_000.0,
        saturation_fmax_hz=80_000.0,
        saturation_min_flat=500,
        saturation_zero_threshold=1.0e-4,
        saturation_guard_before=0,
        saturation_guard_after=0,
    )
    cfg = overlap.preprocessing_config_from_args(args)
    signals = np.random.default_rng(2).normal(size=(3, 16)).astype(np.float32)
    meta = pd.DataFrame({"event_id": ["a", "b", "c"], "__input_preprocessing_id": cfg.preprocessing_id})

    out_signals, out_meta, summary = overlap.maybe_preprocess_signals(signals, meta, args, reject_saturation=True)

    np.testing.assert_array_equal(out_signals, signals)
    assert out_meta["event_id"].tolist() == ["a", "b", "c"]
    assert summary["already_applied"] is True


def test_p1_validation_rejects_unprovenanced_inputs() -> None:
    args = argparse.Namespace(
        preprocess_mode="p1_bandpass_saturation",
        preprocess_sampling_frequency_hz=2_000_000.0,
        preprocess_low_khz=5.0,
        preprocess_high_khz_max=100.0,
        saturation_fmin_hz=7_000.0,
        saturation_fmax_hz=80_000.0,
        saturation_min_flat=500,
        saturation_zero_threshold=1.0e-4,
        saturation_guard_before=0,
        saturation_guard_after=0,
    )

    try:
        overlap.maybe_preprocess_signals(
            np.random.default_rng(3).normal(size=(3, 16)).astype(np.float32),
            pd.DataFrame({"event_id": ["a", "b", "c"]}),
            args,
            reject_saturation=True,
        )
    except ValueError as exc:
        assert "matching preprocessing_id" in str(exc)
    else:
        raise AssertionError("Expected p1 validation to reject inputs without preprocessing provenance")
