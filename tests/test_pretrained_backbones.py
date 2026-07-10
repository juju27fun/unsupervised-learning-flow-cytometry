from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from p3_ssl.pretrained_backbones import (
    ParticleEvent,
    collect_particle_events,
    signal_to_spectrogram_image,
)
from p3_ssl.aligned_backbones import (
    build_aligned_signal,
    build_aligned_512_signal,
    materialize_conv_train_views,
    validate_no_test_leakage,
)
from p3_ssl.backbone_benchmark import (
    MOMENT_OFFICIAL_PATCH_LEN,
    MOMENT_OFFICIAL_PATCH_STRIDE,
    PATCHTST_PRETRAIN_CONTEXT_LENGTH,
    PATCHTST_PRETRAIN_PATCH_LENGTH,
    PATCHTST_PRETRAIN_PATCH_STRIDE,
    adaptive_bandpass_decimate_np,
    encode_conv1dgap_features,
    filter_and_remap_classes,
    patchtst_native_metadata,
    plot_pretrained_model_comparison,
)



def test_filter_and_remap_classes_keeps_requested_order(tmp_path: Path) -> None:
    signal_path = tmp_path / "event.npy"
    np.save(signal_path, np.zeros(16, dtype=np.float32))
    events = [
        ParticleEvent(
            event_id=f"event_{name}",
            sample_id=f"event_{name}",
            split="train",
            signal_path=str(signal_path),
            label_path="",
            class_id=old_id,
            class_name=name,
            center_norm=0.5,
            width_norm=0.0,
            center_index=8,
            crop_start=0,
            crop_end=16,
        )
        for old_id, name in [(3, "unclear"), (2, "10um"), (0, "2um"), (1, "4um")]
    ]
    signals = np.arange(16, dtype=np.float32).reshape(4, 4)

    filtered, filtered_signals = filter_and_remap_classes(events, signals, ["2um", "4um", "10um"])

    assert [event.class_name for event in filtered] == ["10um", "2um", "4um"]
    assert [event.class_id for event in filtered] == [2, 0, 1]
    np.testing.assert_array_equal(filtered_signals, signals[[1, 2, 3]])

def test_signal_to_spectrogram_image_shape_and_finiteness() -> None:
    x = np.sin(np.linspace(0, 12 * np.pi, 512, dtype=np.float32))
    image = signal_to_spectrogram_image(x)
    assert tuple(image.shape) == (3, 224, 224)
    assert torch.isfinite(image).all()
    assert float(image.std()) > 0.0


def test_collect_particle_events_center_crop(tmp_path: Path) -> None:
    signal_path = tmp_path / "signal.npy"
    label_path = tmp_path / "label.txt"
    manifest_path = tmp_path / "manifest.csv"
    np.save(signal_path, np.arange(16, dtype=np.float32))
    label_path.write_text("2 0.5 0.25\n")
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "id", "signal_path", "label_path"])
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "id": "sample",
                "signal_path": str(signal_path),
                "label_path": str(label_path),
            }
        )

    events, crops = collect_particle_events(
        manifest_csv=manifest_path,
        input_length_raw=16,
        decimation_factor=1,
        input_length_ssl=16,
        event_length=8,
        normalization="none",
    )
    assert len(events) == 1
    assert events[0].class_id == 2
    assert events[0].center_index == 8
    assert crops.shape == (1, 8)
    np.testing.assert_array_equal(crops[0], np.arange(4, 12, dtype=np.float32))


def test_plot_pretrained_model_comparison_writes_outputs(tmp_path: Path) -> None:
    labels = np.asarray([0, 1, 2, 3] * 3, dtype=np.int64)
    pca = np.column_stack((np.arange(labels.size), labels)).astype(np.float32)
    tsne = np.column_stack((labels, np.arange(labels.size))).astype(np.float32)
    model_dirs: dict[str, Path] = {}
    for model_key in ["moment_official", "patchtst_pretrained"]:
        model_dir = tmp_path / model_key / "zero_shot"
        model_dir.mkdir(parents=True)
        np.savez_compressed(model_dir / "embeddings.npz", labels=labels, pca=pca, tsne=tsne)
        model_dirs[model_key] = model_dir

    output_pdf = tmp_path / "comparison.pdf"
    output_png = tmp_path / "comparison.png"
    plot_pretrained_model_comparison(output_pdf=output_pdf, output_png=output_png, model_output_dirs=model_dirs)

    assert output_pdf.is_file()
    assert output_png.is_file()
    assert output_pdf.stat().st_size > 0
    assert output_png.stat().st_size > 0


def test_native_patch_stride_constants_are_explicit() -> None:
    assert MOMENT_OFFICIAL_PATCH_LEN == 8
    assert MOMENT_OFFICIAL_PATCH_STRIDE == 8
    assert PATCHTST_PRETRAIN_CONTEXT_LENGTH == 512
    assert PATCHTST_PRETRAIN_PATCH_LENGTH == 12
    assert PATCHTST_PRETRAIN_PATCH_STRIDE == 12


def test_patchtst_native_metadata_validates_pretraining_config() -> None:
    class Config:
        context_length = 4096
        patch_length = 12
        patch_stride = 12
        num_input_channels = 1

    class Model:
        config = Config()

    metadata = patchtst_native_metadata(Model())

    assert metadata["context_length"] == 4096
    assert metadata["pretrained_context_length"] == 512
    assert metadata["patch_length"] == 12
    assert metadata["patch_stride"] == 12
    assert metadata["paper_forecasting_patch_length"] == 16
    assert metadata["paper_forecasting_patch_stride"] == 8


def test_adaptive_bandpass_decimate_np_returns_target_length() -> None:
    x = np.sin(np.linspace(0, 8 * np.pi, 64, dtype=np.float32))
    y = adaptive_bandpass_decimate_np(
        x,
        target_length=16,
        native_length=64,
        native_fs_hz=2_000_000.0,
        low_khz=5.0,
        high_khz_max=100.0,
    )

    assert y.shape == (16,)
    assert np.isfinite(y).all()


def test_conv1dgap_latent_shape_for_native_input() -> None:
    from p0.models import create_model

    model = create_model("Conv1DGAP", input_length=4096, num_classes=4).eval()
    signals = torch.randn(2, 4096)

    with torch.no_grad():
        features = encode_conv1dgap_features(model, signals, device="cpu")

    assert tuple(features.shape) == (2, 256)


def test_build_aligned_signal_defaults_to_p3_4096() -> None:
    raw = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    aligned = build_aligned_signal(raw)

    assert aligned.shape == (4096,)
    assert np.isfinite(aligned).all()
    assert abs(float(aligned.mean())) < 1.0e-5
    assert 0.99 < float(aligned.std()) < 1.01


def test_build_aligned_512_signal_shape_and_scale_compat_alias() -> None:
    raw = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    aligned = build_aligned_512_signal(raw)

    assert aligned.shape == (512,)
    assert np.isfinite(aligned).all()
    assert abs(float(aligned.mean())) < 1.0e-5
    assert 0.99 < float(aligned.std()) < 1.01


def test_materialize_conv_train_views_size_and_no_test_split(tmp_path: Path) -> None:
    signal_path = tmp_path / "event.npy"
    np.save(signal_path, np.sin(np.linspace(0, 4 * np.pi, 64, dtype=np.float32)))
    event = ParticleEvent(
        event_id="train/4um/event",
        sample_id="event",
        split="train",
        signal_path=str(signal_path),
        label_path="",
        class_id=1,
        class_name="4um",
        center_norm=0.5,
        width_norm=0.0,
        center_index=32,
        crop_start=0,
        crop_end=64,
    )

    payload = materialize_conv_train_views(
        events=[event],
        labels=np.asarray([1], dtype=np.int64),
        train_indices=np.asarray([0], dtype=np.int64),
        views_per_event=3,
        raw_crop_length=64,
        output_length=16,
        jitter_frac=0.25,
        aug_snr_db=25.0,
        aug_scale_min=0.9,
        aug_scale_max=1.1,
        seed=123,
    )

    assert payload["signals"].shape == (3, 16)
    np.testing.assert_array_equal(payload["labels"], np.asarray([1, 1, 1], dtype=np.int64))
    assert set(payload["source_split"].tolist()) == {"train"}
    assert np.isfinite(payload["signals"]).all()


def test_validate_no_test_leakage_rejects_test_split() -> None:
    with pytest.raises(ValueError, match="test-derived"):
        validate_no_test_leakage(np.asarray(["train", "test"]))
