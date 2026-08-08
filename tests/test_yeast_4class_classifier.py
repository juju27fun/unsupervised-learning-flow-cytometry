from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from p3_ssl.yeast_4class_classifier import (
    CLASS_NAMES,
    BalancedBatchSampler,
    FROZEN_STFT_CONFIG,
    augment_training_batch,
    build_source_disjoint_80_20_split,
    classification_metrics,
    create_yeast_classifier_model,
    encode_signals,
    load_checkpoint,
    supervised_contrastive_loss,
)


def test_source_disjoint_80_20_split_is_deterministic_and_complete() -> None:
    rows = []
    labels = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        for group_index in range(10):
            for row_index in range(2):
                rows.append(
                    {
                        "sample_id": f"{class_name}-{group_index}-{row_index}",
                        "record_id": f"{class_name}-record-{group_index}",
                    }
                )
                labels.append(class_id)
    labels_array = np.asarray(labels, dtype=np.int64)
    first = build_source_disjoint_80_20_split(rows, labels_array, np.arange(len(rows)), candidates=64)
    second = build_source_disjoint_80_20_split(rows, labels_array, np.arange(len(rows)), candidates=64)
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.validation_indices, second.validation_indices)
    train_groups = {rows[index]["record_id"] for index in first.train_indices}
    validation_groups = {rows[index]["record_id"] for index in first.validation_indices}
    assert train_groups.isdisjoint(validation_groups)
    assert set(first.train_indices) | set(first.validation_indices) == set(range(len(rows)))
    assert set(labels_array[first.validation_indices]) == set(range(4))


def test_balanced_sampler_yields_exact_quarters_and_is_deterministic() -> None:
    labels = np.asarray([0] * 3 + [1] * 5 + [2] * 7 + [3] * 9, dtype=np.int64)
    first = BalancedBatchSampler(labels, np.arange(labels.size), batch_size=8, seed=42)
    second = BalancedBatchSampler(labels, np.arange(labels.size), batch_size=8, seed=42)
    batches = list(first)
    assert batches == list(second)
    assert len(batches) == 5
    for batch in batches:
        assert Counter(labels[batch].tolist()) == {0: 2, 1: 2, 2: 2, 3: 2}


def test_minority_epoch_sampler_does_not_replay_the_smallest_class() -> None:
    labels = np.asarray([0] * 12 + [1] * 5 + [2] * 9 + [3] * 7, dtype=np.int64)
    sampler = BalancedBatchSampler(
        labels,
        np.arange(labels.size),
        batch_size=8,
        seed=42,
        epoch_size_policy="minority",
    )
    batches = list(sampler)
    assert len(batches) == 3
    smallest_draws = [index for batch in batches for index in batch if labels[index] == 1]
    assert len(smallest_draws) == 6
    assert len(set(smallest_draws[:5])) == 5


def test_explicit_batch_budget_overrides_dataset_imbalance() -> None:
    labels = np.asarray([0] * 30 + [1] * 4 + [2] * 8 + [3] * 12, dtype=np.int64)
    sampler = BalancedBatchSampler(
        labels,
        np.arange(labels.size),
        batch_size=8,
        seed=7,
        batches_per_epoch=11,
    )
    assert len(sampler) == 11
    assert len(list(sampler)) == 11


def test_classification_metrics_include_explicit_calibration() -> None:
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    probabilities = np.asarray(
        [[.7, .1, .1, .1], [.1, .7, .1, .1], [.1, .1, .7, .1], [.1, .1, .1, .7]],
        dtype=np.float32,
    )
    metrics = classification_metrics(labels, probabilities)
    assert metrics["expected_calibration_error_10bin"] == pytest.approx(.3)
    assert set(metrics["ovr_auroc_per_class"]) == set(CLASS_NAMES)
    assert sum(row["count"] for row in metrics["calibration_bins"]) == 4
    assert metrics["event_only"]["balanced_accuracy"] == pytest.approx(1.0)


def test_event_only_metrics_do_not_get_inflated_by_background() -> None:
    labels = np.asarray([0] * 20 + [1, 2, 3], dtype=np.int64)
    probabilities = np.zeros((23, 4), dtype=np.float32)
    probabilities[:, 0] = 1.0
    metrics = classification_metrics(labels, probabilities)
    assert metrics["accuracy"] == pytest.approx(20 / 23)
    assert metrics["event_only"]["accuracy"] == 0.0
    assert metrics["event_only"]["balanced_accuracy"] == 0.0


def test_train_augmentation_is_deterministic_and_has_no_wraparound() -> None:
    inputs = torch.zeros(2, 4096)
    inputs[:, 0] = 1.0
    labels = torch.tensor([1, 0])
    backgrounds = np.ones((3, 4096), dtype=np.float32)
    first = augment_training_batch(
        inputs,
        labels,
        background_bank=backgrounds,
        rng=np.random.default_rng(7),
        max_shift_points=3,
        amplitude_scale_min=1.0,
        amplitude_scale_max=1.0,
        real_noise_fraction_max=0.0,
    )
    second = augment_training_batch(
        inputs,
        labels,
        background_bank=backgrounds,
        rng=np.random.default_rng(7),
        max_shift_points=3,
        amplitude_scale_min=1.0,
        amplitude_scale_max=1.0,
        real_noise_fraction_max=0.0,
    )
    torch.testing.assert_close(first, second)
    assert float(first[:, -1].sum()) == 0.0


def test_projected_models_and_contrastive_loss_share_the_512d_contract() -> None:
    inputs = torch.randn(8, 1, 4096)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    for model_name in ("Conv1DGAP-L", "InceptionTime1D-XS", "ResNet1D-XS"):
        model = create_yeast_classifier_model(model_name).eval()
        with torch.no_grad():
            logits, features = model(inputs, return_features=True)
        assert logits.shape == (8, 4)
        assert features.shape == (8, 512)
        assert torch.isfinite(features).all()
    loss = supervised_contrastive_loss(features, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_spectral_and_dual_branch_models_share_frozen_512d_contract() -> None:
    inputs = torch.randn(3, 1, 4096)
    assert FROZEN_STFT_CONFIG == {
        "n_fft": 256,
        "win_length": 256,
        "hop_length": 64,
        "window": "hann",
        "center": False,
        "magnitude_transform": "log1p",
    }
    for model_name in ("STFT-CNN", "DualBranch-ResNet1D-STFT"):
        model = create_yeast_classifier_model(model_name, head_type="hierarchical").eval()
        with torch.no_grad():
            logits, features = model(inputs, return_features=True)
        assert logits.shape == (3, 4)
        assert features.shape == (3, 512)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(features).all()


def test_group_normalized_model_has_no_running_batch_statistics() -> None:
    model = create_yeast_classifier_model("ResNet1D-XS", normalization="group")
    assert not any(isinstance(module, torch.nn.BatchNorm1d) for module in model.modules())
    assert any(isinstance(module, torch.nn.GroupNorm) for module in model.modules())


def test_hierarchical_head_returns_normalized_four_class_distribution() -> None:
    model = create_yeast_classifier_model("ResNet1D-XS", head_type="hierarchical").eval()
    with torch.no_grad():
        logits, features = model(torch.randn(6, 1, 4096), return_features=True)
        probabilities = torch.softmax(logits, dim=1)
        event_logits, condition_logits = model.classifier.component_logits(features)
    assert logits.shape == (6, 4)
    assert event_logits.shape == (6,)
    assert condition_logits.shape == (6, 3)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(6))


def test_hierarchical_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = create_yeast_classifier_model("Conv1DGAP-L", head_type="hierarchical").eval()
    checkpoint = tmp_path / "hierarchical.pt"
    torch.save(
        {
            "classifier_schema_version": 2,
            "model_name": "Conv1DGAP-L",
            "normalization": "batch",
            "head_type": "hierarchical",
            "input_length": 4096,
            "class_names": list(CLASS_NAMES),
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    loaded, payload = load_checkpoint(checkpoint)
    assert payload["head_type"] == "hierarchical"
    assert loaded.head_type == "hierarchical"


def test_encoding_contract_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    from p0.models import create_model

    model = create_model("Conv1DGAP-L", input_length=4096, num_classes=4).eval()
    signals = np.random.default_rng(7).normal(size=(3, 4096)).astype(np.float32)
    output = encode_signals(model, signals, device="cpu", batch_size=2)
    assert output["logits"].shape == (3, 4)
    assert output["probabilities"].shape == (3, 4)
    assert output["embeddings"].shape == (3, 512)
    assert output["embeddings_l2"].shape == (3, 512)
    np.testing.assert_allclose(np.linalg.norm(output["embeddings_l2"], axis=1), 1.0, atol=1e-6)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_name": "Conv1DGAP-L",
            "input_length": 4096,
            "class_names": list(CLASS_NAMES),
            "model_state_dict": model.state_dict(),
            "input_contract": {"contract_id": "test"},
        },
        checkpoint,
    )
    loaded, payload = load_checkpoint(checkpoint)
    assert payload["class_names"] == list(CLASS_NAMES)
    roundtrip = encode_signals(loaded, signals, device="cpu")
    np.testing.assert_allclose(roundtrip["logits"], output["logits"], atol=1e-6)
