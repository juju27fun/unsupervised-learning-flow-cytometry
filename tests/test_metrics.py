from __future__ import annotations

import torch

from p3_ssl.metrics import (
    finalize_mask_coherence_sums,
    finalize_reconstruction_metric_sums,
    mask_coherence_batch_sums,
    reconstruction_metric_sums,
    reconstruction_strata_masks,
)


def test_reconstruction_strata_masks_classifies_event_overlap() -> None:
    target = torch.zeros(4, 6, dtype=torch.bool)
    hidden = torch.zeros(4, 6, dtype=torch.bool)
    event = torch.zeros(4, 6, dtype=torch.bool)

    event[0, 2:4] = True
    target[0, 0:2] = True
    event[1, 1:5] = True
    target[1, 1:2] = True
    event[2, 1:5] = True
    target[2, 1:4] = True
    event[3, 2:4] = True
    target[3, 2:4] = True
    hidden[3, 2:4] = True

    strata = reconstruction_strata_masks(target, event, hidden_mask=hidden)

    assert strata["background_only"].tolist() == [True, False, False, False]
    assert strata["partial_event"].tolist() == [False, True, False, False]
    assert strata["mostly_event"].tolist() == [False, False, True, False]
    assert strata["impossible_event"].tolist() == [False, False, False, True]


def test_reconstruction_metric_sums_aggregate_exactly() -> None:
    pred = torch.tensor([[[0.0, 2.0, 4.0]]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[False, True, True]])

    metrics = finalize_reconstruction_metric_sums(reconstruction_metric_sums(pred, target, mask))

    assert metrics["masked_points"] == 2.0
    assert metrics["masked_mse"] == 10.0
    assert metrics["masked_mae"] == 3.0


def test_mask_coherence_batch_sums_report_impossible_rate() -> None:
    target = torch.tensor([[False, True, True], [True, False, False]])
    hidden = torch.tensor([[False, True, True], [False, False, False]])
    event = torch.tensor([[False, True, True], [False, True, False]])

    summary = finalize_mask_coherence_sums(mask_coherence_batch_sums(target, hidden, event))

    assert summary["event_samples"] == 2.0
    assert summary["fully_hidden_event_sample_rate"] == 0.5
    assert summary["mean_event_target_fraction"] == 0.5
