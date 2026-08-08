from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import p3_ssl.bead_ssl as bead_ssl_module
from p3_ssl.bead_ssl import (
    RealBeadValidationDataset,
    SingleBeadSimulationDataset,
    build_cyclic25_mask_batch,
    build_cyclic25_masks_for_sample,
    build_p25_mask_batch,
    configure_experiment,
    evaluate_reconstruction,
    load_bead_ssl_config,
    training_mask_seed,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs/bead_ssl_p25_v1.yaml"


def test_bead_ssl_config_is_label_free_and_test_sealed() -> None:
    config = load_bead_ssl_config(CONFIG)
    assert config["study"]["protocol"] == "bead-ssl-comparison-v1"
    assert config["study"]["simulation_dataset"] == "yeast-passage-simulations@v1"
    assert config["study"]["training_stage"] == "synthetic_only"
    assert config["masking"]["evaluation_policy"] == "P25"
    assert config["model"]["mask_encoding"] == "sample_visibility_v1"
    assert set(config["loss"]["cells"]) == {"B0", "B1", "B2", "B3"}
    assert "test" in config["study"]["forbidden_splits"]


def test_bead_ssl_rejects_superseded_v2_reassessment() -> None:
    v2_config = CONFIG.with_name("bead_ssl_p25_v2.yaml")
    with pytest.raises(ValueError, match="frozen on protocol"):
        load_bead_ssl_config(v2_config)


def test_single_bead_dataset_filters_split_and_component_count(
    tmp_path: Path,
) -> None:
    np.save(tmp_path / "signals.npy", np.arange(4 * 4096, dtype=np.float32).reshape(4, 4096))
    fields = (
        "signal_row",
        "latent_id",
        "view_index",
        "split",
        "component_count",
        "event_position_fraction",
        "duration_ms",
    )
    with (tmp_path / "simulation_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                dict(signal_row=0, latent_id="a", view_index=0, split="train", component_count=1, event_position_fraction=0.5, duration_ms=0.8),
                dict(signal_row=1, latent_id="b", view_index=0, split="train", component_count=2, event_position_fraction=0.5, duration_ms=0.8),
                dict(signal_row=2, latent_id="c", view_index=0, split="validation", component_count=1, event_position_fraction=0.5, duration_ms=0.8),
                dict(signal_row=3, latent_id="d", view_index=0, split="train", component_count=1, event_position_fraction=0.5, duration_ms=0.8),
            ]
        )
    dataset = SingleBeadSimulationDataset(
        tmp_path, split="train", normalization="window_zscore"
    )
    assert len(dataset) == 2
    assert dataset[0]["sample_id"] == "a:view-0"
    assert float(dataset[0]["signal"].mean()) == pytest.approx(0.0, abs=1e-5)
    assert int(dataset[0]["event_mask"].sum()) == 800


def test_real_bead_dataset_exposes_annotated_event_mask(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = SimpleNamespace(
        identifier="val:example:0",
        signal=np.linspace(-1.0, 1.0, 4096, dtype=np.float32),
        metadata={"event_start_index": 1500.2, "event_end_index": 2100.7},
    )
    monkeypatch.setattr(
        bead_ssl_module,
        "load_particle_population",
        lambda _root, split: {"2um": [row], "4um": [], "10um": []},
    )
    dataset = RealBeadValidationDataset(
        tmp_path,
        split="val",
        normalization="window_zscore",
    )
    sample = dataset[0]
    assert sample["event_mask"].dtype == torch.bool
    assert int(sample["event_mask"].sum()) == 601
    assert sample["event_mask"][1500]
    assert sample["event_mask"][2100]


def test_p25_batch_masks_are_deterministic_and_exact() -> None:
    config = load_bead_ssl_config(CONFIG)
    signals = torch.randn(3, 1, 4096)
    indices = torch.tensor([3, 7, 11])
    first = build_p25_mask_batch(signals, indices, config, seed=42)
    second = build_p25_mask_batch(signals, indices, config, seed=42)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[0].sum(dim=1).tolist() == [1024, 1024, 1024]
    assert first[1].sum(dim=1).tolist() == [64, 64, 64]


def test_experiment_cells_preserve_common_protocol() -> None:
    base = load_bead_ssl_config(CONFIG)
    configured = configure_experiment(
        base,
        loss_cell="B3",
        mask_policy="CYCLIC25",
        seed=44,
    )
    assert configured["loss"]["selected_cell"] == "B3"
    assert configured["masking"]["training_policy"] == "CYCLIC25"
    assert configured["masking"]["evaluation_policy"] == "P25"
    assert configured["training"]["seed"] == 44
    assert base["loss"]["selected_cell"] == "B0"


def test_cyclic25_batch_is_exact_seeded_and_advances() -> None:
    config = configure_experiment(
        load_bead_ssl_config(CONFIG),
        loss_cell="B0",
        mask_policy="CYCLIC25",
        seed=42,
    )
    events = torch.zeros(2, 4096, dtype=torch.bool)
    events[0, 1700:2164] = True
    events[1, 1400:2400] = True
    indices = torch.tensor([3, 7])
    first = build_cyclic25_mask_batch(
        events,
        indices,
        config,
        seed=42,
        cycle_step=0,
    )
    repeated = build_cyclic25_mask_batch(
        events,
        indices,
        config,
        seed=42,
        cycle_step=0,
    )
    advanced = build_cyclic25_mask_batch(
        events,
        indices,
        config,
        seed=42,
        cycle_step=1,
    )
    assert torch.equal(first, repeated)
    assert first.sum(dim=1).tolist() == [1024, 1024]
    assert not torch.equal(first, advanced)


def test_cyclic25_seed_is_stable_while_p25_seed_changes() -> None:
    base = load_bead_ssl_config(CONFIG)
    cyclic = configure_experiment(
        base,
        loss_cell="B0",
        mask_policy="CYCLIC25",
        seed=42,
    )
    p25 = configure_experiment(
        base,
        loss_cell="B0",
        mask_policy="P25",
        seed=42,
    )
    assert training_mask_seed(
        cyclic, seed=42, epoch=1, batch_index=3
    ) == training_mask_seed(
        cyclic, seed=42, epoch=12, batch_index=99
    )
    assert training_mask_seed(
        p25, seed=42, epoch=1, batch_index=3
    ) != training_mask_seed(
        p25, seed=42, epoch=12, batch_index=99
    )


def test_cyclic25_evaluation_aggregates_every_unique_pass() -> None:
    config = load_bead_ssl_config(CONFIG)
    event = np.zeros(4096, dtype=bool)
    event[1700:2164] = True
    cycle = build_cyclic25_masks_for_sample(
        event,
        0,
        config,
        seed=42,
    )
    loader = DataLoader(
        [
            {
                "signal": torch.linspace(-1.0, 1.0, 4096).unsqueeze(0),
                "event_mask": torch.from_numpy(event),
                "sample_index": 0,
                "sample_id": "sim:0",
            }
        ],
        batch_size=1,
    )

    class ZeroModel(torch.nn.Module):
        def forward(
            self,
            signal: torch.Tensor,
            *,
            time_mask: torch.Tensor,
        ) -> torch.Tensor:
            return torch.zeros_like(signal)

    metrics, examples = evaluate_reconstruction(
        ZeroModel(),
        loader,
        config,
        torch.device("cpu"),
        mask_seed=42,
        evaluation_policy="CYCLIC25",
        max_examples=1,
        include_regions=True,
    )
    assert metrics["model"]["masked_points"] == cycle.shape[0] * 1024
    assert (
        metrics["model"]["event_support_masked_points"]
        + metrics["model"]["background_masked_points"]
        == metrics["model"]["masked_points"]
    )
    assert examples["mask"].shape == (1, 4096)
