from __future__ import annotations

import numpy as np

from p3_ssl.yeast_budding_m2_sweep import (
    PARAMETERS,
    Carrier,
    analyze_embeddings,
    build_sweep_bank,
    interpolate_parameter_grids,
    physical_parameters,
    select_real_carriers,
    wrap_phase,
)


def _anchor() -> dict[str, float | str | bool]:
    return {
        "event_id": "anchor",
        "log_A_A": 0.0,
        "fD_A_khz": 18.0,
        "log_tau_A_ms": np.log(0.08),
        "snr_db": 8.0,
        "log_B_over_A": np.log(0.8),
        "delta_t0_ms": 0.18,
        "delta_fD_khz": 2.0,
        "delta_phi_rad": 0.2,
        "log_tau_B_over_tau_A": np.log(1.2),
        "anchor_shape_A": 2.0,
        "anchor_shape_B": 2.4,
        "anchor_sigma_left_over_tau_A": 0.8,
        "anchor_sigma_right_over_tau_A": 1.2,
        "anchor_sigma_left_over_tau_B": 1.1,
        "anchor_sigma_right_over_tau_B": 0.9,
        "anchor_chirp_A_khz_per_ms": 0.0,
        "anchor_chirp_B_khz_per_ms": 0.0,
    }


def _carriers() -> list[Carrier]:
    x = np.linspace(0, 2 * np.pi, 4096, dtype=np.float32)
    return [
        Carrier("c0", "r0", 0, 0.5, np.sin(37 * x).astype(np.float32)),
        Carrier("c1", "r1", 1, 0.75, np.cos(41 * x).astype(np.float32)),
    ]


def test_physical_conversion_and_phase_wrap() -> None:
    anchor = _anchor()
    transformed = {parameter: float(anchor[parameter]) for parameter in PARAMETERS}
    transformed["delta_phi_rad"] = 4.0
    values = physical_parameters(
        transformed,
        anchor,
        common_phase=np.pi / 2,
        position_fraction=0.4,
        amplitude_v2_to_v1=2.0,
    )
    assert np.isclose(values["amplitude_a"], 2.0)
    assert values["center_a_ms"] < values["center_b_ms"]
    assert -np.pi <= values["phase_b_rad"] < np.pi
    assert np.isclose(wrap_phase(3 * np.pi), -np.pi)


def test_bank_varies_one_parameter_and_is_deterministic() -> None:
    anchor = _anchor()
    grids = {
        parameter: np.linspace(float(anchor[parameter]) - 0.1, float(anchor[parameter]) + 0.1, 31)
        for parameter in PARAMETERS
    }
    first, metadata = build_sweep_bank(
        anchor=anchor,
        grids=grids,
        carriers=_carriers(),
        amplitude_v2_to_v1=1.0,
        parameter_subset=("log_A_A", "delta_phi_rad"),
        quantile_indices=(0, 15, 30),
        phases=(0.0,),
        positions=(0.5,),
    )
    second, second_metadata = build_sweep_bank(
        anchor=anchor,
        grids=grids,
        carriers=_carriers(),
        amplitude_v2_to_v1=1.0,
        parameter_subset=("log_A_A", "delta_phi_rad"),
        quantile_indices=(0, 15, 30),
        phases=(0.0,),
        positions=(0.5,),
    )
    assert first.shape == (14, 4096)
    assert np.array_equal(first, second)
    assert [row["sample_id"] for row in metadata] == [row["sample_id"] for row in second_metadata]
    anchor_values = {parameter: float(anchor[parameter]) for parameter in PARAMETERS}
    for row in metadata:
        if row["sample_kind"] != "sweep":
            continue
        changed = [parameter for parameter in PARAMETERS if not np.isclose(float(row[parameter]), anchor_values[parameter])]
        assert changed in ([], [row["sweep_parameter"]])


def test_dense_grid_interpolation_preserves_endpoints_and_monotonicity() -> None:
    anchor = _anchor()
    grids = {
        parameter: np.linspace(float(anchor[parameter]) - 0.2, float(anchor[parameter]) + 0.3, 31)
        for parameter in PARAMETERS
    }
    dense, probabilities = interpolate_parameter_grids(grids, count=225)
    assert probabilities.shape == (225,)
    assert np.isclose(probabilities[0], 0.01)
    assert np.isclose(probabilities[-1], 0.99)
    for parameter in PARAMETERS:
        assert dense[parameter].shape == (225,)
        assert dense[parameter][0] == grids[parameter][0]
        assert dense[parameter][-1] == grids[parameter][-1]
        assert np.all(np.diff(dense[parameter]) >= 0.0)


def test_dense_bank_uses_explicit_probabilities() -> None:
    anchor = _anchor()
    grids = {
        parameter: np.linspace(float(anchor[parameter]) - 0.1, float(anchor[parameter]) + 0.1, 5)
        for parameter in PARAMETERS
    }
    probabilities = tuple(np.linspace(0.01, 0.99, 5).tolist())
    bank, metadata = build_sweep_bank(
        anchor=anchor,
        grids=grids,
        carriers=_carriers(),
        amplitude_v2_to_v1=1.0,
        parameter_subset=("log_A_A",),
        quantile_indices=tuple(range(5)),
        quantile_probabilities=probabilities,
        phases=(0.0,),
        positions=(0.5,),
    )
    assert bank.shape == (12, 4096)
    sweep = [row for row in metadata if row["sample_kind"] == "sweep"]
    assert [row["quantile_probability"] for row in sweep[:5]] == list(probabilities)


def test_carriers_are_source_disjoint_and_deterministic() -> None:
    rows = []
    signals = np.stack([np.arange(4096, dtype=np.float32) + index for index in range(8)])
    for index in range(8):
        rows.append(
            {
                "class_name": "background",
                "source_group_original": "budding",
                "development_split": "development_train",
                "background_energy": str(index + 1),
                "record_id": f"record-{index // 2}",
                "sample_id": f"sample-{index}",
                "signal_row": str(index),
            }
        )
    first = select_real_carriers(rows, signals)
    second = select_real_carriers(rows, signals)
    assert [row.carrier_id for row in first] == [row.carrier_id for row in second]
    assert len({row.record_id for row in first}) == 2


def test_latent_metrics_are_finite_and_regular_path_scores_well() -> None:
    metadata = []
    embeddings = []
    probabilities = []
    for context in range(8):
        context_id = f"c{context}"
        anchor = np.zeros(512, dtype=np.float32)
        anchor[0] = 1.0
        metadata.append({"sample_id": f"a-{context}", "sample_kind": "anchor", "sweep_parameter": "anchor", "context_id": context_id, "quantile_index": -1, "quantile_probability": 0.5})
        embeddings.append(anchor)
        probabilities.append([0.1, 0.7, 0.1, 0.1])
        for parameter in PARAMETERS:
            for quantile in range(31):
                angle = 0.2 * quantile / 30.0
                vector = np.zeros(512, dtype=np.float32)
                vector[0] = np.cos(angle)
                vector[1] = np.sin(angle)
                metadata.append({"sample_id": f"{parameter}-{context}-{quantile}", "sample_kind": "sweep", "sweep_parameter": parameter, "context_id": context_id, "quantile_index": quantile, "quantile_probability": 0.01 + quantile * 0.98 / 30})
                embeddings.append(vector)
                probabilities.append([0.1, 0.6 + 0.1 * quantile / 30, 0.2 - 0.1 * quantile / 30, 0.1])
    point, summary = analyze_embeddings(
        model_name="test",
        metadata=metadata,
        embeddings_l2=np.asarray(embeddings, dtype=np.float32),
        probabilities=np.asarray(probabilities, dtype=np.float32),
    )
    assert len(point) == len(metadata)
    assert len(summary) == 9
    assert min(row["path_efficiency_median"] for row in summary) > 0.99
    assert min(row["monotonicity_median"] for row in summary) > 0.99
    assert all(np.isfinite(list(row.values())[3:]).all() for row in summary)


def test_latent_metrics_accept_a_smoke_parameter_subset() -> None:
    metadata = []
    embeddings = []
    probabilities = []
    for context in range(2):
        context_id = f"c{context}"
        anchor = np.zeros(512, dtype=np.float32)
        anchor[0] = 1.0
        metadata.append({"sample_id": f"a-{context}", "sample_kind": "anchor", "sweep_parameter": "anchor", "context_id": context_id, "quantile_index": -1, "quantile_probability": 0.5})
        embeddings.append(anchor)
        probabilities.append([0.1, 0.7, 0.1, 0.1])
        for quantile in range(3):
            angle = 0.1 * quantile
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = np.cos(angle)
            vector[1] = np.sin(angle)
            metadata.append({"sample_id": f"p-{context}-{quantile}", "sample_kind": "sweep", "sweep_parameter": "log_A_A", "context_id": context_id, "quantile_index": quantile, "quantile_probability": quantile / 2})
            embeddings.append(vector)
            probabilities.append([0.1, 0.7, 0.1, 0.1])
    _, summary = analyze_embeddings(
        model_name="test",
        metadata=metadata,
        embeddings_l2=np.asarray(embeddings, dtype=np.float32),
        probabilities=np.asarray(probabilities, dtype=np.float32),
    )
    assert [row["sweep_parameter"] for row in summary] == ["log_A_A"]
