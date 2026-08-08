from __future__ import annotations

import numpy as np

from p3_ssl.yeast_budding_sweep_domain import (
    PARAMETERS,
    canonical_parameter_row,
    circular_center,
    domain_statistics,
    finalize_phase,
    robust_medoid,
    unwrap_around,
    wrap_phase,
)


def _fit(event_id: str, *, reverse: bool = False) -> dict[str, float | str]:
    components = [
        {
            "amplitude": 2.0,
            "center_ms": 1.0,
            "sigma_left_ms": 0.08,
            "sigma_right_ms": 0.12,
            "shape": 2.0,
            "frequency_khz": 15.0,
            "chirp_khz_per_ms": -2.0,
            "phase_rad": 3.0,
        },
        {
            "amplitude": 1.0,
            "center_ms": 1.4,
            "sigma_left_ms": 0.06,
            "sigma_right_ms": 0.10,
            "shape": 2.5,
            "frequency_khz": 17.0,
            "chirp_khz_per_ms": 3.0,
            "phase_rad": -3.0,
        },
    ]
    if reverse:
        components.reverse()
    row: dict[str, float | str] = {
        "event_id": event_id,
        "delta_bic_m1_minus_m2": 20.0,
        "resolvability_score": 0.5,
    }
    for index, component in enumerate(components, start=1):
        for name, value in component.items():
            row[f"m2_c{index}_{name}"] = value
    return row


def test_canonical_order_is_invariant_to_fit_component_order() -> None:
    first = canonical_parameter_row(
        _fit("a"), snr_db=10.0, amplitude_scale=1.25, population="primary"
    )
    second = canonical_parameter_row(
        _fit("b", reverse=True),
        snr_db=10.0,
        amplitude_scale=1.25,
        population="primary",
    )
    for parameter in PARAMETERS:
        assert np.isclose(first[parameter], second[parameter])
    assert first["delta_t0_ms"] > 0.0


def test_phase_unwrap_handles_branch_cut() -> None:
    values = np.asarray([3.10, -3.12, 3.05])
    center = circular_center(values)
    unwrapped = unwrap_around(values, center)
    assert np.ptp(unwrapped) < 0.2
    assert all(-np.pi <= wrap_phase(value) <= np.pi for value in unwrapped)


def test_gold_phase_can_use_primary_reference_center() -> None:
    rows = [
        canonical_parameter_row(
            _fit(str(index)), snr_db=10.0, amplitude_scale=1.0, population="gold"
        )
        for index in range(8)
    ]
    for index, row in enumerate(rows):
        row["delta_phi_rad"] = -3.1 + index * 0.01
    reference = 3.12
    assert finalize_phase(rows, reference_center=reference) == reference
    assert np.ptp([row["delta_phi_rad"] for row in rows]) < 0.1


def test_domain_has_exactly_31_q01_q99_points_and_medoid() -> None:
    rows = []
    for index in range(12):
        row = canonical_parameter_row(
            _fit(str(index)),
            snr_db=5.0 + index,
            amplitude_scale=1.0 + index / 100.0,
            population="primary",
        )
        row["delta_t0_ms"] += index / 100.0
        rows.append(row)
    finalize_phase(rows)
    stats, grid = domain_statistics(rows)
    assert len(stats) == len(PARAMETERS)
    assert len(grid) == len(PARAMETERS) * 31
    assert {row["probability"] for row in grid if row["parameter"] == PARAMETERS[0]} == set(
        np.linspace(0.01, 0.99, 31)
    )
    assert robust_medoid(rows) in {row["event_id"] for row in rows}
