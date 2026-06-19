from __future__ import annotations

import numpy as np

from scripts.run_particle_equation_latent_sweeps import (
    generate_single_particle_panels,
    generate_two_particle_panels,
    particle_wave,
)


def test_particle_wave_shape_and_finiteness() -> None:
    t = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    n = 5
    y = particle_wave(
        t=t,
        amplitude=np.ones(n, dtype=np.float32),
        doppler=np.full(n, 16.0, dtype=np.float32),
        phase=np.zeros(n, dtype=np.float32),
        center=np.full(n, 0.5, dtype=np.float32),
        width=np.full(n, 0.06, dtype=np.float32),
    )

    assert y.shape == (n, 128)
    assert np.isfinite(y).all()
    assert float(np.max(np.abs(y))) <= 1.0


def test_generate_single_particle_panels_shapes_and_ranges() -> None:
    panels = generate_single_particle_panels(n_per_panel=12, length=128, seed=1, noise_std=0.0, normalization="none")
    by_key = {panel.key: panel for panel in panels}

    assert len(panels) == 5
    for panel in panels:
        assert panel.signal.shape == (12, 128)
        assert panel.encoded_signal.shape == (12, 128)
        assert panel.color_value.shape == (12,)
        assert np.isfinite(panel.signal).all()
        assert np.isfinite(panel.encoded_signal).all()

    assert by_key["amplitude_A"].params["A"].min() >= 0.25
    assert by_key["amplitude_A"].params["A"].max() <= 2.0
    assert by_key["doppler_fD"].params["fD"].min() >= 2.0
    assert by_key["doppler_fD"].params["fD"].max() <= 64.0
    assert by_key["phase_phi"].params["phi"].min() >= 0.0
    assert by_key["phase_phi"].params["phi"].max() <= 2.0 * np.pi
    assert by_key["center_t0"].params["t0"].min() >= 0.2
    assert by_key["center_t0"].params["t0"].max() <= 0.8
    assert by_key["width_tau"].params["tau"].min() >= 0.02
    assert by_key["width_tau"].params["tau"].max() <= 0.15


def test_generate_two_particle_panels_parameterization() -> None:
    panels = generate_two_particle_panels(n_per_panel=9, length=128, seed=2, noise_std=0.0, normalization="none", shuffle=False)
    by_key = {panel.key: panel for panel in panels}

    sep = by_key["separation_dt"]
    np.testing.assert_allclose(sep.params["t0B"] - sep.params["t0A"], sep.color_value, rtol=1e-6, atol=1e-6)
    assert float(sep.color_value.min()) == 0.0
    assert sep.signal.shape == (9, 128)

    ratio = by_key["amplitude_ratio"]
    np.testing.assert_allclose(ratio.params["B"] / ratio.params["A"], ratio.color_value, rtol=1e-6, atol=1e-6)

    freq_delta = by_key["frequency_delta"]
    np.testing.assert_allclose(freq_delta.params["fDB"] - freq_delta.params["fDA"], freq_delta.color_value, rtol=1e-6, atol=1e-6)

    width_ratio = by_key["width_ratio"]
    np.testing.assert_allclose(width_ratio.params["tauB"] / width_ratio.params["tauA"], width_ratio.color_value, rtol=1e-6, atol=1e-6)
