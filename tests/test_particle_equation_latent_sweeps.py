from __future__ import annotations

import json

import numpy as np

from p3_ssl.particle_equation_sweeps import (
    GAUSSIAN_FWHM_TO_SIGMA,
    REALISTIC_TAU_BASE_MS,
    REALISTIC_TAU_SWEEP_MS,
    SNR_SWEEP_QUANTILES_DB,
    WINDOW_DURATION_MS,
    _single_particle_display_signal,
    generate_single_particle_panels,
    particle_wave_skewed,
    generate_two_particle_panels,
    generate_yeast_budded_two_particle_panels,
    particle_wave,
    plot_single_model_figure,
    plot_single_particle_signal_examples,
    plot_two_particle_signal_examples,
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

    assert len(panels) == 7
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
    # The sixth physical coordinate stays inside the fitted asymmetry observed
    # on the strict Z8 v2 population, and a = 0 must reproduce the symmetric
    # burst exactly so the sweep extends the analytical family.
    skew = by_key["skew_a"]
    assert skew.params["skew_a"].min() >= -0.6
    assert skew.params["skew_a"].max() <= 0.6
    t = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    ones = np.ones(4, dtype=np.float32)
    symmetric = particle_wave(t, ones, 16.0 * ones, 0.0 * ones, 0.5 * ones, 0.06 * ones)
    unskewed = particle_wave_skewed(
        t, ones, 16.0 * ones, 0.0 * ones, 0.5 * ones, 0.06 * ones, np.zeros(4, dtype=np.float32)
    )
    assert np.array_equal(symmetric, unskewed)
    assert by_key["width_tau"].params["tau"].min() >= 0.02
    assert by_key["width_tau"].params["tau"].max() <= 0.15
    assert by_key["snr_db"].params["snr_db"].min() >= SNR_SWEEP_QUANTILES_DB["q20"]
    assert by_key["snr_db"].params["snr_db"].max() <= SNR_SWEEP_QUANTILES_DB["q80"]


def test_generate_single_particle_paper_table_ranges_and_conversions() -> None:
    panels = generate_single_particle_panels(
        n_per_panel=12,
        length=128,
        seed=4,
        noise_std=0.0,
        normalization="none",
        shuffle=False,
        sweep_source="paper_table",
    )
    by_key = {panel.key: panel for panel in panels}

    np.testing.assert_allclose(by_key["amplitude_A"].params["A"][[0, -1]], [0.10, 3.56], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(by_key["doppler_fD"].params["fD"][[0, -1]] / WINDOW_DURATION_MS, [8.00, 37.60], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        by_key["snr_db"].params["snr_db"][[0, -1]],
        [SNR_SWEEP_QUANTILES_DB["q20"], SNR_SWEEP_QUANTILES_DB["q80"]],
        rtol=1e-6,
        atol=1e-6,
    )

    expected_tau_ms = np.asarray([0.33, 1.10], dtype=np.float32) / GAUSSIAN_FWHM_TO_SIGMA
    np.testing.assert_allclose(by_key["width_tau"].params["tau"][[0, -1]] * WINDOW_DURATION_MS, expected_tau_ms, rtol=1e-6, atol=1e-6)
    assert by_key["center_t0"].params["t0"].min() >= 0.2
    assert by_key["center_t0"].params["t0"].max() <= 0.8


def test_generate_single_particle_snr_panel_uses_target_noise_level() -> None:
    panels = generate_single_particle_panels(
        n_per_panel=5,
        length=4096,
        seed=5,
        noise_std=0.0,
        normalization="none",
        shuffle=False,
        sweep_source="paper_table",
    )
    panel = {item.key: item for item in panels}["snr_db"]
    t = np.linspace(0.0, 1.0, panel.signal.shape[1], dtype=np.float32)
    clean = particle_wave(t, panel.params["A"], panel.params["fD"], panel.params["phi"], panel.params["t0"], panel.params["tau"])
    clean_rms = np.sqrt(np.mean(np.square(clean), axis=1))
    expected_noise_std = clean_rms / np.power(10.0, panel.params["snr_db"] / 20.0)
    np.testing.assert_allclose(panel.params["snr_noise_std"], expected_noise_std, rtol=1e-6, atol=1e-6)

    noise = panel.signal - clean
    measured_db = 20.0 * np.log10(clean_rms / np.sqrt(np.mean(np.square(noise), axis=1)))
    np.testing.assert_allclose(measured_db, panel.params["snr_db"], atol=0.5)


def test_realistic_figure_based_sweeps_generate_model_inputs() -> None:
    panels = generate_single_particle_panels(
        n_per_panel=5,
        length=4096,
        seed=6,
        noise_std=0.0,
        normalization="none",
        shuffle=False,
        sweep_source="paper_table",
        signal_window_duration_ms=1.0,
        realistic_figure_based_sweeps=True,
    )
    by_key = {panel.key: panel for panel in panels}

    for panel in panels:
        assert panel.window_duration_ms == 1.0

    np.testing.assert_allclose(by_key["doppler_fD"].params["fD"][[0, -1]], [8.00, 37.60], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(by_key["width_tau"].params["tau"][[0, -1]], REALISTIC_TAU_SWEEP_MS, rtol=1e-6, atol=1e-6)

    for key in ["amplitude_A", "doppler_fD", "phase_phi", "center_t0", "snr_db"]:
        np.testing.assert_allclose(by_key[key].params["tau"], REALISTIC_TAU_BASE_MS, rtol=1e-6, atol=1e-6)

    displayed_signal, displayed_envelope = _single_particle_display_signal(by_key["amplitude_A"], 2)
    np.testing.assert_allclose(displayed_signal, by_key["amplitude_A"].signal[2], rtol=1e-6, atol=1e-6)
    assert int(np.argmax(displayed_envelope)) in {2047, 2048}


def test_generate_single_particle_paper_table_visual_phase_profile() -> None:
    table_mean = generate_single_particle_panels(
        n_per_panel=12,
        length=128,
        seed=4,
        noise_std=0.0,
        normalization="none",
        shuffle=False,
        sweep_source="paper_table",
    )
    visual = generate_single_particle_panels(
        n_per_panel=12,
        length=128,
        seed=4,
        noise_std=0.0,
        normalization="none",
        shuffle=False,
        sweep_source="paper_table",
        phase_profile="visual_low_cycles",
    )
    table_mean_by_key = {panel.key: panel for panel in table_mean}
    visual_by_key = {panel.key: panel for panel in visual}

    phase = visual_by_key["phase_phi"]
    np.testing.assert_allclose(phase.params["fD"] / WINDOW_DURATION_MS, 8.00, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(phase.params["tau"] * WINDOW_DURATION_MS, 0.33 / GAUSSIAN_FWHM_TO_SIGMA, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(phase.params["A"], 0.70, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(phase.params["phi"][[0, -1]], [0.0, 2.0 * np.pi], rtol=1e-6, atol=1e-6)

    for key in ["amplitude_A", "doppler_fD", "center_t0", "width_tau"]:
        for param_name, values in table_mean_by_key[key].params.items():
            np.testing.assert_allclose(visual_by_key[key].params[param_name], values, rtol=1e-6, atol=1e-6)


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


def test_generate_yeast_budded_two_particle_panels_shapes_and_ranges(tmp_path) -> None:
    summary = {
        "two_peak_only": {
            "delta_t0": {"p05": 0.02, "p50": 0.05, "p95": 0.10},
            "amplitude_ratio_b_over_a": {"p05": 0.6, "p50": 1.0, "p95": 1.4},
            "abs_delta_phi_rad": {"p05": 0.2, "p50": 1.0, "p95": 2.5},
            "tau_ratio_b_over_a": {"p05": 0.7, "p50": 1.0, "p95": 1.6},
            "component_a_amplitude": {"p05": 1.0, "p50": 2.0, "p95": 3.0},
            "component_a_tau": {"p05": 0.01, "p50": 0.03, "p95": 0.06},
            "component_b_tau": {"p05": 0.01, "p50": 0.03, "p95": 0.06},
            "doppler_peak_hz": {"p05": 10000.0, "p50": 15000.0, "p95": 25000.0},
            "dominant_cycles_per_window": {"p05": 20.0, "p50": 30.0, "p95": 40.0},
            "snr_proxy": {"p05": 5.0, "p50": 20.0, "p95": 100.0},
        }
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(__import__("json").dumps(summary))
    panels = generate_yeast_budded_two_particle_panels(
        n_per_panel=7,
        length=128,
        seed=7,
        normalization="none",
        shuffle=False,
        range_summary_path=summary_path,
    )
    by_key = {panel.key: panel for panel in panels}

    assert len(panels) == 6
    for panel in panels:
        assert panel.signal.shape == (7, 128)
        assert panel.encoded_signal.shape == (7, 128)
        assert np.isfinite(panel.signal).all()

    np.testing.assert_allclose(by_key["delta_t0"].params["t0B"] - by_key["delta_t0"].params["t0A"], by_key["delta_t0"].color_value, rtol=1e-6)
    np.testing.assert_allclose(by_key["amplitude_ratio"].params["B"] / by_key["amplitude_ratio"].params["A"], by_key["amplitude_ratio"].color_value, rtol=1e-6)
    np.testing.assert_allclose(by_key["tau_ratio"].params["tauB"] / by_key["tau_ratio"].params["tauA"], by_key["tau_ratio"].color_value, rtol=1e-6)
    assert float(by_key["delta_phi"].color_value.min()) < 0.0
    assert float(by_key["delta_phi"].color_value.max()) > 0.0
    assert by_key["snr_proxy"].params["snr_noise_std"].shape == (7,)


def test_generate_yeast_budded_template_style_uses_real_template_bank(tmp_path) -> None:
    summary = {
        "two_peak_only": {
            "delta_t0": {"p05": 0.02, "p50": 0.05, "p75": 0.08, "p95": 0.10},
            "amplitude_ratio_b_over_a": {"p05": 0.6, "p50": 1.0, "p95": 1.4},
            "abs_delta_phi_rad": {"p05": 0.2, "p50": 1.0, "p95": 2.5},
            "tau_ratio_b_over_a": {"p05": 0.7, "p25": 0.8, "p50": 1.0, "p75": 1.3, "p95": 1.6},
            "component_a_amplitude": {"p05": 1.0, "p50": 2.0, "p95": 3.0},
            "component_a_tau": {"p05": 0.01, "p50": 0.03, "p95": 0.06},
            "component_b_tau": {"p05": 0.01, "p50": 0.03, "p95": 0.06},
            "width_ms": {"p05": 0.6, "p50": 1.0, "p95": 1.4},
            "doppler_peak_hz": {"p05": 10000.0, "p50": 15000.0, "p95": 25000.0},
            "snr_proxy": {"p05": 5.0, "p50": 20.0, "p95": 100.0},
            "background_edge_std": {"p05": 0.05, "p50": 0.08, "p95": 0.12},
        }
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))

    template_root = tmp_path / "templates"
    template_root.mkdir()
    length = 128
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    env = np.exp(-np.square(t - 0.5) / (2.0 * 0.18**2)).astype(np.float32)
    signals = np.stack(
        [
            env * np.sin(2.0 * np.pi * 18.0 * t),
            0.8 * env * np.sin(2.0 * np.pi * 22.0 * t + 0.5),
            0.6 * env * np.sin(2.0 * np.pi * 15.0 * t + 1.0),
        ]
    ).astype(np.float32)
    np.savez_compressed(template_root / "aligned_inputs.npz", signals=signals)
    rows = [
        "event_id,sample_id,quality,source_group\n",
        "a,a,strict,budding\n",
        "b,b,strict,budding\n",
        "c,c,strict,other\n",
    ]
    (template_root / "events_metadata.csv").write_text("".join(rows))
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "best_params": {
                    "template_envelope_strength": 0.35,
                    "texture_strength": 0.10,
                    "noise_scale": 0.40,
                    "tau_width_divisor": 4.0,
                    "asymmetry_strength": 0.10,
                    "secondary_doppler_strength": 0.10,
                }
            }
        )
    )

    panels = generate_yeast_budded_two_particle_panels(
        n_per_panel=6,
        length=length,
        seed=10,
        normalization="none",
        shuffle=False,
        range_summary_path=summary_path,
        style="template_budding",
        template_root=template_root,
        template_calibration=calibration_path,
    )
    by_key = {panel.key: panel for panel in panels}

    for panel in panels:
        assert panel.signal.shape == (6, length)
        assert np.isfinite(panel.signal).all()
        assert "template_bank_row" in panel.params
        assert "template_source_index" in panel.params
        assert panel.params["template_source_index"].max() <= 1.0
        np.testing.assert_allclose(panel.params["template_envelope_strength"], 0.35)
        np.testing.assert_allclose(panel.params["template_tau_width_divisor"], 4.0)

    np.testing.assert_allclose(by_key["delta_t0"].params["t0B"] - by_key["delta_t0"].params["t0A"], by_key["delta_t0"].color_value, rtol=1e-6)
    np.testing.assert_allclose(by_key["amplitude_ratio"].params["B"] / by_key["amplitude_ratio"].params["A"], by_key["amplitude_ratio"].color_value, rtol=1e-6)


def test_single_particle_plot_helpers_write_outputs(tmp_path) -> None:
    panels = generate_single_particle_panels(n_per_panel=6, length=64, seed=3, noise_std=0.0, normalization="none", shuffle=False)
    reductions = {
        panel.key: {
            "pca": np.column_stack((np.arange(panel.color_value.size), panel.color_value)).astype(np.float32),
            "tsne": np.column_stack((panel.color_value, np.arange(panel.color_value.size))).astype(np.float32),
        }
        for panel in panels
    }
    metrics = {panel.key: {} for panel in panels}

    plot_single_model_figure(
        panels=panels,
        reductions=reductions,
        metrics=metrics,
        model_key="moment_official",
        output_pdf=tmp_path / "model.pdf",
        output_png=tmp_path / "model.png",
        scenario="single_particle",
    )
    plot_single_particle_signal_examples(
        panels=panels,
        output_pdf=tmp_path / "signals.pdf",
        output_png=tmp_path / "signals.png",
    )
    two_panels = generate_yeast_budded_two_particle_panels(n_per_panel=6, length=64, seed=4, normalization="none", shuffle=False)
    plot_two_particle_signal_examples(
        panels=two_panels,
        output_pdf=tmp_path / "two_signals.pdf",
        output_png=tmp_path / "two_signals.png",
        scenario="yeast_budded_two_particles",
    )

    assert (tmp_path / "model.pdf").is_file()
    assert (tmp_path / "model.png").is_file()
    assert (tmp_path / "signals.pdf").is_file()
    assert (tmp_path / "signals.png").is_file()
    assert (tmp_path / "two_signals.pdf").is_file()
    assert (tmp_path / "two_signals.png").is_file()
