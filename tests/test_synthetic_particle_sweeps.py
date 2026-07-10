from __future__ import annotations

import numpy as np

from p3_ssl.synthetic_particle_sweeps import (
    DOUBLE_RANGES,
    SINGLE_RANGES,
    generate_double_particle_panel,
    generate_particle_sweep_panels,
    generate_single_particle_panel,
    particle_signal,
    plot_model_comparison_figure,
    reduce_embeddings,
    reduce_model_embeddings,
)


def test_particle_signal_shape_and_peak_location() -> None:
    t = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    signal = particle_signal(
        t=t,
        A=np.asarray([1.0], dtype=np.float32),
        fD=np.asarray([0.0], dtype=np.float32),
        phi=np.asarray([0.0], dtype=np.float32),
        t0=np.asarray([0.5], dtype=np.float32),
        tau=np.asarray([0.05], dtype=np.float32),
    )

    assert signal.shape == (1, 128)
    assert np.isfinite(signal).all()
    assert abs(int(np.argmax(signal[0])) - 64) <= 1


def test_generate_single_particle_panel_shapes_and_ranges() -> None:
    rng = np.random.default_rng(0)
    for parameter, (low, high) in SINGLE_RANGES.items():
        panel = generate_single_particle_panel(rng, parameter, n=12, length=128, noise_std=0.0)

        assert panel.signal.shape == (12, 128)
        assert panel.family == "single"
        assert panel.swept_parameter == parameter
        assert np.isfinite(panel.signal).all()
        assert np.isclose(panel.color_value[0], low)
        assert np.isclose(panel.color_value[-1], high)
        assert np.all(np.diff(panel.color_value) >= 0.0)


def test_generate_double_particle_panel_delta_t0_is_symmetric() -> None:
    rng = np.random.default_rng(0)
    panel = generate_double_particle_panel(rng, "delta_t0", n=9, length=128, noise_std=0.0)
    low, high = DOUBLE_RANGES["delta_t0"]

    assert panel.signal.shape == (9, 128)
    assert panel.family == "double"
    assert np.isclose(panel.color_value[0], low)
    assert np.isclose(panel.color_value[-1], high)
    np.testing.assert_allclose(panel.parameters["t0B"] - panel.parameters["t0A"], panel.parameters["delta_t0"], atol=1.0e-6)
    np.testing.assert_allclose(0.5 * (panel.parameters["t0A"] + panel.parameters["t0B"]), 0.5, atol=1.0e-6)


def test_generate_particle_sweep_panels_family_counts() -> None:
    panels = generate_particle_sweep_panels(
        n_per_panel=6,
        length=64,
        seed=2,
        noise_std=0.0,
        include_single=True,
        include_double=True,
    )

    assert len(panels) == 9
    assert sum(panel.family == "single" for panel in panels) == 5
    assert sum(panel.family == "double" for panel in panels) == 4


def test_reduce_embeddings_shapes_for_small_input() -> None:
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(4, 3)).astype(np.float32)
    pca, tsne, metrics = reduce_embeddings(embeddings, seed=1)

    assert pca.shape == (4, 2)
    assert tsne.shape == (4, 2)
    assert "pca_explained_variance_ratio_sum" in metrics
    assert np.isnan(metrics["trustworthiness"])


def test_plot_model_comparison_figure_writes_outputs(tmp_path) -> None:
    rng = np.random.default_rng(0)
    panels = [generate_single_particle_panel(rng, "A", n=6, length=64, noise_std=0.0)]
    coords = np.column_stack((np.arange(6), np.linspace(0.0, 1.0, 6))).astype(np.float32)
    reductions = {
        "moment_official": {"single_A": {"pca": coords, "tsne": coords[:, ::-1]}},
        "patchtst_pretrained": {"single_A": {"pca": coords * 0.5, "tsne": coords[:, ::-1] * 0.5}},
    }

    output_pdf = tmp_path / "comparison.pdf"
    output_png = tmp_path / "comparison.png"
    plot_model_comparison_figure(
        panels=panels,
        reductions=reductions,
        model_keys=["moment_official", "patchtst_pretrained"],
        family="single",
        output_pdf=output_pdf,
        output_png=output_png,
    )

    assert output_pdf.is_file()
    assert output_png.is_file()
    assert output_pdf.stat().st_size > 0
    assert output_png.stat().st_size > 0


def test_custom_single_ranges_are_used() -> None:
    rng = np.random.default_rng(0)
    custom = dict(SINGLE_RANGES)
    custom["fD"] = (3.0, 7.0)

    panel = generate_single_particle_panel(rng, "fD", n=5, length=64, noise_std=0.0, ranges=custom)

    assert np.isclose(panel.color_value[0], 3.0)
    assert np.isclose(panel.color_value[-1], 7.0)
    np.testing.assert_allclose(panel.parameters["fD"], panel.color_value)


def test_reduce_model_embeddings_family_scope_splits_back_to_panels() -> None:
    rng = np.random.default_rng(0)
    panels = [
        generate_single_particle_panel(rng, "A", n=6, length=64, noise_std=0.0),
        generate_single_particle_panel(rng, "tau", n=6, length=64, noise_std=0.0),
    ]
    embeddings = {panel.key: rng.normal(size=(6, 4)).astype(np.float32) for panel in panels}

    reductions, metrics = reduce_model_embeddings(embeddings, panels, scope="family", seed=1)

    assert set(reductions) == {"single_A", "single_tau"}
    assert reductions["single_A"]["pca"].shape == (6, 2)
    assert reductions["single_tau"]["tsne"].shape == (6, 2)
    assert set(metrics) == {"single"}
