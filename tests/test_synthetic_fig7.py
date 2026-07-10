from __future__ import annotations

import numpy as np

from p3_ssl.synthetic_fig7 import (
    generate_synthetic_panels,
    reduce_embeddings,
)


def test_generate_synthetic_panels_shapes_and_ranges() -> None:
    panels = generate_synthetic_panels(n_per_panel=12, length=128, seed=1, noise_std=0.1)
    assert len(panels) == 5
    by_key = {panel.key: panel for panel in panels}
    for panel in panels:
        assert panel.signal.shape == (12, 128)
        assert panel.color_value.shape == (12,)
        assert np.isfinite(panel.signal).all()
    assert by_key["trend"].c_value.min() >= 1.0 / 8.0
    assert by_key["trend"].c_value.max() <= 8.0
    assert by_key["amplitude"].c_value.min() >= 1.0 / 4.0
    assert by_key["amplitude"].c_value.max() <= 4.0
    assert by_key["frequency"].c_value.min() >= 1.0
    assert by_key["frequency"].c_value.max() <= 32.0
    assert by_key["baseline_shift"].c_value.min() >= -2.0
    assert by_key["baseline_shift"].c_value.max() <= 2.0
    assert set(np.unique(by_key["autocorrelation"].f_value).astype(int).tolist()) == {1, 2, 3, 5}


def test_reduce_embeddings_shapes() -> None:
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(24, 16)).astype(np.float32)
    pca, tsne, metrics = reduce_embeddings(embeddings, seed=1)
    assert pca.shape == (24, 2)
    assert tsne.shape == (24, 2)
    assert "pca_explained_variance_ratio_sum" in metrics
    assert "trustworthiness" in metrics

