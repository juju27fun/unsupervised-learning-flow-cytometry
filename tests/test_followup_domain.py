from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from p3_ssl.followup_domain import fit_domain_probe, matched_pairs, signal_observables


def test_signal_observables_are_finite() -> None:
    time = np.arange(512) / 1_000_000.0
    signals = np.stack(
        [np.exp(-0.5 * ((time - 0.00025) / 0.00005) ** 2) * np.cos(2 * np.pi * f * time) for f in (10_000, 20_000)]
    )
    values = signal_observables(signals)
    assert values.shape == (2, 6)
    assert np.isfinite(values).all()


def test_matching_reduces_to_one_to_one_common_support() -> None:
    real = np.asarray([[0.0], [1.0], [10.0]])
    synthetic = np.asarray([[0.1], [1.1], [-10.0]])
    scaler = StandardScaler().fit(np.concatenate([real[:2], synthetic[:2]]))
    left, right, report = matched_pairs(real, synthetic, scaler=scaler, caliper=0.5)
    assert len(left) == len(set(left.tolist())) == 2
    assert len(right) == len(set(right.tolist())) == 2
    assert report["n_pairs"] == 2


def test_domain_probe_has_chance_and_separable_controls() -> None:
    rng = np.random.default_rng(3)
    real_train = rng.normal(size=(300, 4))
    synthetic_train = rng.normal(size=(300, 4))
    real_validation = rng.normal(size=(200, 4))
    synthetic_validation = rng.normal(size=(200, 4))
    chance = fit_domain_probe(
        real_train, synthetic_train, real_validation, synthetic_validation,
        feature_names=["a", "b", "c", "d"], model="linear"
    )
    separated = fit_domain_probe(
        real_train, synthetic_train + 5.0, real_validation, synthetic_validation + 5.0,
        feature_names=["a", "b", "c", "d"], model="linear"
    )
    assert 0.35 < chance.roc_auc < 0.65
    assert separated.roc_auc > 0.99
