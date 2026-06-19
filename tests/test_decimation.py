from __future__ import annotations

import numpy as np

from p3_ssl.decimation import crop_or_pad, decimate_signal, normalize_signal


def test_decimate_signal_mean() -> None:
    x = np.arange(16, dtype=np.float32)
    y = decimate_signal(x, factor=4, method="mean")
    np.testing.assert_allclose(y, np.array([1.5, 5.5, 9.5, 13.5], dtype=np.float32))


def test_decimate_signal_stride() -> None:
    x = np.arange(16, dtype=np.float32)
    y = decimate_signal(x, factor=4, method="stride")
    np.testing.assert_allclose(y, np.array([0, 4, 8, 12], dtype=np.float32))


def test_crop_or_pad_center() -> None:
    x = np.arange(6, dtype=np.float32)
    np.testing.assert_allclose(crop_or_pad(x, 4), np.array([1, 2, 3, 4], dtype=np.float32))
    assert crop_or_pad(x, 8).shape == (8,)


def test_normalize_window_zscore() -> None:
    x = np.array([1, 2, 3], dtype=np.float32)
    y = normalize_signal(x)
    assert abs(float(y.mean())) < 1.0e-6
    assert abs(float(y.std()) - 1.0) < 1.0e-6
