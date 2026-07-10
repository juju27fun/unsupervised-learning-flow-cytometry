from __future__ import annotations

import json

import numpy as np

from p3_ssl.serialization import json_safe


def test_json_safe_replaces_nonfinite_values() -> None:
    payload = {
        "finite": np.float32(1.5),
        "nan": float("nan"),
        "inf": float("inf"),
        "array": np.asarray([1.0, np.nan], dtype=np.float32),
    }
    safe = json_safe(payload)
    text = json.dumps(safe, allow_nan=False)
    assert "NaN" not in text
    assert safe["nan"] is None
    assert safe["inf"] is None
    assert safe["array"][1] is None
