from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from p3_ssl.data import SSLManifestDataset


def test_ssl_manifest_dataset_full_window_4096_shapes(tmp_path: Path) -> None:
    signal_path = tmp_path / "signal.npy"
    label_path = tmp_path / "label.txt"
    manifest_path = tmp_path / "manifest.csv"
    np.save(signal_path, np.linspace(-1.0, 1.0, 16384, dtype=np.float32))
    label_path.write_text("0 0.5 0.125\n")
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "id", "signal_path", "label_path"])
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "id": "sample",
                "signal_path": str(signal_path),
                "label_path": str(label_path),
            }
        )

    dataset = SSLManifestDataset(
        manifest_csv=manifest_path,
        split="train",
        input_length_raw=16384,
        decimation_factor=4,
        input_length_ssl=4096,
        patch_size=4,
        patch_stride=4,
        min_block_length=24,
        max_block_length=128,
        seed=123,
    )
    item = dataset[0]

    assert tuple(item["signal"].shape) == (1, 4096)
    assert tuple(item["target_time_mask"].shape) == (4096,)
    assert tuple(item["hidden_time_mask"].shape) == (4096,)
    assert tuple(item["token_time_mask"].shape) == (4096,)
    assert tuple(item["event_mask"].shape) == (4096,)
    assert tuple(item["token_mask"].shape) == (1024,)
