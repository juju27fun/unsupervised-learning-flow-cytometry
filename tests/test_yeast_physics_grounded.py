from __future__ import annotations

import numpy as np
import pytest

from p3_ssl.yeast_physics_grounded import (
    BACKGROUND_PROVENANCES,
    SPLIT_ID,
    build_capture_block_80_20_split,
)


def _population() -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    rows: list[dict[str, str]] = []
    labels: list[int] = []
    specs = [
        (0, "background", "budding", 8),
        (0, "background", "mix", 8),
        (0, "background", "shmoo2", 8),
        (1, "budding", "budding", 12),
        (2, "mix", "mix", 12),
        (3, "shmoo", "shmoo2", 10),
        (3, "shmoo", "shmoo", 2),
    ]
    for class_id, class_name, source, block_count in specs:
        for block_number in range(block_count):
            block = f"{source}:{class_name}:block-{block_number:03d}"
            for item in range(3):
                rows.append(
                    {
                        "sample_id": f"{block}:{item}",
                        "record_id": f"record:{block}:{item}",
                        "capture_block_id": block,
                        "development_split": "development_train",
                        "class_name": class_name,
                        "source_group_original": source,
                    }
                )
                labels.append(class_id)
    array = np.asarray(labels, dtype=np.int64)
    return rows, array, np.arange(array.size, dtype=np.int64)


def test_capture_block_split_is_deterministic_disjoint_and_complete() -> None:
    rows, labels, eligible = _population()
    first = build_capture_block_80_20_split(rows, labels, eligible, candidates=512)
    second = build_capture_block_80_20_split(rows, labels, eligible, candidates=512)
    assert first.manifest["split_id"] == SPLIT_ID
    assert np.array_equal(first.train_core_indices, second.train_core_indices)
    assert np.array_equal(first.model_selection_indices, second.model_selection_indices)

    train_blocks = {rows[int(index)]["capture_block_id"] for index in first.train_core_indices}
    selection_blocks = {
        rows[int(index)]["capture_block_id"] for index in first.model_selection_indices
    }
    train_records = {rows[int(index)]["record_id"] for index in first.train_core_indices}
    selection_records = {
        rows[int(index)]["record_id"] for index in first.model_selection_indices
    }
    assert train_blocks.isdisjoint(selection_blocks)
    assert train_records.isdisjoint(selection_records)
    assert set(labels[first.train_core_indices]) == {0, 1, 2, 3}
    assert set(labels[first.model_selection_indices]) == {0, 1, 2, 3}
    for role in ("train_core", "model_selection"):
        assert set(first.manifest[role]["background_source_counts"]) == set(BACKGROUND_PROVENANCES)
        assert len(first.manifest[role]["shmoo1_capture_blocks"]) == 1
    assert first.manifest["external_holdout_status"] == "closed"


def test_capture_block_split_refuses_development_validation() -> None:
    rows, labels, eligible = _population()
    rows[0]["development_split"] = "development_validation"
    with pytest.raises(ValueError, match="only use development_train"):
        build_capture_block_80_20_split(rows, labels, eligible, candidates=32)


def test_capture_block_split_requires_the_two_shmoo1_blocks() -> None:
    rows, labels, eligible = _population()
    for row in rows:
        if row["source_group_original"] == "shmoo":
            row["source_group_original"] = "shmoo2"
    with pytest.raises(ValueError, match="exactly two shmoo1"):
        build_capture_block_80_20_split(rows, labels, eligible, candidates=32)
