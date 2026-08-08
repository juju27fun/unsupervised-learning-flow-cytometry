from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from p3_ssl.yeast_4class_classifier import CLASS_NAMES


SPLIT_ID = "yeast-physics-grounded-v2-block-80-20-s20260807-r1"
BACKGROUND_PROVENANCES = ("budding", "mix", "shmoo2")
SHMOO1_SOURCE_GROUP = "shmoo"


@dataclass(frozen=True)
class PhysicsGroundedSplit:
    train_core_indices: np.ndarray
    model_selection_indices: np.ndarray
    manifest: dict[str, Any]


def _count_classes(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    values = np.bincount(labels[indices], minlength=len(CLASS_NAMES))
    return {name: int(values[class_id]) for class_id, name in enumerate(CLASS_NAMES)}


def _count_values(
    rows: Sequence[dict[str, str]],
    indices: np.ndarray,
    *,
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for index in indices:
        value = rows[int(index)][field]
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _role_summary(
    rows: Sequence[dict[str, str]],
    labels: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    background = indices[labels[indices] == 0]
    events = indices[labels[indices] != 0]
    return {
        "rows": int(indices.size),
        "capture_blocks": len({rows[int(index)]["capture_block_id"] for index in indices}),
        "records": len({rows[int(index)]["record_id"] for index in indices}),
        "class_counts": _count_classes(labels, indices),
        "background_source_counts": _count_values(
            rows, background, field="source_group_original"
        ),
        "event_source_counts": _count_values(
            rows, events, field="source_group_original"
        ),
        "shmoo1_capture_blocks": sorted(
            {
                rows[int(index)]["capture_block_id"]
                for index in events
                if rows[int(index)]["source_group_original"] == SHMOO1_SOURCE_GROUP
            }
        ),
    }


def build_capture_block_80_20_split(
    rows: Sequence[dict[str, str]],
    labels: np.ndarray,
    eligible_indices: Sequence[int],
    *,
    seed: int = 20260807,
    model_selection_fraction: float = 0.20,
    candidates: int = 4096,
) -> PhysicsGroundedSplit:
    """Freeze a deterministic, capture-block-disjoint inner development split.

    The function deliberately refuses rows outside ``development_train``. It
    searches a fixed set of GroupShuffleSplit candidates and applies the hard
    population constraints before minimizing class-fraction errors
    lexicographically, followed by the global fraction error.
    """

    eligible = np.asarray(eligible_indices, dtype=np.int64)
    labels_array = np.asarray(labels, dtype=np.int64)
    if eligible.size == 0:
        raise ValueError("eligible_indices must not be empty")
    if not 0.0 < model_selection_fraction < 1.0:
        raise ValueError("model_selection_fraction must be between zero and one")
    if candidates <= 0:
        raise ValueError("candidates must be positive")
    if np.any(eligible < 0) or np.any(eligible >= len(rows)):
        raise ValueError("eligible_indices contains an out-of-range row")
    forbidden = [
        rows[int(index)]["sample_id"]
        for index in eligible
        if rows[int(index)].get("development_split") != "development_train"
    ]
    if forbidden:
        raise ValueError(
            "Physics-grounded phases may only use development_train; "
            f"found forbidden sample {forbidden[0]}"
        )

    y = labels_array[eligible]
    groups = np.asarray([rows[int(index)]["capture_block_id"] for index in eligible])
    if any(not value for value in groups.tolist()):
        raise ValueError("Every eligible row must define capture_block_id")
    if set(y.tolist()) != set(range(len(CLASS_NAMES))):
        raise ValueError("Eligible population must contain all four classes")

    shmoo1_blocks = {
        rows[int(index)]["capture_block_id"]
        for index in eligible
        if labels_array[int(index)] != 0
        and rows[int(index)]["source_group_original"] == SHMOO1_SOURCE_GROUP
    }
    if len(shmoo1_blocks) != 2:
        raise ValueError(
            "The frozen method requires exactly two shmoo1 event blocks; "
            f"found {len(shmoo1_blocks)}"
        )

    total_class_counts = np.bincount(y, minlength=len(CLASS_NAMES)).astype(np.float64)
    best: tuple[tuple[float, ...], int, np.ndarray, np.ndarray] | None = None

    for offset in range(candidates):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=model_selection_fraction,
            random_state=seed + offset,
        )
        train_local, selection_local = next(splitter.split(eligible, y, groups))
        train_indices = eligible[train_local]
        selection_indices = eligible[selection_local]
        train_labels = labels_array[train_indices]
        selection_labels = labels_array[selection_indices]
        if set(train_labels.tolist()) != set(range(len(CLASS_NAMES))):
            continue
        if set(selection_labels.tolist()) != set(range(len(CLASS_NAMES))):
            continue

        background_ok = True
        for indices in (train_indices, selection_indices):
            background_sources = {
                rows[int(index)]["source_group_original"]
                for index in indices
                if labels_array[int(index)] == 0
            }
            if not set(BACKGROUND_PROVENANCES).issubset(background_sources):
                background_ok = False
                break
        if not background_ok:
            continue

        train_blocks = {rows[int(index)]["capture_block_id"] for index in train_indices}
        selection_blocks = {
            rows[int(index)]["capture_block_id"] for index in selection_indices
        }
        if len(shmoo1_blocks & train_blocks) != 1 or len(shmoo1_blocks & selection_blocks) != 1:
            continue

        train_records = {rows[int(index)]["record_id"] for index in train_indices}
        selection_records = {rows[int(index)]["record_id"] for index in selection_indices}
        if train_records & selection_records:
            continue

        selection_class_counts = np.bincount(
            selection_labels, minlength=len(CLASS_NAMES)
        ).astype(np.float64)
        class_fraction_errors = np.abs(
            selection_class_counts / total_class_counts - model_selection_fraction
        )
        global_fraction_error = abs(
            selection_indices.size / eligible.size - model_selection_fraction
        )
        score = tuple(float(value) for value in class_fraction_errors) + (
            float(global_fraction_error),
            float(offset),
        )
        if best is None or score < best[0]:
            best = (score, offset, train_indices, selection_indices)

    if best is None:
        raise ValueError(
            "No capture-block-disjoint split satisfied the frozen population constraints"
        )

    score, selected_offset, train_indices, selection_indices = best
    train_indices = np.sort(train_indices)
    selection_indices = np.sort(selection_indices)
    train_blocks = {rows[int(index)]["capture_block_id"] for index in train_indices}
    selection_blocks = {rows[int(index)]["capture_block_id"] for index in selection_indices}
    train_records = {rows[int(index)]["record_id"] for index in train_indices}
    selection_records = {rows[int(index)]["record_id"] for index in selection_indices}
    if train_blocks & selection_blocks:
        raise AssertionError("capture_block_id leakage in frozen split")
    if train_records & selection_records:
        raise AssertionError("record_id leakage in frozen split")

    manifest = {
        "schema_version": 1,
        "split_id": SPLIT_ID,
        "seed": int(seed),
        "source_partition": "development_train",
        "external_holdout": "development_validation",
        "external_holdout_status": "closed",
        "group_key": "capture_block_id",
        "model_selection_fraction_requested": float(model_selection_fraction),
        "selection_candidates": int(candidates),
        "selected_candidate_offset": int(selected_offset),
        "selection_rule": (
            "hard population constraints; then lexicographic absolute model-selection "
            "fraction error for background, budding, mix and shmoo; then global fraction error"
        ),
        "selection_score": {
            "class_fraction_errors": {
                name: score[class_id] for class_id, name in enumerate(CLASS_NAMES)
            },
            "global_fraction_error": score[len(CLASS_NAMES)],
        },
        "sealed_holdout_accessed": False,
        "train_core": _role_summary(rows, labels_array, train_indices),
        "model_selection": _role_summary(rows, labels_array, selection_indices),
    }
    return PhysicsGroundedSplit(train_indices, selection_indices, manifest)
