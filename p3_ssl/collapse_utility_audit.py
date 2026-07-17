from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def validation_block_support(
    rows: list[dict[str, str]],
    labels: np.ndarray,
    validation_indices: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Summarize independent validation support for proxy-label inference."""
    by_class: dict[str, set[str]] = defaultdict(set)
    all_blocks: set[str] = set()
    for index in np.asarray(validation_indices, dtype=np.int64):
        row = rows[int(index)]
        block = row.get("capture_block_id") or row["record_id"]
        class_name = class_names[int(labels[int(index)])]
        by_class[class_name].add(block)
        all_blocks.add(block)
    counts = {name: len(by_class[name]) for name in class_names}
    return {
        "group_unit": "capture_block_id_fallback_record_id",
        "n_blocks": len(all_blocks),
        "blocks_per_class": counts,
        "minimum_blocks_per_class": min(counts.values()),
        "class_pure_blocks": all(
            len(
                {
                    class_names[int(labels[int(index)])]
                    for index in validation_indices
                    if (rows[int(index)].get("capture_block_id") or rows[int(index)]["record_id"])
                    == block
                }
            )
            == 1
            for block in all_blocks
        ),
    }


def classify_supplement(
    *,
    source_decision: str,
    c1_vs_handcrafted: dict[str, dict[str, Any]],
    fusion_vs_handcrafted: dict[str, Any],
    all_probes_converged: bool,
    minimum_blocks_per_class: int,
    required_blocks_per_class: int,
    minimum_complementarity_gain: float,
) -> dict[str, Any]:
    if source_decision != "reject_mask_only_rescue":
        raise ValueError("The supplement requires the frozen mask-only rejection")
    strongest = max(
        c1_vs_handcrafted,
        key=lambda name: float(c1_vs_handcrafted[name]["baseline_mean_macro_f1"]),
    )
    support_sufficient = minimum_blocks_per_class >= required_blocks_per_class
    complementarity_point = float(fusion_vs_handcrafted["gain"])
    complementarity_interval = float(fusion_vs_handcrafted["paired_interval"][0]) > 0.0
    checks = {
        "all_primary_probes_converged": all_probes_converged,
        "independent_block_support_sufficient": support_sufficient,
        "c1_beats_strongest_handcrafted_point_estimate": (
            float(c1_vs_handcrafted[strongest]["gain"]) > 0.0
        ),
        "fusion_practical_gain": complementarity_point >= minimum_complementarity_gain,
        "fusion_interval_lower_above_zero": complementarity_interval,
    }
    if not all_probes_converged:
        decision = "invalid_supplement_nonconverged_probe"
    elif checks["fusion_practical_gain"] and complementarity_interval and support_sufficient:
        decision = "mask_only_rejected_but_confirmatory_complementarity_candidate"
    elif checks["fusion_practical_gain"]:
        decision = "mask_only_rejected_exploratory_complementarity_signal"
    else:
        decision = "mask_only_rejection_confirmed_no_complementarity"
    return {
        "decision": decision,
        "strongest_handcrafted": strongest,
        "checks": checks,
        "inferential_status": (
            "eligible_for_group_interval_interpretation"
            if support_sufficient
            else "descriptive_insufficient_independent_blocks_per_class"
        ),
        "additional_representation_seeds_authorized": False,
    }
