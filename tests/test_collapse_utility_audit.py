import numpy as np

from p3_ssl.collapse_utility_audit import classify_supplement, validation_block_support


def test_validation_block_support_exposes_single_block_class() -> None:
    rows = [
        {"capture_block_id": "a", "record_id": "r0"},
        {"capture_block_id": "b", "record_id": "r1"},
        {"capture_block_id": "c", "record_id": "r2"},
    ]
    result = validation_block_support(
        rows,
        np.asarray([0, 0, 1]),
        np.asarray([0, 1, 2]),
        ["first", "second"],
    )
    assert result["n_blocks"] == 3
    assert result["blocks_per_class"] == {"first": 2, "second": 1}
    assert result["minimum_blocks_per_class"] == 1
    assert result["class_pure_blocks"] is True


def _comparison(gain: float, baseline: float = 0.30) -> dict[str, object]:
    return {
        "gain": gain,
        "baseline_mean_macro_f1": baseline,
        "paired_interval": [gain - 0.01, gain + 0.01],
    }


def test_supplement_confirms_rejection_without_complementarity() -> None:
    result = classify_supplement(
        source_decision="reject_mask_only_rescue",
        c1_vs_handcrafted={
            "handcrafted_signal": _comparison(-0.02, 0.31),
            "handcrafted_full": _comparison(-0.08, 0.37),
        },
        fusion_vs_handcrafted=_comparison(0.01, 0.37),
        all_probes_converged=True,
        minimum_blocks_per_class=1,
        required_blocks_per_class=5,
        minimum_complementarity_gain=0.03,
    )
    assert result["decision"] == "mask_only_rejection_confirmed_no_complementarity"
    assert result["strongest_handcrafted"] == "handcrafted_full"
    assert result["inferential_status"].startswith("descriptive_")
    assert result["additional_representation_seeds_authorized"] is False


def test_supplement_does_not_call_weak_support_confirmatory() -> None:
    result = classify_supplement(
        source_decision="reject_mask_only_rescue",
        c1_vs_handcrafted={"handcrafted_full": _comparison(-0.05, 0.35)},
        fusion_vs_handcrafted=_comparison(0.04, 0.35),
        all_probes_converged=True,
        minimum_blocks_per_class=1,
        required_blocks_per_class=5,
        minimum_complementarity_gain=0.03,
    )
    assert result["decision"] == "mask_only_rejected_exploratory_complementarity_signal"
    assert result["checks"]["independent_block_support_sufficient"] is False
