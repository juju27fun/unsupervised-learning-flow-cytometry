from __future__ import annotations

from pathlib import Path

from p3_ssl.config import load_config
from p3_ssl.followup_reporting import plot_week2, summarize_week2, write_decision_markdown


CONFIG_PATH = Path(__file__).parents[1] / "configs/yeast_ssl_followup_week2_v1.yaml"


def _payload() -> dict:
    checkpoints = {}
    probes = []
    ranks = {"R0": 2.0, "R1": 3.0, "R2": 4.0, "R3": 5.0}
    offsets = {"R0": 0.00, "R1": 0.01, "R2": 0.02, "R3": 0.05}
    for cell in ("R0", "R1", "R2", "R3"):
        for representation_seed in (42, 43, 44):
            name = f"{cell.lower()}_s{representation_seed}"
            checkpoints[name] = {
                "cell": cell,
                "seed": representation_seed,
                "training_convergence": {"converged": True},
                "training_runtime": {"wall_seconds": 10.0, "optimizer_steps": 20},
                "real_validation_embedding_health": {
                    "effective_rank": ranks[cell],
                    "mean_dimension_std": 0.2,
                    "mean_absolute_off_diagonal_covariance": 0.01,
                    "mean_off_diagonal_cosine_similarity": 0.1,
                },
                "mean_continuous_relative_mse_reduction": 0.30 + offsets[cell],
                "component_count_balanced_accuracy": 0.60 + offsets[cell],
                "cross_recording_retrieval": {"topk_label_purity": 0.50 + offsets[cell]},
            }
            for fraction in (0.01, 0.05, 0.10, 0.25, 1.00):
                for probe_seed in (42, 43, 44):
                    handcrafted = 0.45 + 0.1 * fraction
                    learned = 0.40 + 0.1 * fraction + offsets[cell]
                    fusion = handcrafted + 0.01 + offsets[cell]
                    for method, score in (
                        ("learned", learned),
                        ("handcrafted", handcrafted),
                        ("handcrafted_plus_learned", fusion),
                    ):
                        probes.append(
                            {
                                "cell": cell,
                                "representation_seed": representation_seed,
                                "method": method,
                                "probe": "linear",
                                "label_fraction": fraction,
                                "probe_seed": probe_seed,
                                "macro_f1": score,
                            }
                        )
    return {
        "protocol": "yeast-ssl-followup-week2-v1-20260716",
        "checkpoint_results": checkpoints,
        "probe_results": probes,
        "sealed_splits_used": [],
    }


def test_week2_gate_passes_only_complete_positive_matrix() -> None:
    summary = summarize_week2(_payload(), load_config(CONFIG_PATH))
    assert summary["gate"]["effective_rank"]["ratio"] == 2.5
    assert summary["gate"]["r3_promoted"] is True
    assert summary["week3_quality_adaptation_authorized"] is True


def test_week2_gate_rejects_rank_failure() -> None:
    payload = _payload()
    for metadata in payload["checkpoint_results"].values():
        if metadata["cell"] == "R3":
            metadata["real_validation_embedding_health"]["effective_rank"] = 3.0
    summary = summarize_week2(payload, load_config(CONFIG_PATH))
    assert summary["gate"]["effective_rank"]["pass"] is False
    assert summary["gate"]["r3_promoted"] is False


def test_week2_report_writes_publication_figure_and_decision(tmp_path: Path) -> None:
    summary = summarize_week2(_payload(), load_config(CONFIG_PATH))
    outputs = plot_week2(summary, tmp_path / "comparison")
    decision = tmp_path / "decision.md"
    write_decision_markdown(decision, summary)
    assert {path.suffix for path in outputs} == {".png", ".pdf"}
    assert all(path.stat().st_size > 0 for path in outputs)
    assert "PROMOTE R3" in decision.read_text(encoding="utf-8")
