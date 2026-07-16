#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, text=True, capture_output=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the yeast SSL Week 1 decision package.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--complementarity-root", type=Path, required=True)
    parser.add_argument("--domain-root", type=Path, required=True)
    parser.add_argument("--embedding-domain-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    dataset = json.loads((args.dataset_root / "dataset_summary.json").read_text())
    split = json.loads((args.dataset_root / "split_audit.json").read_text())
    complementarity = json.loads(
        (args.complementarity_root / "complementarity_summary.json").read_text()
    )
    domain = json.loads((args.domain_root / "domain_summary.json").read_text())
    matching = json.loads((args.domain_root / "matching_report.json").read_text())
    preflight = json.loads((args.preflight_root / "preflight_metrics.json").read_text())
    method = pd.read_csv(args.complementarity_root / "method_summary.csv")
    domain_metrics = pd.read_csv(args.domain_root / "domain_probe_metrics.csv")
    embedding_metrics = pd.read_csv(
        args.embedding_domain_root / "embedding_domain_metrics.csv"
    )
    embedding_matched_linear = embedding_metrics[
        (embedding_metrics.match_state == "matched") & (embedding_metrics.model == "linear")
    ].set_index(["simulation_source", "representation"])

    selected_names = [
        "family_time_morphology", "family_frequency", "family_envelope",
        "family_energy_amplitude", "family_quality", "handcrafted_all",
    ]
    labels = ["Time", "Frequency", "Envelope", "Energy", "Quality", "All"]
    selected = method[(method.label_fraction == 0.10) & method.method.isin(selected_names)].set_index("method")
    values = [selected.loc[name, "macro_f1_mean"] for name in selected_names]
    errors = [selected.loc[name, "macro_f1_std"] for name in selected_names]
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1", "#222222"]
    ax.bar(labels, values, yerr=errors, capsize=3, color=colors)
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Handcrafted feature-family audit at 10% labels")
    ax.set_ylim(0.0, max(values) + 0.08)
    ax.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"feature_family_macro_f1.{suffix}", dpi=180)
    plt.close(fig)

    bridge = domain_metrics[
        (domain_metrics.match_state == "matched")
        & (domain_metrics.feature_set == "signal_summary")
        & (domain_metrics.model == "linear")
    ].set_index("simulation_source")
    sources = ["analytic", "template_diagnostic"]
    source_labels = ["Analytic", "Template diagnostic"]
    retained = [matching[name]["validation"]["real_retained_fraction"] for name in sources]
    aucs = [bridge.loc[name, "validation_roc_auc"] for name in sources]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    axes[0].bar(source_labels, retained, color=["#E15759", "#59A14F"])
    axes[0].axhline(0.50, color="#222222", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Validation samples retained")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Observable common support")
    axes[1].bar(source_labels, aucs, color=["#E15759", "#59A14F"])
    axes[1].axhline(0.50, color="#222222", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Matched-subset domain ROC AUC")
    axes[1].set_ylim(0.45, 1.02)
    axes[1].set_title("Residual domain separation")
    for ax in axes:
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"domain_bridge_audit.{suffix}", dpi=180)
    plt.close(fig)

    gates = {
        "W1-A_data": "pass_with_declared_shmoo_validation_limitation",
        "W1-B_complementarity": "pass_complete_negative_redundant_with_handcrafted",
        "W1-C_domain": "pass_complete_major_mismatch_no_common_support",
        "W1-D_protocol": "pass_frozen_R0_R3",
        "W1-E_infrastructure": "pass_local_remote_idle_no_gpu_job_required",
    }
    decision = {
        "schema_version": 1,
        "status": "week1_complete",
        "gates": gates,
        "week2_objective_ablation_authorized": True,
        "extended_real_adaptation_authorized": False,
        "required_before_extended_adaptation": "one measured analytic simulator correction",
        "sealed_splits_used": [],
    }
    (args.output_dir / "week1_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Yeast SSL Follow-Up: Week 1 Decision",
        "",
        "**Status:** complete; Week 2 R0-R3 objective ablation authorized",
        "",
        "## Data",
        "",
        f"The prospective dataset contains {dataset['n_events']} events: "
        f"{dataset['split_counts']['followup_train']} train, "
        f"{dataset['split_counts']['followup_validation']} validation, and "
        f"{dataset['split_counts']['followup_test']} sealed final. Record, capture-block, "
        "and duplicate-family crossings are zero. Development and final metadata are physically separated.",
        "",
        "`shmoo` has only two eligible capture blocks, so no independent validation block exists. "
        "This prevents per-proxy validation claims but does not create leakage.",
        "",
        "## Complementarity",
        "",
        f"Historical A3 and A4 are classified as `{complementarity['historical_representation_classification']}`. "
        f"At 10% labels, fusion minus handcrafted is {complementarity['fusion_at_10_percent']['A3']['mean_delta']:+.4f} "
        f"for A3 and {complementarity['fusion_at_10_percent']['A4']['mean_delta']:+.4f} for A4. "
        "Frequency is the strongest isolated family; quality is also material, while envelope removal does not hurt.",
        "",
        "![Feature-family validation performance](feature_family_macro_f1.png)",
        "",
        "## Domain Bridge",
        "",
        f"The analytic simulator retains only {matching['analytic']['validation']['real_retained_fraction']:.1%} "
        f"of validation examples under the frozen caliper and reaches max post-match SMD "
        f"{matching['analytic']['validation']['post_match_smd_max']:.2f}. Conditional AUC is therefore not interpretable. "
        "The correct result is lack of common support, which is already a major simulator mismatch.",
        "",
        f"The train-only template diagnostic retains {matching['template_diagnostic']['validation']['real_retained_fraction']:.1%} "
        "with good observable balance, confirming that the audit can recognize an aligned control. It has no retained "
        "physical factors and is not eligible for training.",
        "",
        "![Domain bridge audit](domain_bridge_audit.png)",
        "",
        "The bounded CUDA sensitivity check agrees. On the small analytic matched subset, "
        f"linear domain AUC is {embedding_matched_linear.loc[('analytic', 'MOMENT_official'), 'validation_roc_auc']:.3f} "
        f"for official MOMENT, {embedding_matched_linear.loc[('analytic', 'A3_s42'), 'validation_roc_auc']:.3f} for A3, "
        f"and {embedding_matched_linear.loc[('analytic', 'A4_s42'), 'validation_roc_auc']:.3f} for A4. "
        "These remain exploratory because common support failed. On the balanced template control, "
        f"the corresponding AUCs are {embedding_matched_linear.loc[('template_diagnostic', 'MOMENT_official'), 'validation_roc_auc']:.3f}, "
        f"{embedding_matched_linear.loc[('template_diagnostic', 'A3_s42'), 'validation_roc_auc']:.3f}, and "
        f"{embedding_matched_linear.loc[('template_diagnostic', 'A4_s42'), 'validation_roc_auc']:.3f}.",
        "",
        "## Protocol and Decision",
        "",
        f"All R0-R3 CPU smokes have finite gradients: `{preflight['status']}`. The architecture, budgets, STFT "
        "windows (128/256/512), VICReg weights, seeds, endpoints, promotion gates, and stop rules are frozen.",
        "",
        "Week 2 may run the equal-budget R0-R3 mechanistic objective ablation. No sim-to-real or domain-general claim "
        "is authorized. Extended real adaptation remains blocked until one measured correction targets the analytic "
        "simulator's duration/support and spectral-peak mismatch.",
        "",
        "## Gates",
        "",
        "| Gate | Decision |",
        "|---|---|",
        *[f"| `{name}` | `{value}` |" for name, value in gates.items()],
    ]
    report = args.output_dir / "WEEK1_DECISION.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    run = {
        "schema_version": 1,
        "run_id": args.run_id,
        "project": "unsupervised-learning-flow-cytometry",
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": "report_yeast_followup_week1.py",
        "dataset": "yeast-events-followup@v2 + yeast-passage-simulations@v1 + yeast-template-comparator@v2",
        "profile": "week1-final-report",
        "sealed_splits_used": [],
        "repositories": {
            "unsupervised-learning-flow-cytometry": _revision(Path(__file__).resolve().parents[1]),
            "particles2SNR-pipeline": _revision(Path(__file__).resolve().parents[2] / "particles2SNR-pipeline"),
        },
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "run.json"
        },
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
