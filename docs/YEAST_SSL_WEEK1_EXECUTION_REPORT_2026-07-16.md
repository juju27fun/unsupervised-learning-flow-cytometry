# Yeast SSL Follow-Up: Week 1 Execution Report

**Date:** 2026-07-16  
**Status:** complete  
**Week 2 R0-R3 objective ablation:** authorized  
**Extended sim-to-real adaptation:** not authorized

This report closes the preregistered
[`YEAST_SSL_WEEK1_PLAN_2026-07-16.md`](YEAST_SSL_WEEK1_PLAN_2026-07-16.md).
It does not reopen the v1 final result. Neither the old `in_session_test` nor
the new `followup_test` was used.

## 1. Prospective Data Protocol

The active registered dataset is `yeast-events-followup@v2`. It contains only
events from the v1 `development_train` pool:

| Split | Events | Records | Capture blocks |
|---|---:|---:|---:|
| `followup_train` | 4,299 | 2,729 | 94 |
| `followup_validation` | 1,284 | 858 | 30 |
| `followup_test` | 1,426 | 911 | 31 |

Record, capture-block, and duplicate-family crossings are zero. Normalization
uses `followup_train` only. Development and final metadata are physically
separated, and ordinary loaders reject every old or new final split.

Only two eligible `shmoo` capture blocks remain. One is in train and one in the
sealed final split; no independent validation block can exist without leakage.
Week 2 therefore cannot make a per-`shmoo` validation claim.

Evidence:

- [`SPLIT_AUDIT.md`](../../datasets/processed/particles2SNR-pipeline/yeast-events-followup/v2/SPLIT_AUDIT.md)
- [`split_audit.json`](../../datasets/processed/particles2SNR-pipeline/yeast-events-followup/v2/split_audit.json)

## 2. Handcrafted Features and Complementarity

At 10% labels, all handcrafted features reach validation macro-F1
`0.3812 +/- 0.0145`.

| Family | Macro-F1 |
|---|---:|
| Frequency | 0.3081 |
| Time morphology | 0.2967 |
| Detector quality | 0.2844 |
| Energy/amplitude | 0.2244 |
| Envelope | 0.2092 |

Leave-one-family-out results identify frequency and quality as material.
Removing frequency lowers macro-F1 to `0.3515`; removing quality lowers it to
`0.3576`. Removing envelope does not hurt (`0.3829`). RMS alone is weak
(`0.1650`), so this is not a trivial amplitude-only shortcut.

Historical embeddings do not add residual validation information. At 10%
labels, fusion minus handcrafted is `-0.0099` for A3 and `-0.0112` for A4. The
best cluster-bootstrap upper confidence limits remain below `+0.03`; the fixed
MLP probe also finds no `+0.03` nonlinear gain. The classification is
`redundant_with_handcrafted`.

![Handcrafted feature-family audit](../../artifacts/unsupervised-learning-flow-cytometry/reports/yeast-followup-week1-v1/feature_family_macro_f1.png)

Evidence:

- [`probe_metrics.csv`](../../artifacts/unsupervised-learning-flow-cytometry/runs/yeast-followup-week1-complementarity-v2/probe_metrics.csv)
- [`fusion_cluster_bootstrap.csv`](../../artifacts/unsupervised-learning-flow-cytometry/runs/yeast-followup-week1-complementarity-v2/fusion_cluster_bootstrap.csv)
- [`complementarity_summary.json`](../../artifacts/unsupervised-learning-flow-cytometry/runs/yeast-followup-week1-complementarity-v2/complementarity_summary.json)

## 3. Simulation-Real Bridge

The earlier visual MOMENT overlap is reconciled with its domain AUC near
`0.99`: PCA/t-SNE overlap was never evidence of domain invariance.

Under the current 4.096 ms contract, the analytic simulator lacks observable
common support. Frozen matching retains `14.6%` of train and `16.2%` of
validation examples. Maximum post-match standardized mean difference is `1.95`
and `2.34`, above the frozen `0.25` limit. The matched-subset signal-summary AUC
is `1.00`, but is not a valid conditional estimate after the support failure.
The defensible result is **major simulator mismatch through lack of common
support**.

The train-only template diagnostic retains `82.8%` of validation samples with
maximum SMD `0.018`; its linear signal-summary AUC is `0.546`. This positive
control shows the audit recognizes an aligned source. It has no controlled
physical factors and is not eligible for SSL training.

![Conditional domain bridge audit](../../artifacts/unsupervised-learning-flow-cytometry/reports/yeast-followup-week1-v1/domain_bridge_audit.png)

Evidence:

- [`matching_report.json`](../../artifacts/unsupervised-learning-flow-cytometry/runs/yeast-followup-week1-domain-v3/matching_report.json)
- [`domain_probe_metrics.csv`](../../artifacts/unsupervised-learning-flow-cytometry/runs/yeast-followup-week1-domain-v3/domain_probe_metrics.csv)
- [`feature_importance.json`](../../artifacts/unsupervised-learning-flow-cytometry/runs/yeast-followup-week1-domain-v3/feature_importance.json)

## 4. Frozen R0-R3 Protocol

[`configs/yeast_ssl_followup_v2.yaml`](../configs/yeast_ssl_followup_v2.yaml)
freezes the common 96-dimensional patch transformer, equal budgets, seeds
42/43/44, access rules, endpoints, stop rules, and promotion criteria.

R0-R3 isolate time reconstruction, multi-resolution log-magnitude STFT
reconstruction at 128/256/512 samples, and VICReg variance/covariance control.
All CPU smokes produced finite losses and gradients. The weighted spectral term
is about `0.31` and VICReg about `0.047` on the fixed smoke batch, so neither
silently overwhelms time reconstruction.

Evidence:

- [`frozen_protocol.json`](../../artifacts/unsupervised-learning-flow-cytometry/runs/yeast-followup-week1-preflight-v2/frozen_protocol.json)
- [`preflight_metrics.json`](../../artifacts/unsupervised-learning-flow-cytometry/runs/yeast-followup-week1-preflight-v2/preflight_metrics.json)

## 5. Gates and Consequence

| Gate | Decision |
|---|---|
| W1-A data | Pass with declared missing `shmoo` validation block |
| W1-B complementarity | Pass; complete negative, redundant with handcrafted |
| W1-C domain | Pass; major mismatch, no analytic common support |
| W1-D protocol | Pass; R0-R3 frozen |
| W1-E infrastructure | Pass; tests/manifests pass, pfcalcul idle |

Week 2 may run the equal-budget R0-R3 objective ablation as a mechanistic test.
No domain-general or sim-to-real claim is authorized. Before extended real
adaptation, Week 3 must run one measured simulator correction targeting the
duration/support and spectral-peak mismatch. Broad domain-adversarial training
remains unjustified.

The complete manifested handoff is
[`WEEK1_DECISION.md`](../../artifacts/unsupervised-learning-flow-cytometry/reports/yeast-followup-week1-v1/WEEK1_DECISION.md).
