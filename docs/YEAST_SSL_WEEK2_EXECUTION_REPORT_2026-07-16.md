# Yeast SSL Follow-Up: Week 2 Execution Report

**Date:** 2026-07-16  
**Status:** complete  
**R3 promotion:** rejected  
**Week 3 quality adaptation:** not authorized  
**Week 3 targeted simulator correction:** required

This report closes the Week 2 ablation defined in
[`YEAST_SSL_ONE_MONTH_FOLLOWUP_PLAN_2026-07-16.md`](YEAST_SSL_ONE_MONTH_FOLLOWUP_PLAN_2026-07-16.md).
All selection and interpretation use `followup_validation`. Neither
`followup_test` nor the exhausted v1 `in_session_test` was opened.

## 1. Experiment

R0-R3 used the same 96-dimensional patch transformer, datasets, optimizer,
batch size, 20 simulation epochs, 10 real-adaptation epochs, 4,710 optimizer
steps, and representation seeds 42/43/44. The only differences were the frozen
objective terms:

| Cell | Time reconstruction | Spectral reconstruction | VICReg |
|---|:---:|:---:|:---:|
| R0 | yes | no | no |
| R1 | yes | no | yes |
| R2 | yes | yes | no |
| R3 | yes | yes | yes |

The spectral term used log-magnitude STFT windows of 128, 256, and 512 samples.
The deterministic CUDA correction (`center=false` and strict deterministic
algorithms) was made after the first smoke exposed nondeterministic PyTorch
kernels and before any full-run outcome was inspected. The corrected R0-R3
smoke passed with finite losses and gradients.

The full pfcalcul Jupyter execution produced all 12 cells. A runner-liveness
monitor initially classified the queue item as failed after the runner process
disappeared, but the last checkpoint, evaluation, and report had completed.
The idempotent resume job reused all complete outputs and returned code 0; no
cell was trained twice.

## 2. Main Results

| Cell | Converged seeds | Effective rank | Macro F1, 10% labels | Retained-factor gain | Fusion delta | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 2/3 | 3.147 | 0.315 | 0.405 | -0.018 | 7.35 min |
| R1 | 2/3 | 4.086 | 0.335 | 0.413 | -0.004 | 8.63 min |
| R2 | 2/3 | 2.486 | 0.314 | 0.438 | -0.008 | 7.75 min |
| R3 | 2/3 | 3.050 | 0.323 | 0.464 | -0.006 | 8.79 min |

Values are medians over representation seeds except the convergence count.
Every run completed the same budget with finite values. Seed 44 failed the
frozen final-three-versus-first-three adaptation-loss trend in all four cells;
this is shared seed sensitivity rather than an R3-specific failure, but the
preregistered R3 gate still requires 3/3 convergence.

![Week 2 R0-R3 ablation](../../artifacts/unsupervised-learning-flow-cytometry/reports/yeast-followup-week2-full-v1-report-v3/week2_r0_r3_comparison.png)

## 3. Frozen R3 Gate

All paired effects below are R3 minus R0 with a 95% hierarchical bootstrap
interval.

| Criterion | Result | Pass |
|---|---:|:---:|
| Effective-rank ratio, minimum 2.0 | 0.969 | no |
| Continuous-factor recovery not degraded | +0.055 `[+0.044, +0.070]` | yes |
| Component balanced accuracy not degraded | +0.013 `[+0.013, +0.014]` | yes |
| Macro F1 at 10% not degraded | +0.000 `[-0.021, +0.019]` | yes |
| One primary metric strictly improved | none | no |
| All R3 seeds converged | 2/3 | no |

The other primary effects were label-efficiency AUC `-0.006`
`[-0.014, +0.004]` and handcrafted-fusion delta `+0.006`
`[-0.006, +0.019]`. No primary interval excludes zero on the positive side.
R3 therefore fails three independent requirements: rank, positive downstream
or complementarity evidence, and all-seed convergence.

## 4. Mechanistic Interpretation

- **H1, collapse control:** R1 raises median effective rank from 3.147 to 4.086
  and lowers cosine concentration, but remains far below the required 2x gain
  and does not establish downstream or complementarity improvement.
- **H2, spectral reconstruction:** R2 improves retained-factor recovery but
  reduces effective rank and leaves 10%-label macro F1 essentially at R0.
- **Combined R3:** R3 gives the strongest physical-factor recovery but does not
  combine the partial R1 and R2 gains into useful geometry or predictive value.
- **Complementarity:** every median fusion delta is negative. Learned features
  still do not add stable information to the stronger handcrafted baseline.
- **Shortcut behavior:** median cross-recording acquisition purity and quality
  purity remain 1.0 for every cell. Objective correction did not remove the
  acquisition/quality shortcut.
- **Cost:** median runtime rises by about 18% for R1 and 20% for R3 relative to
  R0, while R3 peak CUDA memory rises from about 0.60 to 0.86 GiB.

The result is not that spectral or variance losses learn nothing. They improve
different mechanistic diagnostics, but their combination does not satisfy the
prospective selection criteria. This is a useful negative boundary result, not
a reason to retune losses against validation.

## 5. Evaluation Integrity and Limits

The evaluator contains 12 checkpoints and 1,080 probe rows: three methods,
linear and fixed-MLP probes, five label fractions, three probe seeds, and three
representation seeds for each cell. All 1,080 probes converged without warning.
All run and report manifests validate, contain no non-finite values, and record
`sealed_splits_used = []`.

The validation split has no independent `shmoo` capture block, so `shmoo`
recall is not estimable. Source-group labels remain acquisition-condition
proxies, not morphology labels. The study supports no acquisition-OOD,
biological, or general transformer-versus-CNN claim.

## 6. Decision and Next Work

Do not run quality-balanced R3 adaptation, a parameter-matched CNN extension,
another objective sweep, or the v2 final test. No representation passed the
validation gate, so opening `followup_test` would add no defensible selection
evidence.

Week 3 is restricted to the separately required measured simulator correction
from Week 1: change one documented simulator family targeting the observed
duration/support and spectral mismatch, then repeat the conditional bridge
audit on development data. This work diagnoses simulator adequacy; it does not
authorize a new representation search. If common support remains inadequate,
stop experimental work and proceed to the final study documentation with
`followup_test` still sealed.

## 7. Evidence

- Corrected decision report:
  [`WEEK2_DECISION.md`](../../artifacts/unsupervised-learning-flow-cytometry/reports/yeast-followup-week2-full-v1-report-v3/WEEK2_DECISION.md)
- Complete seed table:
  [`week2_seed_summary.csv`](../../artifacts/unsupervised-learning-flow-cytometry/reports/yeast-followup-week2-full-v1-report-v3/week2_seed_summary.csv)
- Gate and paired intervals:
  [`week2_decision.json`](../../artifacts/unsupervised-learning-flow-cytometry/reports/yeast-followup-week2-full-v1-report-v3/week2_decision.json)
- Full evaluation metrics:
  [`metrics.json`](../../artifacts/unsupervised-learning-flow-cytometry/runs/yeast-followup-week2-full-v1-evaluation/metrics.json)
- Frozen executable configuration:
  [`yeast_ssl_followup_week2_v1.yaml`](../configs/yeast_ssl_followup_week2_v1.yaml)

The provisional v1 and v2 report artifacts are retained for provenance. The v3
report fixes display-only `n/a` gate fields, limits positive candidates to the
frozen primary-metric list, and adds subgroup, shortcut, per-seed loss-component,
runtime, and memory tables; the scientific decision is unchanged.
