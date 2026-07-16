# Yeast SSL: One-Month Follow-Up Plan

**Date:** 2026-07-16

**Status:** working plan; freeze protocol v2 before the first new full training

**Duration:** four weeks

**Week 1 status:** complete. See
[`YEAST_SSL_WEEK1_EXECUTION_REPORT_2026-07-16.md`](YEAST_SSL_WEEK1_EXECUTION_REPORT_2026-07-16.md).

**Previous study:**
[`YEAST_SSL_REBUILT_STUDY_REPORT.md`](YEAST_SSL_REBUILT_STUDY_REPORT.md)

**Detailed Week 1 plan:**
[`YEAST_SSL_WEEK1_PLAN_2026-07-16.md`](YEAST_SSL_WEEK1_PLAN_2026-07-16.md)

## 1. Purpose

Protocol `yeast-ssl-rebuild-v1` is complete and frozen. It established that
physics supervision and real adaptation improve the custom representation, but
A4 remains below handcrafted signal features. The one-time v1
`in_session_test` is exhausted and must never be used for further model
selection.

The follow-up is a new mechanistic study, not an attempt to rescue v1 by tuning
against its final result. Its central question is:

> Can collapse-aware, time-frequency-sensitive self-supervised learning capture
> information that is complementary to handcrafted features under a new,
> strictly prospective within-acquisition protocol?

The month should produce either a validated improvement or a useful negative
boundary result. A positive result is not required for completion.

## 2. Evidence Already Established

The follow-up starts from these frozen findings:

- reviewed v7 extraction is adequate for the available acquisition;
- source folders are acquisition-condition proxies, not morphology labels;
- masked reconstruction beats interpolation but does not ensure useful geometry;
- A1/A2 embeddings are nearly collapsed;
- A3 physical supervision improves A2 and retained-factor recovery;
- A4 real adaptation improves A3, but by less than the practical-effect target;
- handcrafted features are the strongest current in-session representation;
- quality-stratum retrieval purity of `1.0` exposes a strong quality shortcut;
- simulation and reality remain nearly perfectly separable in A3/A4 embeddings;
- no acquisition-OOD or biological-generalization conclusion is available.

These results must not be re-tested until a desired interpretation appears.
They define the new hypotheses and controls.

## 3. Scope and Claim Boundary

The follow-up may claim only within-acquisition source-proxy representation
results. It may study physical recovery on held-out simulations. It may not
claim:

- morphology recognition;
- transfer to another acquisition;
- general transformer superiority over CNNs;
- biological discovery from latent-space visualization;
- simulator realism from PCA or t-SNE overlap alone.

The main practical comparison is learned features versus handcrafted features,
and especially whether their fusion contains complementary information.

## 4. Prospective Data Protocol

The old `in_session_test` is archival evidence. No v2 script may read it.

Before implementation, construct a new grouped split using only records that
belonged to v1 `development_train`:

| Split | Allowed use |
|---|---|
| `followup_train` | SSL, physics-supervised training, and probe fitting |
| `followup_validation` | Loss and method selection, thresholds, and stopping |
| `followup_test` | One-time v2 comparison after protocol freeze |

Requirements:

- split by duplicate-safe capture block before event extraction;
- keep every descendant event of a record in one split;
- preserve every source proxy where the block counts permit;
- fit preprocessing statistics on `followup_train` only;
- exclude `followup_test` signals from SSL, including unlabeled adaptation;
- register the split and derived dataset through `particles2SNR-pipeline`;
- include dataset IDs and revisions in every run manifest.

If the available v1 development-train blocks cannot support a representative
three-way split, stop and redesign the endpoint before training. Do not borrow
the old final test.

## 5. Primary and Mechanistic Hypotheses

| ID | Hypothesis | Falsifying result |
|---|---|---|
| H1 | Explicit variance/covariance regularization reduces embedding collapse | Effective rank does not materially improve, or downstream/physical information falls |
| H2 | Multi-resolution spectral reconstruction captures information missed by time-domain MSE | No validation gain over the identical time-only objective |
| H3 | The learned embedding adds information beyond handcrafted features | Handcrafted + embedding does not improve over handcrafted alone |
| H4 | Part of the measured simulation-real gap remains after matching observable physical distributions | A conditional domain probe falls close to chance |
| H5 | Quality-balanced real adaptation reduces the detector-quality shortcut | Quality separability remains unchanged or proxy/physical information is erased |

### Primary endpoint

At 10% labels, compare the frozen probe on:

```text
handcrafted + selected learned embedding
versus
handcrafted only
```

Promotion requires a mean macro-F1 improvement of at least `0.03` and a paired
hierarchical bootstrap interval excluding zero on the positive side. Learned-only
performance is secondary. This endpoint directly tests complementary value,
rather than requiring the network to rediscover every useful handcrafted
quantity.

## 6. Week 1: Diagnosis and Protocol Freeze

### 6.1 Handcrafted complementarity audit

Run only on the new train/validation protocol:

1. separate temporal, spectral, envelope, energy, and quality-related feature
   families;
2. ablate one family at a time;
3. compare handcrafted, A3, A4, handcrafted+A3, and handcrafted+A4;
4. use the same linear probe and a small fixed MLP probe as a sensitivity check;
5. report errors by source proxy, capture block, SNR, and quality stratum;
6. measure whether learned features add residual predictive information after
   conditioning on handcrafted features.

This audit decides which signal structures the SSL objective is failing to
represent. It is more informative than another unconstrained latent sweep.

### 6.2 Two-to-three-day simulation-real bridge audit

The historical `template_budding` experiment produced visually overlapping
MOMENT PCA/t-SNE projections, but domain ROC AUC remained about `0.99`. It used
an obsolete 2.048 ms/window-z-score contract and injected textures from real
budding crops without a template-source holdout. It is useful diagnostic
history, not current simulator validation.

The bridge audit must compare, under the v2 input contract:

- the current analytic `yeast-passage-identifiable-v1` simulator;
- a historical-style template-conditioned comparator rebuilt from
  `followup_train` only;
- real `followup_train` and `followup_validation` capture blocks.

Measure domain ROC AUC in raw/handcrafted, MOMENT, R0/A3, and R0/A4 features.
Repeat after matching or conditioning on duration, Doppler, RMS, SNR proxy,
component count where available, and quality. Split synthetic data by latent
and template source, never by individual generated view.

The following thresholds are triage rules, not claims of domain equivalence:

| Conditional domain AUC | Decision |
|---:|---|
| `<= 0.70` | Domain gap is not the month's priority |
| `0.70-0.85` | Permit one focused simulator-calibration ablation |
| `> 0.85` | Treat simulator mismatch as a major limitation before more adaptation |

Do not add adversarial domain alignment during Week 1. First determine which
measurable signal properties separate the domains.

### 6.3 Week 1 exit gate

Week 1 passes only when:

- the prospective split is registered and integrity-tested;
- no code path can open the old or new final split during training;
- the handcrafted feature-family audit is complete;
- the conditional domain audit is complete;
- R0-R3 losses, seeds, budgets, and promotion rules are frozen.

## 7. Week 2: Collapse-Aware Spectral SSL

Use one transformer architecture, the same training data, optimizer, batch size,
epoch budget, and representation seeds `42`, `43`, and `44`.

| Cell | Time reconstruction | Spectral reconstruction | Variance/covariance regularization | Purpose |
|---|---:|---:|---:|---|
| R0 | Yes | No | No | Frozen A4-style reference under v2 |
| R1 | Yes | No | Yes | Isolate collapse prevention |
| R2 | Yes | Yes | No | Isolate time-frequency supervision |
| R3 | Yes | Yes | Yes | Combined candidate |

### Spectral objective

Use a small fixed set of STFT resolutions selected in physical time before
training. Compare magnitude or log-magnitude only where phase invariance is
declared. Keep time-domain reconstruction so the objective cannot ignore local
waveform structure.

### Collapse control

Use a VICReg-style variance and covariance penalty on paired views. It must:

- maintain per-dimension variance above a fixed floor;
- reduce redundant covariance without forcing arbitrary domain alignment;
- retain physical-factor recovery;
- be monitored together with effective rank and mean cosine similarity.

### Week 2 gate

R3 may proceed only if, on `followup_validation`, it:

- at least doubles median effective rank relative to R0;
- does not reduce retained-factor recovery below R0 beyond uncertainty;
- does not reduce 10%-label macro F1 below R0 beyond uncertainty;
- improves at least one preregistered downstream or complementarity metric;
- converges for all three representation seeds.

If no cell passes, report that objective correction did not repair the current
representation and skip expensive extensions.

## 8. Week 3: Targeted Transfer Experiments

Week 3 is conditional. Run only the branches authorized by Weeks 1-2.

### 8.1 Quality-balanced real adaptation

If R3 passes, adapt it on a fixed mixture of:

- strict events;
- medium-quality events;
- low-quality or rejected candidates where their semantics are known;
- random background windows from training recordings.

Balance sampling by source proxy and quality stratum. No source label may enter
the SSL loss. Compare quality-probe AUC, source-proxy performance, physical
recovery, and robustness before and after adaptation. Reduced quality
separability counts as useful only if retained information is preserved.

### 8.2 Simulator calibration

Run only if the conditional bridge audit justifies it. Change one documented
simulator family at a time, prioritizing measured residual differences such as
noise spectrum, baseline process, envelope asymmetry, or sensor response. Do
not copy held-out real signals into synthetic examples.

### 8.3 Parameter-matched CNN control

Run only after the objective is frozen. Compare the selected objective with a
parameter-matched 1D CNN encoder using the same inputs, losses, data, epochs,
and seeds. This is the only architecture comparison in the month. Treat it as
a controlled mechanism experiment, not a general CNN-versus-transformer claim.

## 9. Week 4: Frozen Evaluation and Publication

1. Freeze the selected representation, fusion, and probe protocol.
2. Verify that `followup_test` has never appeared in a completed or failed run.
3. Open `followup_test` once.
4. Evaluate handcrafted, R0, selected learned-only, and
   handcrafted+selected representation.
5. Run grouped hierarchical bootstrap and convergence checks.
6. Generate all final tables and figures from manifested results.
7. Write a concise follow-up report linked to the v1 report.

The final result must report every completed R0-R3 cell. A negative final result
cannot be followed by another architecture or loss search on `followup_test`.

## 10. Common Evaluation Surface

Every eligible method receives:

- the same numerical input contract;
- the same grouped split;
- label fractions 1%, 5%, 10%, 25%, and 100%;
- the same paired probe seeds;
- linear-probe and fixed-MLP sensitivity results;
- macro F1, balanced accuracy, class recall, and calibration;
- label-efficiency AUC;
- effective rank, spectrum, and cosine concentration;
- retained synthetic-factor recovery;
- grouped cross-recording retrieval;
- quality and domain probes;
- bounded perturbation robustness;
- convergence diagnostics and uncertainty intervals.

PCA/t-SNE figures are supplementary explanations only. They cannot pass a gate.

## 11. Stop Rules

Stop or skip an experiment when:

- it requires reading either final split before protocol freeze;
- it changes more than one causal factor without an ablation;
- reconstruction improves while geometry and downstream metrics do not;
- domain separability falls only because embeddings collapse;
- quality invariance erases retained physical or proxy information;
- simulator calibration uses validation/test real templates as generated input;
- an optional branch lacks the Week 1 or Week 2 gate that authorizes it;
- the only motivation is making the result positive.

Do not spend the month on UMAP/t-SNE sweeps, broad hyperparameter searches,
large MOMENT fine-tuning, pseudo-morphology labels, or repeated final-test use.

## 12. Deliverables

### Code and data

- registered v2 grouped split and datasets;
- importable spectral and collapse-aware losses with tests;
- split guards preventing final-set access;
- conditional domain-gap evaluator;
- complementarity and feature-family evaluator;
- quality-balanced adaptation sampler;
- optional parameter-matched CNN control.

### Results

- manifested local smokes and pfcalcul full runs;
- complete R0-R3 result matrix;
- feature-family and fusion tables;
- conditional domain-gap report;
- embedding-health and physical-retention report;
- one-time v2 final predictions and uncertainty;
- publication-ready PDF and PNG figures.

### Documentation

- short iterative execution log;
- `YEAST_SSL_FOLLOWUP_STUDY_REPORT.md` as the final scientific account;
- updated README navigation;
- explicit links between v1 evidence, v2 hypotheses, artifacts, and conclusions.

## 13. Definition of Done

The month is complete when:

- the old final split remains untouched;
- the new split and protocol are registered and frozen;
- handcrafted complementarity is understood quantitatively;
- the domain gap has been conditionally diagnosed rather than judged visually;
- R0-R3 are complete for all authorized seeds;
- optional branches obey their gates;
- the v2 test is opened at most once;
- all runs and figures validate through workspace manifests;
- the conclusion states what improved, what failed, why, and within which
  limits;
- no remaining experiment can be justified using only the available exhausted
  acquisition evidence.

At that point P3 can be frozen, whether the learned representation is promoted
or the study establishes that handcrafted features and acquisition diversity
remain the dominant constraints.
