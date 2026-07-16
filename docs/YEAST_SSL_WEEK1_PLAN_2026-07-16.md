# Yeast SSL Follow-Up: Week 1 Plan

**Dates:** 2026-07-16 to 2026-07-22

**Parent plan:**
[`YEAST_SSL_ONE_MONTH_FOLLOWUP_PLAN_2026-07-16.md`](YEAST_SSL_ONE_MONTH_FOLLOWUP_PLAN_2026-07-16.md)

**Week objective:** diagnose the v1 failure mechanisms, create a genuinely
prospective v2 data protocol, and freeze every R0-R3 choice before new full
training

**Full GPU training:** not authorized during Week 1

**Execution status:** complete. See
[`YEAST_SSL_WEEK1_EXECUTION_REPORT_2026-07-16.md`](YEAST_SSL_WEEK1_EXECUTION_REPORT_2026-07-16.md).

## 1. Required Week 1 Decisions

By the end of the week, the evidence must answer four questions:

1. Can the available v1 development records support a leakage-free
   `followup_train` / `followup_validation` / `followup_test` split?
2. Which handcrafted feature families explain their advantage, and do historical
   A3/A4 embeddings contain complementary information on train/validation?
3. After matching observable physical properties, is simulation origin still
   easy to predict?
4. Are the R0-R3 objectives, loss scales, training budgets, metrics, and stop
   rules sufficiently frozen to authorize Week 2?

Week 1 is successful when these questions have reproducible answers, including
a justified decision not to proceed if the data protocol fails.

## 2. Non-Negotiable Boundaries

- Never read the old v1 `in_session_test` in new analysis or training code.
- Do not inspect `followup_test` labels, embeddings, signal plots, or metrics
  after the new split is frozen.
- Do not include `followup_test` signals in SSL or physics-supervised training.
- Do not overwrite v1 datasets, checkpoints, metrics, predictions, or reports.
- Use source folders only as source-condition proxies, never morphology labels.
- Do not select a simulator from PCA/t-SNE appearance.
- Do not add domain-adversarial training, new model families, or broad
  hyperparameter sweeps this week.
- Accept a failed split audit or negative complementarity result as valid
  evidence.

Historical A3/A4 checkpoints were trained on the old development pool. They may
be used for exploratory train/validation diagnosis, but they are not eligible
for the prospective v2 final endpoint. R0-R3 must later be trained from scratch
without access to `followup_test` signals.

## 3. Repository Ownership

| Work | Repository |
|---|---|
| Source/capture-block audit, v2 split, dataset registration | `particles2SNR-pipeline` |
| Template-conditioned comparator generation and provenance | `particles2SNR-pipeline` |
| Handcrafted ablations, fusion, domain probes, configs | `unsupervised-learning-flow-cytometry` |
| Cross-project validation and pfcalcul control | Workspace root |
| Generated evidence | `artifacts/<github-project>/...` with valid `run.json` |

Shared datasets must be resolved by registered dataset ID. No new absolute path
or symlink view is permitted.

## 4. Day 1: Data Inventory and Prospective Split

### 4.1 Inventory before splitting

Using only records assigned to v1 `development_train`, report:

- unique records and duplicate families;
- capture blocks per source proxy;
- events per record and capture block;
- strict, medium, low-quality, and background availability;
- SNR, duration, Doppler, RMS, and event-count distributions;
- whether any record or duplicate family appears in another v1 split;
- the minimum source-proxy block count that constrains a three-way split.

The inventory must operate from registered dataset manifests, not inferred
download paths.

### 4.2 Candidate split

Target a deterministic source-proxy-stratified 60/20/20 allocation by capture
block:

```text
followup_train       60%
followup_validation  20%
followup_test        20%
```

The allocation ratio may change only if the inventory proves that 60/20/20
cannot preserve meaningful block coverage. Any change must be justified before
labels or model outputs are evaluated.

### 4.3 Split integrity tests

Require automated tests for:

- zero record crossing;
- zero capture-block crossing;
- zero duplicate-family crossing;
- all event descendants inheriting their record split;
- all available source proxies represented in each split, or an explicit
  blocker explaining why this is impossible;
- preprocessing statistics derived from `followup_train` only;
- forbidden v1 and v2 final split names rejected by training loaders;
- stable reproduction from the frozen seed and source manifest hash.

### 4.4 Day 1 deliverables

- registered source-index/split candidate owned by `particles2SNR-pipeline`;
- split audit JSON, CSV, and concise Markdown report;
- proposed v2 representation dataset ID;
- test coverage for split and loader guards;
- written `pass`, `revise`, or `block` decision.

If Day 1 is blocked, Days 2-3 may continue as historical diagnosis, but Week 2
training remains unauthorized.

## 5. Day 2: Handcrafted and Complementarity Audit

### 5.1 Freeze feature families

Map every handcrafted feature to one declared family before measuring its
performance:

| Family | Expected content |
|---|---|
| Time morphology | duration, width, asymmetry, peak structure |
| Frequency | centroid, bandwidth, dominant Doppler structure |
| Envelope | support, envelope peaks, concentration |
| Energy/amplitude | RMS and related scale summaries |
| Quality | SNR and detector-quality summaries |

Features that cannot be assigned unambiguously must be documented and excluded
from causal family claims.

### 5.2 Fixed comparison matrix

On `followup_train` and `followup_validation` only, evaluate:

- every feature family alone;
- all handcrafted features;
- leave-one-family-out handcrafted ablations;
- historical A3 embedding alone;
- historical A4 embedding alone;
- handcrafted + A3;
- handcrafted + A4;
- quality-only and RMS-only shortcut controls.

Use the same preprocessing and record-group label sampling for every method.
Run label fractions 1%, 5%, 10%, 25%, and 100% with paired seeds.

### 5.3 Probe policy

The primary diagnostic probe is the existing converged logistic probe. Add one
fixed small MLP sensitivity probe to detect information that is present but not
linearly accessible. Freeze its width, regularization, optimizer, epoch cap, and
early-stopping policy before reading validation results.

The MLP cannot replace the linear primary endpoint merely because it produces a
more favorable ranking.

### 5.4 Complementarity analysis

For each learned representation, report:

- paired macro-F1 difference of fusion versus handcrafted;
- label-efficiency AUC difference;
- per-proxy recall changes;
- calibration changes;
- errors by capture block, SNR, duration, and quality;
- residual prediction gain after a probe has already received handcrafted
  features;
- convergence diagnostics for every probe.

Use hierarchical bootstrap over probe/representation runs and capture blocks.
This is validation diagnosis, not the final v2 promotion test.

### 5.5 Day 2 decision

Classify the historical representation as:

- `complementary_signal_present`;
- `nonlinear_only_signal`;
- `redundant_with_handcrafted`;
- `quality_shortcut_dominated`; or
- `not_evaluable`.

The classification must cite quantitative evidence and uncertainty, not latent
appearance.

## 6. Days 3-4: Simulation-Real Bridge Audit

### 6.1 Historical evidence inventory

Record, but do not rerun as final evidence, the July 3 experiment:

- visually overlapping MOMENT PCA/t-SNE;
- MOMENT real-versus-template ROC AUC `0.9907`;
- Conv1D-GAP ROC AUC `0.9896`;
- real-versus-real control AUC near `0.52`;
- obsolete 2.048 ms/window-z-score contract;
- template texture sourced from the same real budding pool;
- no template-source grouped holdout;
- historical artifact provenance below current `run.json` standard.

This explains why visual overlap cannot settle the current domain-gap question.

### 6.2 Comparable v2 signal sets

Build two simulation sources under the exact v2 input contract:

1. current analytic `yeast-passage-identifiable-v1`;
2. a template-conditioned diagnostic comparator whose real template bank uses
   `followup_train` only.

The template comparator is a diagnostic upper-bound on visual realism, not a
candidate physical simulator. Its validation examples must be grouped by the
source record from which the template was derived.

Use the same signal-derived estimators on real and synthetic data. Do not compare
generator latent values directly with differently estimated real quantities.

### 6.3 Observable matching

Construct a balanced matched subset using preregistered observable covariates:

- duration;
- dominant Doppler frequency;
- RMS;
- SNR proxy;
- component or peak count where estimable by the same algorithm;
- event offset and quality-related summaries as diagnostics.

Standardize covariates using domain-training data only. Perform nearest-neighbor
or caliper matching with a fixed distance rule, report discarded samples, and
verify post-match standardized mean differences. If overlap is insufficient,
report lack of common support rather than extrapolating.

### 6.4 Domain probes

Primary domain probe:

- standardized logistic regression;
- train on real `followup_train` capture blocks and synthetic training latents;
- validate on disjoint real `followup_validation` blocks and synthetic latents;
- balance domain sample counts;
- record convergence and ROC AUC uncertainty.

Secondary sensitivity probe:

- bounded-depth random forest;
- identical grouped train/validation identities;
- used only to detect nonlinear residual separation.

Evaluate both before and after observable matching in:

- raw/downsampled signal summaries;
- handcrafted features;
- frozen official MOMENT embeddings;
- historical A3/A4 embeddings as exploratory references.

### 6.5 Explain the separation

For any conditional AUC above `0.70`, report which measured features or frequency
regions carry domain information. Use permutation importance or grouped feature
ablation. Do not infer simulator defects from a black-box AUC alone.

### 6.6 Domain triage decision

| Conditional validation AUC | Week 3 authorization |
|---:|---|
| `<= 0.70` | No simulator branch; prioritize collapse and spectral SSL |
| `0.70-0.85` | Authorize one measured simulator-calibration ablation |
| `> 0.85` | Mark simulator mismatch major; require correction before extended real adaptation |

The decision also requires retained-factor identifiability. A simulator that
looks real but loses controlled physical factors is not an acceptable
replacement.

## 7. Day 4: Freeze the R0-R3 Protocol

Day 4 uses only train-only numerical smoke checks and the completed diagnostic
reports. It does not run full representation training.

### 7.1 Common architecture and budget

Freeze for R0-R3:

- 4096 samples at 1 MHz under the v2 registered contract;
- the existing compact 96-dimensional patch transformer;
- patch size/stride and masking policy;
- optimizer, learning rate, weight decay, batch size, and epoch budget;
- representation seeds 42, 43, and 44;
- paired probe seeds and label fractions;
- early-stopping and checkpoint-selection metric;
- maximum pfcalcul runtime and retry policy.

No cell may receive a larger model or training budget.

### 7.2 Spectral objective freeze

Use physical-time reasoning to freeze a small multi-resolution STFT set. The
initial candidate is:

| Window | Duration at 1 MHz | Frequency-bin width |
|---:|---:|---:|
| 128 samples | 0.128 ms | 7.8125 kHz |
| 256 samples | 0.256 ms | 3.90625 kHz |
| 512 samples | 0.512 ms | 1.953125 kHz |

Use fixed quarter-window hops. Confirm that this set covers the measured
7.8-23.4 kHz Doppler and 0.464-1.424 ms duration ranges. Freeze magnitude versus
log-magnitude treatment, normalization, epsilon, and relative loss weights.

Loss weights may be scale-calibrated on a fixed set of `followup_train` batches
to prevent one term from dominating numerically. They may not be selected from
proxy-label validation performance.

### 7.3 Variance/covariance objective freeze

Specify before Week 2:

- variance floor;
- variance and covariance weights;
- which paired views receive the loss;
- batch-size requirements;
- behavior when a batch lacks sufficient independent latents;
- logged per-dimension variance, off-diagonal covariance, effective rank, and
  mean cosine similarity.

Synthetic paired views share retained factors. Real paired views may use only
physically allowed nuisance transformations from the frozen information policy.

### 7.4 R0-R3 freeze artifact

Produce one versioned config and one machine-readable protocol summary containing:

- exact R0-R3 loss composition;
- all weights and physical resolutions;
- datasets and forbidden splits;
- model/training budgets;
- primary and secondary endpoints;
- gates and stop rules;
- expected artifact paths and hashes.

## 8. Day 5: Preflight and Week 1 Decision

### 8.1 Smallest applicable validation

Run only:

- unit tests for split leakage and registered dataset loading;
- unit tests for feature-family and fusion construction;
- chance and deliberately separable synthetic tests for domain probes;
- loss shape, gradient, finite-value, and deterministic tests;
- CPU mini-batch smoke for R0-R3;
- one bounded CUDA smoke only if CPU cannot exercise a required code path;
- artifact-manifest validation;
- `workspace doctor`, root tests, and affected project tests.

Before any pfcalcul smoke, verify Slurm, runner queues, and existing artifacts to
avoid duplicate work.

### 8.2 Week 1 evidence package

Create manifested evidence under clear project-owned paths, including:

- split inventory and integrity report;
- handcrafted family and complementarity report;
- bridge audit before/after matching;
- domain feature-importance report;
- frozen v2 config and protocol summary;
- test/smoke report;
- concise Week 1 decision Markdown.

Every figure must be generated from stored tables, and every table/figure must
be listed and checksummed in `run.json`.

### 8.3 Final Week 1 gate

| Gate | Pass condition |
|---|---|
| W1-A data | Prospective grouped split passes every integrity test |
| W1-B complementarity | Feature-family and fusion diagnosis is complete and converged |
| W1-C domain | Conditional domain audit and common-support report are complete |
| W1-D protocol | R0-R3 config, losses, budgets, metrics, and stops are frozen |
| W1-E infrastructure | Tests, manifests, workspace doctor, and applicable smoke pass |

Week 2 full training is authorized only if W1-A, W1-D, and W1-E pass. W1-B and
W1-C may produce negative findings, but they must be complete enough to justify
the selected objective and optional Week 3 branches.

## 9. Daily Execution Rhythm

At the end of each day:

1. record commands, revisions, inputs, outputs, findings, and blockers in a
   short append-only execution note;
2. validate new artifacts and checksums;
3. run the narrow tests affected that day;
4. update gate states without rewriting earlier observations;
5. stop any branch whose evidence already satisfies a stop rule.

Keep notes concise: decision, evidence path, consequence, and next action. The
final Week 1 report should synthesize them rather than duplicate every command.

## 10. Week 1 Definition of Done

Week 1 is complete when:

- the old final split was never read;
- the v2 split is either valid and registered or scientifically blocked with a
  documented reason;
- handcrafted superiority is decomposed by feature family;
- learned-feature complementarity is measured on train/validation only;
- historical visual overlap is reconciled with quantitative domain separation;
- the current conditional domain gap has a triage decision;
- every R0-R3 choice is frozen before full training;
- code paths, manifests, links, and tests validate;
- the Week 2 authorization decision is explicit.

No additional visualization or exploratory model is required to close Week 1.
