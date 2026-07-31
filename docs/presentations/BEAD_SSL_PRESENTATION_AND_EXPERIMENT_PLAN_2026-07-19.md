# Single-Bead SSL Presentation and Experiment Plan

**Status:** full GPU comparison complete; visual checkpoint awaiting review  
**Dataset decision:** freeze `yeast-passage-simulations@v1`  
**Decision date:** 2026-07-19

## Scope Decision

The single-bead SSL study remains on `yeast-passage-simulations@v1`. The v1
pool owns the user-approved 2, 4, and 10 µm visual pairs and the completed B0
P25 training result. A v2 rerun would change the simulation population without
answering the current presentation question, so it is not required.

The v2 dataset and its historical smoke run are preserved. They are not active
single-bead evidence and must not replace the validated v1 figures or metrics.
This exception does not change the separate v2 policy for yeast studies.

## Presentation Question

Can a compact patch transformer learn non-trivial masked reconstruction from a
jointly randomized single-particle simulator, and does that learned predictor
reconstruct held-out simulated and limited real bead signals better than fixed
signal-processing controls?

## Evidence Already Available

| Block | Status | Evidence |
|---|---|---|
| Joint parameter simulation | implemented | Registered v1 signals and complete latent metadata |
| Local realism | validated | User-approved 2, 4, and 10 µm real/simulated pairs |
| Historical reconstruction | available, causal wording limited | Legacy training history and reconstruction outputs |
| Corrected P25 masking | validated | 4096 samples, 256 patches, 64 hidden patches |
| B0 optimization | validated | Validation masked MSE `1.5771 -> 0.0304` |
| Simulated reconstruction | validated | Model `0.0304`, interpolation `0.2101` |
| Real reconstruction | conditionally validated | Model `0.1230`, interpolation `0.5888`, 28 traces |

The existing aggregate support-overview visual is excluded and must not be
reused. Its metric-selected neighbours did not consistently match visual
judgement.

## Required Presentation Visuals

1. **Joint simulation design.** Show that duration, Doppler, position, SNR,
   amplitude, phase, and other nuisance factors vary together. Contrast this
   with a one-factor latent sweep.
2. **Validated local realism.** Show only the approved 2, 4, and 10 µm pairs.
   State that these prove existence of credible neighbours, not population
   coverage.
3. **Real/simulated overlap method.** Explain descriptor normalization,
   nearest-neighbour distance, and support threshold with a simple overlap
   diagram. Do not publish a coverage percentage until the metric contract is
   visually qualified.
4. **Historical failure.** Show the original input, masks, loss components, and
   reconstructions. State that waveform MSE improved little while the composite
   objective decreased; do not claim that masking was the unique cause.
5. **P25 remediation.** Show the full signal, 16-sample patches, exact 25%
   label-free mask, learned mask token, and visible local context.
6. **Learning proof.** Plot train and validation masked MSE with zero,
   visible-mean, nearest, and interpolation controls.
7. **Held-out simulated reconstruction.** Show predeclared full traces and
   masked-region zooms.
8. **Evaluation-only real reconstruction.** Show the three approved examples
   and the aggregate 28-trace control comparison.
9. **Limits and next experiment.** Separate established B0 evidence from the
   untested loss and masking extensions.

## Experiments Still Required

### Presentation-Critical

1. Qualify or replace the descriptor contract used for real/simulated overlap.
2. Complete a blind metric diagnostic before reporting aggregate coverage.
3. Produce the final simple overlap visual after that diagnostic.

The overlap quantification is being produced in a separate workstream and will
be integrated later. It is not recomputed by this experiment matrix.

### Scientific Extension

Run a matched v1 ablation under the same model, fixed P25 evaluation masks,
seed set (`42`, `43`, `44`), data splits, and 20-epoch training budget:

| Cell | Objective |
|---|---|
| B0 | masked waveform MSE |
| B1 | waveform + derivative |
| B2 | waveform + energy |
| B3 | waveform + derivative + energy |

Evaluate masked MSE, derivative error, energy error, output RMS, improvement
over interpolation, and predeclared simulated and real reconstructions. Use
multiple seeds before claiming improvement.

The energy term is computed from mean masked energy, not an unnormalized sum,
so its scale does not grow with the 1,024-point mask budget.

The bead port of the overlapping-candidate cycle is named `CYCLIC25`. Its
catalogue uses 16-sample targets at stride 8. Each pass selects 32 event-region
and 32 background windows without within-pass overlap, for exactly 1,024
hidden samples. Short simulated events are expanded symmetrically with local
context to make the 32-window event budget feasible. Successive passes cover
all event candidates. P25 and CYCLIC25 use the same sample-visibility mask
encoding; evaluation remains fixed P25 so aggregate metrics are comparable.

Execution order is evidence-gated but uses full GPU runs only:

1. `B1/P25/seed42` technical and numerical gate.
2. Remaining `B0-B3/P25` cells for seeds 42, 43, and 44.
3. `B0/CYCLIC25/seed42` mask-policy gate.
4. Remaining `B0/CYCLIC25` seeds 43 and 44.

The earlier B0 result remains valid historical evidence, but it is not reused
inside the new numeric matrix because the common sample-visibility encoding is
new. No CPU training smoke is part of this plan.

## Full-GPU Results

The complete comparison contains three seeds (`42`, `43`, `44`) per cell.
Values below are mean +/- sample standard deviation. All models are selected on
simulated validation only; the 28 real traces remain evaluation-only.

| Cell | Simulated MSE | Simulated derivative MSE | Real MSE | Real derivative MSE |
|---|---:|---:|---:|---:|
| B0 | 0.03194 +/- 0.00033 | 0.13810 +/- 0.00789 | 0.15098 +/- 0.01622 | 0.14846 +/- 0.00758 |
| B1 | 0.03189 +/- 0.00095 | 0.00470 +/- 0.00038 | 0.14929 +/- 0.00370 | 0.01418 +/- 0.00077 |
| B2 | 0.03155 +/- 0.00009 | 0.14525 +/- 0.01326 | 0.15524 +/- 0.01627 | 0.15618 +/- 0.00710 |
| B3 | 0.03197 +/- 0.00115 | 0.00474 +/- 0.00037 | 0.14964 +/- 0.00635 | 0.01455 +/- 0.00112 |

Decision:

- retain **B1**: it removes patch-boundary derivative artifacts while
  preserving B0-level MSE on simulation and real traces;
- reject **B2** as redundant at `lambda_energy=0.05`;
- do not retain **B3** because it reproduces B1 without an energy benefit;
- describe derivative regularization as signal-structure consistency, not as
  a biophysical law.

CYCLIC25 was corrected after an invalid `cmp1` run exposed a changing schedule
seed that defeated the intended cycle cache. The invalid run is preserved but
excluded. The corrected `cmp2` result is:

| Policy | Simulated MSE | Real MSE | Simulated energy error | Real energy error |
|---|---:|---:|---:|---:|
| P25/B0 | 0.03194 +/- 0.00033 | 0.15098 +/- 0.01622 | 0.04995 +/- 0.00415 | 0.27565 +/- 0.01197 |
| CYCLIC25/B0 | 0.11295 +/- 0.01285 | 0.54139 +/- 0.09517 | 0.16558 +/- 0.02235 | 0.56217 +/- 0.03445 |

CYCLIC25 still beats simple interpolation on seed 42, but it generalizes much
worse than P25 under the fixed P25 evaluation. This result establishes
cross-policy transfer toward P25 masks; it is not a policy-neutral ranking.

### Cross-mask evaluation

The fixed-P25 comparison was challenged because it could favor the P25
training distribution. The six B0 models were therefore reloaded without
retraining and evaluated under both P25 and CYCLIC25 masks. CYCLIC25
evaluation averages every unique cycle pass for each trace. Real cycle masks
use reviewed particle annotation bounds. The main comparison uses the fixed
epoch-20 checkpoints to avoid selecting epochs under P25 validation.

| Train -> evaluation | Simulation MSE | Matched interpolation | Real MSE | Matched interpolation |
|---|---:|---:|---:|---:|
| P25 -> P25 | 0.03290 | 0.21062 | 0.13842 | 0.62785 |
| CYCLIC25 -> P25 | 0.11347 | 0.21062 | 0.53746 | 0.62785 |
| P25 -> CYCLIC25 | 1.70223 | 2.10953 | 1.74796 | 2.24162 |
| CYCLIC25 -> CYCLIC25 | 0.35551 | 2.10953 | 1.07694 | 2.24162 |

The original ranking was indeed evaluation-dependent:

- P25 is best on P25 masks and retains 84.4% simulation / 78.0% real gain
  relative to matched interpolation.
- CYCLIC25 is best on CYCLIC25 masks and retains 83.1% simulation / 52.0% real
  gain relative to matched interpolation.
- Each model specializes strongly to its training mask distribution.
- CYCLIC25 transfers less well from simulation to real event-focused
  reconstruction, but it does learn its intended task; it must not be described
  as a failed masking policy.

The practical decision remains to use P25 as the default label-free SSL
objective. CYCLIC25 is a valid informed event-reconstruction objective when
event support is supplied. Raw MSE values across P25 and CYCLIC25 evaluations
must not be compared without their matched baselines because the hidden target
distributions have different difficulty.

## Proposed downstream representation benchmark

### Claim to test

> At the same SSL budget, does CYCLIC25 produce more useful frozen
> representations than P25?

Reconstruction cannot answer this question because P25 and CYCLIC25 define
different target distributions. The comparison therefore discards the
reconstruction decoder and evaluates both frozen encoders on identical
downstream tasks.

### Available population

The v1 single-component simulation population contains:

| Split | Signals | Unique latents | Views per latent |
|---|---:|---:|---:|
| train | 6,982 | 3,491 | 2 |
| validation | 1,368 | 684 | 2 |
| sealed held-out sensor test | 1,416 | 708 | 2 |

The two views of one latent share retained physical factors but have independent
nuisance realizations. Sampling, splitting, confidence intervals, and reported
sample counts must therefore use the latent as the independent unit.

The v1 metadata do not contain particle diameter, shape, velocity, or
orientation. The defensible retained targets for single-component signals are:

- `duration_ms`;
- `doppler_khz`.

Phase, event position, SNR, target RMS, baseline drift, and sensor response are
declared nuisances. Their predictability is reported separately as a leakage
diagnostic and is never averaged into the retained-factor score.

The downstream benchmark no longer uses the earlier single-annotation subset.
It uses every known annotation from the visually reviewed saturation dataset,
materialized as
`particles2snr-f-dual-clean-c1-descriptor-events-saturation-reviewed-development@v1`.
Its class counts are:

| Split | 2 µm | 4 µm | 10 µm |
|---|---:|---:|---:|
| train | 150 | 973 | 41 |
| validation | 49 | 248 | 13 |
| sealed test | 60 | 279 | 9 |

The real benchmark is group-disjoint by source trace. Real label-efficiency
starts at 25%, because smaller fractions are not representative for 10 µm.
Even the confirmatory test contains only nine 10 µm events from nine files, so
the real 10 µm result remains explicitly exploratory and is always accompanied
by individual predictions, recall, confusion matrix, and grouped uncertainty.

### Compared methods and budget

| Method | Encoder | SSL objective | Seeds |
|---|---|---|---|
| Random frozen | same randomly initialized architecture | none | 42-46 |
| P25/B0 | fixed epoch-20 encoder | waveform MSE under P25 | 42-46 |
| CYCLIC25/B0 | fixed epoch-20 encoder | waveform MSE under CYCLIC25 | 42-46 |
| Raw PCA-64 | train-only waveform PCA | none | deterministic per subset |
| Physical descriptors | eight classical relative descriptors | none | deterministic per subset |

The first P25 42-44 runs and the CYCLIC25 runs were generated from different
untracked revisions of `bead_ssl.py`; they are not accepted as final paired
runs. The final matrix retrains P25 42-46 with the exact source hash used by
CYCLIC25 42-46. Each pair shares initialization, shuffled batch order, dataset,
20 epochs, batch size 32, 4,380 updates, 139,640 signal presentations, and
142,991,360 masked values contributing to the loss. The budget is equalized;
the mask geometry and task difficulty are intentionally not equalized.

Embeddings are extracted from the complete unmasked signal with the mean pool.
The decoder is ignored. The two view embeddings are averaged per latent before
probing, leaving one independent embedding and one target per latent.

### Label-efficiency design

The labeled fractions are computed on the 3,491 training latents:

| Fraction | Labeled latents | Signals represented |
|---:|---:|---:|
| 1% | 35 | 70 |
| 5% | 175 | 350 |
| 10% | 349 | 698 |
| 25% | 873 | 1,746 |
| 100% | 3,491 | 6,982 |

For each of ten subset seeds, construct nested subsets shared by every method:
the 1% subset is contained in 5%, then 10%, then 25%. Sampling is balanced over
a fixed joint duration/Doppler quantile grid. No validation latent participates
in subset construction or probe tuning.

Use one `StandardScaler + Ridge` probe per target. Fit scalers on the selected
training latents only. Select Ridge regularization from a fixed logarithmic
grid using inner cross-validation inside the labeled subset. Evaluate every
probe on the same 684 validation latents.

Primary metrics:

- mean retained-factor R2 across duration and Doppler;
- target-specific R2;
- normalized MAE, divided by the validation-target IQR;
- normalized area under the label-efficiency curve;
- paired CYCLIC25-minus-P25 difference at 10% labels.

Each fraction below 100% has five encoder seeds crossed with ten shared subset
seeds. Report individual encoder-seed points and a paired hierarchical 95%
bootstrap interval that first resamples encoder seed and then subset seed.
The 100% result has five encoder-seed observations.

All design, debugging, probe selection, and figure iteration use validation
only. These are development results. After code, checkpoints, hyperparameters,
metrics, figure templates, and interpretation rules are frozen, a separately
approved manifest authorizes one joint opening of the 708-latent simulated test
and the real test. Final probes are refit on train plus validation with no new
tuning. All confirmatory predictions are emitted once and no post-test
adaptation is permitted.

### Interpretation and fine-tuning

Simulation and real endpoints are interpreted separately. The report states
whether the evidence is coherent across domains, domain-specific, mixed, or
compatible with no established difference. Five paired seed differences remain
visible; bootstrap replication is not described as additional training
replication.

Fine-tuning is systematic rather than triggered by a favorable frozen result.
It compares from-scratch, P25, and CYCLIC25 initialization for all five paired
seeds at 10% and 100% simulation labels and at 25% and 100% real labels.
Internal early stopping is group-safe and uses only a calibration partition of
the training subset.

### Frozen development result

The five-seed development benchmark is complete. At 10% simulated labels,
`CYCLIC25 - P25` gives `+0.0101` mean R2 with hierarchical 95% interval
`[+0.0012, +0.0187]`. The difference across the complete simulated
label-efficiency curve is `+0.0157` normalized AUC
`[+0.0054, +0.0262]`. This is evidence of a modest, reproducible CYCLIC25
advantage for low-label simulated factor probing.

At full real labels, mean macro-F1 is `0.628` for random frozen, `0.635` for
P25, and `0.652` for CYCLIC25. The paired `CYCLIC25 - P25` interval
`[-0.026, +0.057]` and source-grouped interval `[-0.037, +0.065]` include
zero. The real result is therefore a favorable but uncertain development trend,
not an established advantage.

At full simulated labels, the physical-descriptor baseline remains strongest
(`0.985` mean R2), followed by CYCLIC25 (`0.980`), P25 (`0.972`), and random
frozen (`0.965`). The current evidence does not justify calling either SSL
representation universally superior to known physical features.

Canonical frozen evidence:

- `artifacts/cross-project/reports/bead-representation-benchmark-development-v3/`
- `bead_representation_decision_figure.png`
- `bead_representation_diagnostics.png`
- `summary.json`, `comparisons.json`, and retained 10 µm predictions.

Fine-tuning remains necessary because frozen probing and end-to-end adaptation
answer different questions. No sealed test has been opened.

### Fine-tuning development result

The full 20-run GPU matrix is complete and audited from saved predictions.
At 10% simulated labels, both initializations beat from-scratch in all five
seeds: `+0.0090` mean R2 for P25 and `+0.0092` for CYCLIC25. The paired
`CYCLIC25 - P25` difference is only `+0.0002` and changes sign across seeds.
At full simulated labels it is `-0.0001`. The two mask policies are therefore
effectively tied after end-to-end adaptation.

On real validation data, `CYCLIC25 - P25` is `+0.0017` macro-F1 at 25% labels
with grouped interval `[-0.0568, +0.0481]`, and `+0.0287` at full labels with
interval `[-0.0178, +0.0978]`. At full labels, mean differences against
from-scratch are `-0.0356` for P25 and `-0.0069` for CYCLIC25. These real
effects are mixed and uncertain; they do not establish a benefit from SSL
initialization.

Canonical fine-tuning evidence:

- `artifacts/cross-project/reports/bead-finetuning-development-v4/`
- `bead_finetuning_decision_figure.png` and `.pdf`;
- `effect_sizes.csv`, `fine_tuning_summary.csv`, and `summary.json`.

The report recalculates metrics from prediction CSVs and checks sample, target,
group, split, dataset-content, source, and checkpoint alignment. Real
fine-tuning remains exploratory because checkpoint selection used unweighted
internal cross-entropy rather than macro-F1, the five seeds share one validation
population, and 10 µm is rare.

The confirmatory readiness audit is intentionally blocked rather than falsely
frozen:

- `artifacts/cross-project/audits/bead-ssl-confirmatory-readiness-pending-v1/`

No sealed split has been accessed. A future test opening requires a separately
approved registered real confirmatory dataset, exact or revalidated execution
sources, a blocker-free readiness audit, and explicit one-shot authorization.

### Predeclared final figure

The final decision figure has three panels:

**A - Reconstruction-task difficulty and specialization.** Two 2x2 heatmaps
(simulation and real) report
`1 - model MSE / matched interpolation MSE`, with training policy on rows and
evaluation policy on columns. This panel explains why raw reconstruction MSE
cannot rank the encoders.

**B - Full-label frozen representation quality.** Paired seed points report
simulation mean R2 and real macro-F1 for random frozen, P25, CYCLIC25, raw
PCA-64, and physical descriptors. P25 and CYCLIC25 points from the same seed
are connected; bars alone are forbidden.

**C - Label efficiency.** The main panel plots mean retained-factor R2 against
1%, 5%, 10%, 25%, and 100% labeled latents for random frozen, P25, and
CYCLIC25. Show hierarchical 95% intervals and annotate the actual latent count
under every fraction. Duration and Doppler curves are provided as supplementary
panels so the mean cannot hide target disagreement.

The final headline is written only after the frozen, fine-tuning, and one-shot
confirmatory results are available. PCA, t-SNE, UMAP, and selected
reconstructions remain explanatory supplements and cannot decide the
comparison.

Canonical aggregate evidence:

- `artifacts/cross-project/reports/bead-ssl-comparison-cmp2-r1/`
- `loss_ablation_metrics.png`
- `learning_curves.png`
- `mask_policy_metrics.png`
- `run_metrics.csv` and `summary.json`
- Cross-mask correction:
  `artifacts/cross-project/reports/bead-ssl-cross-mask-comparison-r2/`
- Main corrected figure: `mask_policy_cross_evaluation.png`
- Checkpoint-selection control: `checkpoint_selection_sensitivity.png`

## Limitations

- The synthetic pool was not densified inside the empirically realistic region.
- The approved 2, 4, and 10 µm pairs are local examples, not coverage estimates.
- Real reconstruction uses only 28 evaluation traces from the available
  acquisition.
- The earlier reconstruction comparison has only three optimization seeds;
  the downstream benchmark uses five paired seeds.
- CYCLIC25 uses simulator event support during training and is therefore an
  informed synthetic objective, not label-free masking.
- Reconstruction quality does not by itself establish downstream
  representation utility.
