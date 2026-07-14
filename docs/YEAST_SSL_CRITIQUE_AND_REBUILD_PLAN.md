# Yeast Representation Learning: Scientific Critique and Rebuild Plan

**Status:** pre-experiment redesign; no new large training run is authorized

**Date:** 2026-07-14

**Scope:** yeast and particle event representation learning formerly called
P3 SSL

**Execution record:** [`YEAST_SSL_EXECUTION_LOG.md`](YEAST_SSL_EXECUTION_LOG.md)

## Executive Decision

The existing project contains useful components, but it does not yet form one
defensible scientific study. It currently mixes event-dataset construction,
masked reconstruction, physical simulation, physics-guided contrastive losses,
public pretrained models, supervised classification, latent-space figures, and
full-trace detector transfer. These components answer different questions and
use partially incompatible data contracts.

The study should be rebuilt around one primary question:

> Does physics-informed synthetic pretraining followed by self-supervised
> adaptation on unlabeled real signals improve label-efficient and
> cross-acquisition yeast-event representation relative to strong pretrained,
> supervised, raw-feature, and random controls?

Operationally, the objective is to learn a physically meaningful representation
of yeast signals from abundant simulations and scarce unlabeled real
measurements, then test whether that representation supports downstream tasks
with few labels. "Physically meaningful" is not inferred from the use of SSL:
it must be demonstrated through retained-factor recovery, nuisance robustness,
cross-acquisition transfer, and comparison with non-physical controls.

The new study is not authorized to begin full training until the data identity,
split, input semantics, nuisance-variable policy, event-selection audit, and
baseline protocol pass the gates in this document.

## 1. What the Existing Study Was Trying to Do

The original idea was scientifically reasonable:

1. exploit raw signals without requiring dense class labels;
2. learn an encoder by reconstructing masked waveform regions;
3. test whether its latent space organizes particle and yeast passages in a
   useful way;
4. improve physical fidelity by introducing synthetic signals with known
   parameters;
5. compare the learned representation with MOMENT, PatchTST, and a supervised
   Conv1D-GAP model;
6. reuse a promising pretrained representation in downstream detection.

The first pretext task was MOMENT-like masked reconstruction. A 4096-value input
was patchified, approximately 25% of the timeline was hidden in contiguous
blocks, and the model reconstructed waveform, derivative, and energy terms.
Labels were not reconstruction targets.

The later hybrid design added synthetic signals with known physical parameters,
physical contrastive supervision, and perturbation invariance. This extension
was intended to prevent reconstruction from learning only local waveform
texture.

The yeast branch added a handcrafted time-frequency detector that extracts and
centers high-quality yeast passages before encoding them. These event crops were
then used for pretrained-model comparison, classification probes, retrieval,
and latent visualization.

## 2. Overall Scientific Critique

The current project is not merely unfinished. Several design choices are
mutually incompatible. Running the existing configuration longer would produce
more precise measurements of an ambiguous experiment rather than a stronger
scientific conclusion.

| Severity | Finding | Required disposition |
|---|---|---|
| Blocker | Two incompatible physical inputs share the 4096 tensor length | Freeze one physical contract before training |
| Blocker | Augmentations remove variables that physical losses try to encode | Freeze a retained-versus-nuisance policy |
| Blocker | Event-level splitting can leak source recordings | Split by acquisition identity before extraction |
| Blocker | Strict event selection is not independently validated | Build a manually reviewed selection-bias audit |
| Major | Real-only from-scratch SSL may be underpowered | Prefer synthetic/pretrained initialization plus real adaptation |
| Major | The hybrid objective is partly physics-supervised | Rename and ablate supervision sources honestly |
| Major | Latent visualization can substitute for held-out evidence | Make grouped label efficiency and OOD transfer primary |
| Ownership | Dataset extraction and simulation live with representation code | Move generation/provenance ownership to `particles2SNR-pipeline` |

### 2.1 The project has too many primary objectives

The current code can support all of the following:

- waveform reconstruction;
- physical-parameter organization;
- particle-class separation;
- yeast morphology classification;
- particle-to-yeast transfer;
- robustness and invariance;
- pretrained-backbone comparison;
- full-trace detector transfer.

No single model should be declared successful because it performs well on any
one of these axes. A representation can reconstruct well while being poor for
classification, classify well while using acquisition shortcuts, or preserve a
physical variable that should actually be treated as nuisance variation.

**Decision:** label-efficient yeast representation and cross-acquisition
transfer become primary. Reconstruction, physical fidelity, retrieval, and
robustness become mechanism or safety diagnostics. Full-trace detection remains
a separate P1 study.

### 2.2 The 4096-value input has incompatible physical meanings

Two existing paths produce arrays of length 4096:

| Path | Construction | Physical meaning |
|---|---|---|
| Historical SSL | 16,384-sample trace decimated by 4 | Longer duration at a reduced effective sampling rate |
| Current event pipeline | Centered raw crop of 4096 samples | Shorter duration at the original sampling rate |

Equal tensor shape does not imply equal representation. These paths differ in
sampling frequency, duration, anti-aliasing assumptions, context, and event
occupancy. Pooling their embeddings into one analysis would be invalid.

**Decision:** select one input contract from physical coverage before training.
The contract must record sampling frequency, duration, crop rule, filtering,
normalization, and allowed padding. The old and new 4096 paths must retain
different dataset IDs and must never be silently combined.

### 2.3 Preprocessing destroys variables the loss asks the model to preserve

The hybrid configuration includes physical parameters such as amplitude,
phase, event position, duration, frequency, and SNR. At the same time, the
pipeline applies transformations that remove or suppress some of them.

| Variable | Existing transformation or objective | Contradiction |
|---|---|---|
| Absolute amplitude `A` | Per-window z-score and amplitude-scale invariance | Absolute scale is removed while the latent loss may ask to preserve it |
| Event position `t0` | Centered event crops and shift invariance | Position becomes nearly constant and is simultaneously treated as nuisance |
| Absolute phase `phi` | Phase-jitter invariance | Phase is perturbed while physical supervision may ask to encode it |
| SNR | Strict high-quality selection and window normalization | The observed SNR range is truncated and transformed |
| Duration `tau` | Fixed crop plus detector width filters | Extreme durations are censored before learning |

A model cannot be expected to preserve and discard the same information.

**Decision:** the primary yeast representation will preserve event duration,
local Doppler structure, envelope morphology, and multi-peak structure. Global
gain, DC offset, small temporal shifts, absolute phase, and modest acquisition
noise will be treated as nuisances. Absolute amplitude, absolute phase, and
event position are removed from primary physical-fidelity claims. Any later
study of those variables requires a different non-normalized, non-centered
contract.

### 2.4 Training a transformer from scratch on a small curated set is weakly justified

Self-supervision is most useful when the unlabeled pool substantially exceeds
the labeled pool. A transformer trained from scratch on a small collection of
strictly selected event crops can memorize waveform families or solve the
pretext task through local interpolation.

Rare real data are valuable, but rarity changes how they should be used:

- broad physically controlled synthetic data can establish coverage;
- public pretrained encoders provide strong starting points;
- all available unlabeled real recordings can adapt the representation to the
  acquisition domain;
- scarce reviewed yeast labels should be reserved for grouped model selection
  and held-out evaluation.

**Decision:** do not train the custom SSL transformer from random initialization
unless the usable unlabeled pool is demonstrated to be sufficiently larger and
more diverse than the labeled pool. Synthetic pretraining plus real adaptation,
or pretrained MOMENT plus parameter-efficient adaptation, are the primary
routes.

### 2.5 Strict event extraction creates selection bias

The current yeast builder favors events with high robust energy, concentrated
frequency content, acceptable duration, complete centered crops, and strong
quality scores. This produces clean inputs, but it can exclude weak, partial,
long, unusual, or biologically relevant morphologies.

Without a manually reviewed evaluation subset, the detector's recall and
class-dependent selection behavior are unknown. A coherent latent space over
accepted events would only prove that the model represents events preferred by
the extraction algorithm.

**Decision:** create a manually reviewed event-detection audit before SSL. It
must include accepted strict events, medium candidates, rejected candidates,
random background windows, and difficult boundary cases. Report event recall,
precision, duration bias, quality-group composition, and source-group retention.

### 2.6 Centered event-only pretraining is mismatched to broad representation claims

Centered crops are suitable for event classification. They are insufficient for
claims about event discovery, full-trace detection, or background rejection.
They also reduce temporal variability before the encoder can learn it.

**Decision:** real SSL adaptation should include both event-enriched windows and
random windows from continuous recordings. Event-enriched sampling may use an
unlabeled energy detector, but the pretraining pool must not consist exclusively
of accepted strict events. Full-trace detection remains outside this study.

### 2.7 Source-recording leakage is a major risk

One raw recording can generate multiple correlated event crops. If train,
validation, and test splitting occurs after extraction, nearly identical noise,
instrument response, or biological batch signatures can appear on both sides
of evaluation. Random event-level splits would inflate classification,
retrieval, and manifold metrics.

**Decision:** split by original recording and acquisition session before event
extraction. When possible, hold out complete biological or acquisition batches
for the final OOD test. No downstream script may resplit individual events.

### 2.8 Source-folder labels may encode acquisition conditions

Folders such as `budding`, `mix`, `shmoo`, and `shmoo2` are useful provenance
groups, but they are not automatically independent biological ground truth.
Differences in acquisition date, operator, preparation, or instrument settings
can make folder prediction easier than morphology prediction.

**Decision:** define a label dictionary and annotation protocol. Separate
biological label, source group, acquisition session, and quality status in the
metadata. Evaluate whether a classifier can predict acquisition source after
conditioning on biological class; strong source predictability is a warning of
confounding.

### 2.9 Masked reconstruction can be a trivial interpolation task

Very small patches and short masks allow neighboring samples to predict hidden
values without learning event semantics. Overlapping patches can leak masked
content unless guard regions are correct. MSE-like losses also favor smooth or
high-energy regions.

The existing visual patch audit is necessary but not sufficient.

**Decision:** choose patch and mask scales in physical time, relative to event
duration and oscillation periods. Require a mask-leakage test, a persistence or
local-interpolation baseline, and downstream representation metrics. Better
reconstruction alone cannot promote a model.

### 2.10 The hybrid method is not purely self-supervised

Masked reconstruction is self-supervised. Contrastive or regression losses
using known synthetic parameters are physics-supervised. Real labels used for
classification probes are supervised evaluation.

**Decision:** use the following terminology:

- **self-supervised:** targets are derived from the observed real signal;
- **physics-supervised:** targets or pair relations use known simulation
  parameters;
- **supervised downstream:** biological labels train a probe or classifier.

The final method may combine all three, but their contributions must be ablated.

### 2.11 Synthetic-only training is not an adequate solution

Synthetic data provide dense physical coverage and exact parameters, but a model
can learn artifacts of the simulator. A positive synthetic result does not
establish real yeast transfer.

**Decision:** use synthetic data for coverage and controlled physical tests,
then adapt on unlabeled real recordings. The final primary metric must use held-
out real data. Hold out both parameter combinations and simulation variants to
measure simulator overfitting.

### 2.12 Existing model comparisons are pragmatic, not causal

MOMENT, PatchTST, custom SSL, supervised Conv1D-GAP, raw features, and random
features differ in model size, external pretraining data, architecture,
objective, and label access.

**Decision:** report them as systems or representation baselines. Do not infer
that one architecture family or learning paradigm is intrinsically superior.
The causal comparisons are limited to predeclared ablations that hold the
encoder, data, and training budget fixed.

### 2.13 Visualization is not evidence of representation quality

PCA, t-SNE, and UMAP can create visually persuasive clusters even when
generalization is weak. Hyperparameters and sample selection strongly affect
their appearance.

**Decision:** visualizations are supplementary. Primary evidence comes from
grouped held-out probes, label-efficiency curves, cross-recording retrieval,
OOD transfer, and controlled physical-parameter tests.

### 2.14 Real physical estimates can create circular validation

Some real-event physical values are estimated by the same signal-processing
pipeline that detects or filters the events. Correlation with those values can
reward agreement with the preprocessing algorithm rather than independent
physical truth.

**Decision:** known synthetic parameters are the primary physical-fidelity
reference. Real estimated parameters are clearly labeled diagnostic. Any real
physical claim requires an independently measured or manually validated target.

## 3. Repository and Study Ownership

The scientific workflow crosses projects, but ownership must remain explicit.

| Component | Owner | Reason |
|---|---|---|
| Raw yeast ingestion and provenance | `particles2SNR-pipeline` | Dataset generation and source identity |
| Yeast event extraction and reviewed event dataset | `particles2SNR-pipeline` | Signal-processing and dataset ownership |
| Physical signal simulation and sweep dataset generation | `particles2SNR-pipeline` | Reusable generated data and provenance |
| SSL and physics-supervised encoder training | This repository | Representation-learning method |
| Latent sweeps over registered physical datasets | This repository | Evaluation of representation geometry |
| Label-efficiency, retrieval, and robustness assessment | This repository | Downstream representation evidence |
| MOMENT dense-token full-trace detector | P1 | Detection implementation and evaluation |
| Compact TCN-FCOS representation experiments | P2 | Compact detector pipeline |

Latent sweeps should not move to P2 when their question is whether an embedding
preserves physical structure. The simulation code and generated sweep ownership
should move to `particles2SNR-pipeline`; this repository should consume those
sweeps by registered dataset ID.

## 4. Rebuilt Study Definition

### 4.1 Working title

*Physics-Informed Pretraining and Real Self-Supervised Adaptation for
Label-Efficient Yeast Passage Representation*

### 4.2 Primary research question

Does synthetic physics-informed pretraining followed by unlabeled real-domain
adaptation improve grouped, low-label yeast morphology classification and
cross-acquisition transfer over the strongest frozen pretrained and supervised
controls?

### 4.3 Primary endpoint

The primary endpoint is macro F1 on a held-out acquisition group using a frozen
encoder and a linear probe trained with 10% of the labeled training events.
Splits, label fractions, probe regularization, and thresholding are fixed before
final test evaluation.

### 4.4 Secondary endpoints

- area under the label-efficiency curve at 1%, 5%, 10%, 25%, and 100%;
- balanced accuracy and per-class recall;
- nearest-neighbor retrieval purity across different source recordings;
- synthetic held-out recovery of retained physical factors;
- embedding stability under predeclared nuisance perturbations;
- performance on medium-quality events not used in strict-only baselines;
- calibration and uncertainty of downstream class probabilities.

### 4.5 Hypotheses

| ID | Hypothesis | Falsifying result |
|---|---|---|
| H1 | Synthetic pretraining plus real SSL adaptation improves 10%-label OOD macro F1 over the strongest frozen baseline | Improvement is below the minimum effect or its paired interval includes zero |
| H2 | Real adaptation improves over synthetic-only pretraining | No consistent grouped-test improvement |
| H3 | Physics-informed synthetic pretraining improves retained physical-factor organization over reconstruction-only pretraining | No improvement over reconstruction-only or raw features |
| H4 | The representation is stable to declared nuisances without erasing retained morphology | Robustness improves only by collapsing class or physical separation |

### 4.6 Non-goals

- proving transformers are generally better than CNNs;
- building or ranking full-trace detectors;
- predicting absolute amplitude, absolute phase, or event position under the
  normalized centered-crop contract;
- claiming unsupervised biological discovery from t-SNE or UMAP;
- launching a broad architecture or augmentation search;
- merging P3 crop-classification metrics into the P1/P2 detection leaderboard.

## 5. Data Rebuild

### 5.1 Required registered datasets

The data owner should produce separate registered IDs for:

1. immutable raw yeast recordings with acquisition metadata;
2. reviewed yeast event candidates containing strict, medium, rejected, and
   manually annotated rows;
3. unlabeled real pretraining windows sampled from complete recordings;
4. physical synthetic signals with exact generator parameters;
5. the final frozen labeled downstream split.

Names are assigned during registration. Paths must not be hard-coded into this
repository.

### 5.2 Split policy

The split unit is the highest available independent acquisition unit:

```text
biological batch -> acquisition session -> raw recording -> event crop
```

All descendants of one split unit remain in that split. Recommended roles:

| Split | Role |
|---|---|
| Pretraining train | Synthetic and unlabeled real representation learning |
| Pretraining validation | Early stopping and pretext model selection |
| Downstream train | Labeled probe training at fixed fractions |
| Downstream validation | Probe regularization and decision selection |
| In-domain test | Held-out recordings from represented acquisition groups |
| OOD test | Entire held-out acquisition or biological batch |

The final OOD test is opened once after the method and probe protocol freeze.

### 5.3 Manual review requirement

Before model training, annotate a stratified event-detection audit set covering:

- every biological/source group;
- multiple acquisition sessions;
- strict, medium, and rejected candidates;
- low, median, and high energy;
- short, typical, and long durations;
- single- and multi-Doppler patterns;
- random background windows;
- crop-boundary failures.

The audit must report event-level precision and recall with uncertainty and the
retention rate by group, quality, duration, and acquisition. Failure to establish
acceptable event recall blocks representation claims.

Before annotation, the Gate 1 detector thresholds are frozen as follows:

- retained-candidate precision at least `0.90`, with Wilson 95% lower bound at
  least `0.80`;
- retained-event recall on complete traces at least `0.85`, with Wilson 95%
  lower bound at least `0.75`;
- retained precision at least `0.75` and recall at least `0.70` in every source
  group represented in the audit;
- no more than `0.25` of reviewed rejected candidates may contain a true event;
- at least two independent acquisitions are required for acquisition-OOD
  readiness.

Report both the balanced stratified audit estimates with Wilson intervals and
population-weighted point estimates using stratum expansion weights. These are
dataset-validity thresholds, not downstream performance endpoints. The
registered candidate-review CSV files are immutable templates; completed
annotations are stored and registered separately.

### 5.4 Canonical input selection

Candidate physical windows are evaluated before training, for example raw 4096
and raw 8192 samples at 2 MHz. Selection is based on event coverage and required
context, not downstream model performance.

The chosen contract must include:

- sampling frequency and physical duration;
- raw or filtered signal identity;
- crop centering and boundary behavior;
- anti-aliasing policy if resampling is used;
- normalization and retained physical variables;
- allowed missing/padded region mask;
- exact tensor shape and dtype.

Once selected, every compared encoder receives the same numerical input unless
an explicitly labeled native-input sensitivity analysis is performed.

### 5.5 Unlabeled real sampling

Real adaptation should combine:

- random windows from complete recordings;
- event-enriched windows from a label-free energy proposal mechanism;
- strict and medium candidate windows;
- background and difficult windows.

Sampling proportions are fixed from the data audit. Biological labels are not
used during real SSL adaptation.

### 5.6 Synthetic coverage

The generator must vary retained signal factors and nuisance factors
independently where physically possible. Training and test simulations use
disjoint seeds, held-out parameter combinations, and at least one held-out
generator variant. Simulator metadata must include parameter definitions,
units, ranges, sampling distributions, and generator revision.

Domain randomization is bounded by the acquisition audit and the simulator's
validated regime. It does not mean randomizing every parameter over arbitrary
ranges. Retained factors must remain recoverable and sufficiently covered;
nuisance factors should vary enough to prevent simulator- or acquisition-specific
shortcuts. Implausible combinations are excluded or explicitly labeled as
stress tests.

Synthetic coverage does not justify extrapolation beyond the validated real
domain.

## 6. Representation and Loss Design

### 6.1 Declared information policy

| Factor | Primary role | Training treatment | Evaluation treatment |
|---|---|---|---|
| Duration/envelope morphology | Preserve | Physics-aware positives and reconstruction | Synthetic recovery and downstream probes |
| Doppler/frequency structure | Preserve | Physics-aware positives and spectral diagnostic | Held-out synthetic and real diagnostic |
| Multi-peak structure | Preserve | Sample coverage and reconstruction | Stratified retrieval/probe analysis |
| Global gain | Nuisance | Modest scale consistency | Robustness only |
| DC offset | Nuisance | Remove or augment | Robustness only |
| Small crop shift | Nuisance | Consistency augmentation | Robustness only |
| Absolute phase | Nuisance under current task | Do not supervise | No physical-fidelity claim |
| Event position in crop | Nuisance | Center/jitter consistently | No physical-fidelity claim |
| Acquisition noise | Nuisance within bounded range | Real adaptation and noise consistency | OOD and robustness |

No loss may encourage invariance to a factor that is also a primary recovery
target.

Before training, every candidate variable is assigned one of three roles:
`preserve/predict`, `randomize/invariant`, or `unresolved/excluded`. The register
must record its physical meaning, units, evidence for plausible ranges, allowed
transformations, and downstream relevance. An unresolved variable is not used
as an augmentation or supervision target until its role is justified. This is
especially important for amplitude and temporal scaling, which may encode yeast
properties in one task and acquisition variation in another.

### 6.2 Training stages

**Stage A: baseline extraction.** Freeze public pretrained encoders and compute
raw, random, MOMENT, and PatchTST representations on the frozen data contract.

**Stage B: physics-informed synthetic pretraining.** Train the custom encoder
with masked reconstruction and supervision only on retained physical factors.
This stage is named physics-supervised, not self-supervised.

**Stage C: real self-supervised adaptation.** Start from Stage B and adapt on a
predeclared mixture of unlabeled real windows and synthetic replay, then move to
a real-dominant mixture if validation supports it. Synthetic replay limits
catastrophic forgetting of retained physical factors; it is not a source of
biological labels. Use masked reconstruction and consistency only for declared
nuisances, and do not use biological labels or source-folder labels. Report the
mixture schedule and include A3 so any gain from real adaptation is identifiable.

**Stage D: frozen downstream evaluation.** Freeze every encoder and train the
same linear-probe implementation at each label fraction.

**Stage E: optional parameter-efficient adaptation.** Run only if Stage D shows
useful signal and a predeclared gate authorizes it. Full MOMENT fine-tuning is
not part of the initial study.

### 6.3 Masking gate

Mask size is specified in milliseconds and converted to samples/tokens under
the frozen input contract. The selected policy must:

- hide complete physically meaningful regions rather than isolated points;
- prevent overlap leakage through guard bands;
- preserve enough context to make reconstruction possible but nontrivial;
- avoid systematically masking only event centers or only background;
- beat persistence, linear interpolation, and local convolution baselines;
- improve at least one downstream validation metric over reconstruction-free
  controls.

### 6.4 Deferred simulation-to-real methods

Explicit domain alignment, adversarial domain confusion, pseudo-labeling, and
inverse reconstruction through a differentiable simulator are not part of the
initial A0-A4 study. They are authorized only as preregistered follow-ups when
the domain diagnostic identifies a material gap and the simpler Stage C recipe
fails to reduce it without losing retained information.

Any alignment loss must target documented acquisition or simulator nuisances,
not erase all simulation-real differences. Simulator inversion additionally
requires an identifiability audit, uncertainty estimates, and robustness to
forward-model misspecification. A low reconstruction residual alone is not
evidence that inferred physical parameters are correct.

## 7. Baselines and Controlled Ablations

### 7.1 Required baselines

| Baseline | Purpose |
|---|---|
| Raw normalized signal + linear model | Detect whether learned features add value |
| Handcrafted time/frequency features | Strong domain baseline |
| Random encoder | Detect architecture and dimensionality artifacts |
| Supervised Conv1D-GAP | Practical small supervised reference |
| Frozen official MOMENT | Strong public pretrained reference |
| Frozen pretrained PatchTST | Alternative public time-series reference |
| Reconstruction-only custom encoder | Isolate physics-informed contribution |

The supervised CNN uses labels and is not interpreted as an SSL control with
equal supervision. Its role is to show whether SSL is useful enough to justify
its complexity.

### 7.2 Minimal ablation matrix

| ID | Synthetic physics stage | Real SSL adaptation | Purpose |
|---|---|---|---|
| A0 | No | No | Raw/random/public pretrained baselines |
| A1 | No | Yes | Real-only self-supervision |
| A2 | Reconstruction only | No | Synthetic reconstruction control |
| A3 | Physics-informed | No | Synthetic-only physical pretraining |
| A4 | Physics-informed | Yes | Proposed complete method |

Use one encoder architecture and one fixed budget for A1-A4. Do not add model
families until this causal matrix is complete.

## 8. Evaluation Protocol

### 8.1 Statistical design

- Freeze one split manifest and reuse it for every method.
- Train at least three representation seeds; use five when compute permits.
- Repeat probe sampling and initialization with paired seeds.
- Report mean, standard deviation, and paired bootstrap or hierarchical
  bootstrap intervals respecting acquisition groups.
- Select hyperparameters on validation groups only.
- Keep the final OOD test sealed until the method freeze.
- Report all completed predeclared cells, including failed runs.

### 8.2 Label-efficiency protocol

Use fixed stratified fractions of downstream training labels: 1%, 5%, 10%, 25%,
and 100%. Sampling occurs within acquisition groups without moving events across
splits. The primary 10% result and area under the label-efficiency curve are
reported with identical probe code for every frozen encoder.

### 8.3 Retrieval protocol

Queries and neighbors must come from different raw recordings. Report top-1 and
top-k biological-label purity, acquisition-source purity, and quality-group
purity. High biological purity accompanied by high acquisition purity is not
automatically a success.

### 8.4 Physical-fidelity protocol

Use held-out synthetic parameter combinations and generator variants. Evaluate
only retained factors with regression, rank correlation, local-neighborhood
continuity, and controlled one-factor sweeps. Real estimated parameters remain
diagnostic.

### 8.5 Robustness protocol

Perturbations are bounded by the measured real acquisition variability. Report
embedding distance and downstream prediction stability under gain, offset,
small temporal shift, bounded noise, and limited missing regions. Also verify
that robustness is not caused by collapsed embeddings.

### 8.6 Visualization policy

PCA, t-SNE, and UMAP use predeclared balanced samples and settings. They support
interpretation but cannot pass a gate. Every visualization is accompanied by a
quantitative grouped-test metric.

### 8.7 Simulation-to-real domain diagnostic

Train the same low-capacity probe to predict `synthetic` versus `real` from the
frozen embeddings of A2, A3, and A4. Use balanced samples, grouped real splits,
held-out simulation seeds, and identical preprocessing. Report ROC AUC and
balanced accuracy alongside retained-factor recovery and downstream OOD
performance.

High domain-prediction accuracy indicates that domain information remains, but
it is not by itself proof of a failed representation because simulated and real
physical populations may genuinely differ. Conversely, chance-level domain
prediction is not automatically desirable. A reduction in domain separability
counts as useful adaptation only when retained physical information and
downstream performance are preserved or improved. Where metadata permit,
repeat the diagnostic conditional on comparable acquisition and physical
variables to isolate simulator or sensor shortcuts from population shift.

## 9. Promotion Gates

### Gate 0: provenance and ownership

Pass only if all inputs are registered, manifests contain source identity and
acquisition groups, generated runs have `run.json`, and data-generation code is
owned by `particles2SNR-pipeline`.

### Gate 1: event and split validity

Pass only if the reviewed event audit is acceptable, no source recording crosses
splits, duplicate-content checks pass, and source-group confounding is measured.

### Gate 2: input and nuisance contract

Pass only if one physical input contract is frozen and every loss,
normalization, augmentation, and metric agrees with the information-policy
register. No unresolved variable may be used as an invariance or supervision
target.

### Gate 3: baseline readiness

Pass only if raw, handcrafted, random, supervised CNN, MOMENT, and PatchTST
baselines run through the same grouped evaluation and produce complete manifests.

### Gate 4: pretext validity

Pass only if reconstruction beats trivial interpolation baselines without mask
leakage and the encoder avoids representation collapse. This gate does not
establish downstream usefulness.

### Gate 5: scientific promotion

The proposed A4 method is promoted only if:

1. its mean 10%-label OOD macro-F1 improvement over the strongest frozen
   eligible baseline under the same label budget is at least the predeclared
   minimum effect of practical interest;
2. the paired uncertainty interval for that improvement excludes zero;
3. real adaptation improves over synthetic-only A3;
4. retained physical-factor metrics improve over reconstruction-only A2;
5. robustness does not erase class or retained physical separation;
6. no major class, acquisition group, or quality stratum collapses;
7. simulation-real separability is reported and any apparent alignment gain is
   shown not to come from representation collapse or loss of retained factors.

If these conditions fail, report the method as a negative result. Do not rescue
it through unplanned architecture search or test-set tuning.

The provisional minimum effect is `0.03` macro F1. Phase 1 must compare this
value with grouped baseline variability and domain relevance, justify the final
value, and freeze it before any A1-A4 test result is opened.

### Gate 6: optional methods

Parameter-efficient or full encoder adaptation is authorized only after Gate 5
or a separately preregistered negative-result follow-up. It cannot replace the
frozen-representation evaluation. Explicit domain alignment and differentiable
simulator inversion follow the same rule and require the additional audits in
Section 6.4.

## 10. Experiment Sequence

### Phase 0: redesign without training

1. Inventory raw recordings, acquisition metadata, labels, and current event
   crops.
2. Audit the two incompatible 4096 input paths.
3. Define the retained/nuisance information policy.
4. Design the reviewed event audit and grouped split.
5. Move or expose dataset generation through `particles2SNR-pipeline`.
6. Register all immutable input datasets.

**Exit condition:** Gates 0-2 pass.

### Phase 1: frozen baselines

1. Run raw and handcrafted features.
2. Run random encoder controls.
3. Extract frozen MOMENT and PatchTST embeddings.
4. Train the supervised Conv1D-GAP reference.
5. Execute grouped label-efficiency, retrieval, physical, and robustness
   evaluation.

**Exit condition:** Gate 3 passes and the difficulty of the task is quantified.

### Phase 2: causal representation matrix

1. Run A1 real-only SSL.
2. Run A2 synthetic reconstruction-only.
3. Run A3 physics-informed synthetic-only.
4. Run A4 physics-informed synthetic plus real SSL adaptation.
5. Run the predeclared simulation-real domain diagnostic for A2-A4.
6. Complete all predeclared seeds before interpreting winners.

**Exit condition:** Gate 4 passes for valid runs.

### Phase 3: final evaluation

1. Freeze the encoder and probe protocol.
2. Open the OOD test once.
3. Run paired statistical comparisons.
4. Generate the final tables and figures from manifested results.
5. Apply Gate 5 without post-hoc exceptions.

### Phase 4: optional extension

Only after the final decision, consider parameter-efficient adaptation, new
biological groups, or a separate full-trace representation study. These are not
required to complete the rebuilt paper.

## 11. Stop Rules

Stop custom from-scratch SSL and use pretrained encoders if the audited
unlabeled pool is not substantially larger or more diverse than the labeled
pool.

Stop representation training if event extraction cannot be validated without
severe class or quality selection bias.

Stop synthetic expansion if held-out generator performance rises while real
adaptation and real OOD performance do not.

Stop architecture search if A4 fails Gate 5. The negative result is then that
the proposed physics-informed plus real-SSL recipe did not outperform strong
pretrained or supervised controls under the frozen protocol.

## 12. Figure and Table Plan

### Main figures

1. **Data and split schema:** raw recordings, source-group split, event audit,
   synthetic pretraining, real adaptation, and sealed OOD evaluation.
2. **Label-efficiency curves:** macro F1 versus labeled fraction with grouped
   uncertainty.
3. **In-domain versus OOD transfer:** performance by acquisition group and
   quality stratum.
4. **Physical-fidelity panel:** held-out synthetic factor recovery for retained
   variables only.
5. **Robustness versus information retention:** nuisance stability plotted
   against biological and physical separation.
6. **Simulation-to-real adaptation:** domain-probe separability before and
   after real adaptation plotted against retained-factor recovery and OOD
   performance.

### Main tables

1. dataset, split-unit, acquisition, event-quality, and class counts;
2. model size, pretraining source, supervision type, and adaptation mode;
3. primary and secondary metrics with uncertainty;
4. causal A1-A4 ablation results;
5. failure, stop-rule, and final-decision summary.

PCA/t-SNE/UMAP, reconstruction examples, and retrieval galleries belong in the
supplement unless they explain a specific failure.

## 13. Documentation and Artifact Contract

Every run must include:

- registered dataset IDs and manifest hashes;
- repository revision and configuration hash;
- split-unit and source-identity summaries;
- seed, model mode, supervision type, and input contract;
- complete metrics, predictions, and failure status;
- a valid `run.json` under
  `artifacts/unsupervised-learning-flow-cytometry/<run-id>/`;
- checksums for publication tables and figures.

The final study report must distinguish implementation history, exploratory
diagnostics, predeclared evidence, and post-hoc analysis. Failed runs remain in
the evidence inventory.

## 14. Status of Existing Components

| Existing component | Decision |
|---|---|
| [`p3_4096_pipeline.md`](p3_4096_pipeline.md) | Historical implementation reference pending a new physical input contract |
| [`yeast_event_dataset_detection.md`](yeast_event_dataset_detection.md) | Retain implementation detail; require manual recall/selection-bias validation and move dataset ownership |
| [`patch_stride_masking_plan.md`](patch_stride_masking_plan.md) | Retain as an initial visualization gate; add quantitative leakage and downstream gates |
| `configs/p3_ssl_moment_v0.yaml` | Legacy first reconstruction experiment; do not launch as final protocol |
| `configs/p3_ssl_hybrid_physics.yaml` | Legacy broad hybrid design; replace contradictory factor/invariance policy |
| MOMENT/PatchTST embedding code | Retain as frozen baseline infrastructure |
| Conv1D-GAP same-input code | Retain as supervised practical control |
| Latent physical sweeps | Retain in this project as representation evaluation; move generation/provenance to the data owner |
| Yeast template and simulation generation | Move ownership to `particles2SNR-pipeline` |
| MOMENT full-trace detector | Keep in P1; cite only as downstream transfer context |

Existing artifacts are not deleted. They remain exploratory or historical until
classified by data contract, split safety, and evidence maturity.

## 15. Immediate Next Actions

No GPU training is needed for the next step.

1. Complete the 73 candidate-window and 73 full-trace v5 reviews for the
   current acquisition and run the frozen Gate 1 analyzer.
2. Acquire and document a second genuinely independent yeast session.
3. Ingest it through the data-owner
   [`YEAST_ACQUISITION_INTAKE.md`](https://github.com/juju27fun/particles2SNR-pipeline/blob/reorg/workspace-20260710/YEAST_ACQUISITION_INTAKE.md)
   protocol, with the current session as development and the new session as
   `sealed_ood_test`.
4. Build the acquisition-stratified candidate audit and require per-acquisition
   detector precision and recall to pass without changing the frozen detector.
5. If Gate 1 passes, register new source-index, candidate, and representation
   versions; otherwise treat the inspected acquisition as development and
   reserve a third acquisition for final OOD evaluation.
6. Run the already frozen multi-seed A0-A4 matrix using development splits only.
7. Freeze the winning baseline, probe, statistical comparisons, and report
   layout, then open the sealed acquisition once for Gate 5.

## 16. Definition of Done

The rebuilt study is complete when:

- data and simulation ownership is correct;
- one physical input contract is used consistently;
- event extraction is independently validated;
- source and acquisition leakage are excluded;
- nuisance and retained variables are coherent with preprocessing and losses;
- raw, handcrafted, random, supervised, and pretrained baselines are complete;
- the minimal A1-A4 matrix is complete over all seeds;
- the sealed OOD test is evaluated once;
- uncertainty and subgroup failures are reported;
- Gate 5 produces an unambiguous positive or negative decision;
- a complete evidence-linked specialist report can be written without relying
  on latent-space appearance alone.

## Final Position

Self-supervision remains justified only if the project can exploit a genuinely
larger and more diverse unlabeled real pool, or if synthetic physical coverage
plus real adaptation yields measurable label efficiency and OOD transfer. The
strongest path is not real-only SSL on a small set of strict event crops, and it
is not synthetic-only training. It is a controlled sequence of physics-informed
synthetic pretraining, unlabeled real adaptation, and grouped held-out real
evaluation against strong pretrained and supervised baselines.

Until the redesign gates pass, the current P3 outputs are exploratory
representation evidence, not a frozen yeast SSL conclusion.
