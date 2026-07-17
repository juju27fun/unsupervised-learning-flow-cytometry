# Yeast SSL objective rescue - execution log

Status: active development study. No sealed split has been opened.

## Question

Why did controlled A1 masked reconstruction remain flat, and can a corrected
self-supervised target produce a nontrivial, non-collapsed representation?

The promotion criterion is not loss decrease alone. A candidate must beat a
zero predictor, reconstruct at nontrivial amplitude, retain healthy embedding
geometry, and later improve a development-only utility endpoint.

## 1. A1 retrospective control

Artifact:
`artifacts/unsupervised-learning-flow-cytometry/audits/yeast-a1-reconstruction-controls-v1`

Across seeds 42, 43, and 44:

- model masked MSE was marginally worse than zero prediction;
- model output RMS was only `0.0067-0.0093` of target RMS;
- approximately 26% of loss-mask samples were in the labeled event region;
- the earlier interpolation-only control therefore produced a false pretext
  pass.

Conclusion: A1 learned the near-zero conditional-mean solution. The previous
"pretext pass with warning" interpretation must be replaced by a pretext
failure.

## 2. Optimization and alignment sanity check

Artifact:
`artifacts/unsupervised-learning-flow-cytometry/audits/yeast-reconstruction-fixed-overfit-v1`

The production Transformer, decoder, masking path, and loss can memorize fixed
real targets:

| Fixed batch | Relative MSE reduction vs zero | Output/target RMS |
|---|---:|---:|
| 1 signal | 0.992 | 1.020 |
| 8 signals | 0.921 | 0.976 |

Conclusion: a basic gradient, decoder, target-alignment, or capacity defect is
not the explanation for A1.

## 3. Single-gap predictability map

Artifact:
`artifacts/unsupervised-learning-flow-cytometry/audits/yeast-mask-predictability-v2`

On 256 development-validation real signals, event-region interpolation reduced
MSE versus zero by `0.968` at 8 us, `0.805` at 16 us, and only `0.096` at 32 us.
It was worse than zero from 64 us onward. A local harmonic fit remained strongly
effective on simulated events through 512 us, but not on real events.

Conclusion: the frozen A1 range of 128-512 us was outside the empirically
predictable real-waveform regime. Simulation is also substantially more
harmonically predictable than reality.

## 4. Complete masking-policy audit

Artifacts:

- rejected naive policy audit:
  `artifacts/unsupervised-learning-flow-cytometry/audits/yeast-mask-policy-predictability-v1`;
- corrected patch-policy audit:
  `artifacts/unsupervised-learning-flow-cytometry/audits/yeast-mask-policy-predictability-v2`.

Naively shortening time blocks did not work. Patch expansion and the 16-sample
guard hid 74% of the trace at a nominal 25% target ratio and 40% at a nominal
10% ratio. Patch-aligned isolated masking removes this amplification:

| Policy | Target ratio | Hidden ratio | Event fraction | Interpolation gain vs zero |
|---|---:|---:|---:|---:|
| P25 | 0.250 | 0.250 | 0.280 | 0.817 |
| P10 | 0.102 | 0.102 | 0.296 | 0.815 |
| PE25 | 0.250 | 0.250 | 0.456 | 0.849 |
| PE10 | 0.102 | 0.102 | 0.780 | 0.869 |

P/PE policies hide isolated complete 16-sample patches with at least one visible
token between targets. PE policies bias selection toward labeled event regions.
A later bidirectional ridge-autoregressive control independently confirmed the
result: it beat zero on every signal for all P/PE policies, with relative MSE
improvements of `0.50-0.72`, while it did not improve the frozen L0 policy.
This result is stored in `yeast-mask-policy-predictability-v3`.

## 5. Training decision

Protocol: `configs/yeast_ssl_mask_ablation_v1.yaml`.

The first meaningful training matrix holds architecture, pointwise target,
optimizer, epochs, dataset, and seed fixed. It changes only mask ratio and event
bias across P25, P10, PE25, and PE10. All four one-epoch CPU smokes completed,
produced finite nonzero predictions, and generated valid run manifests. Their
scientific gates are intentionally not interpreted after four optimizer steps.

Next action: run the equal-budget seed-42 matrix on the pfcalcul Jupyter GPU
runner with
`orchestration/pfcalcul/scripts/pfcalcul_jupyter_yeast_mask_ablation.sh`.
Generate the gate report before any downstream evaluation or additional seed.
If every candidate still collapses, retain the corrected mask and move to a
phase-invariant target or explicit anti-collapse objective rather than tuning
the Transformer.

## 6. Conditional branches frozen before seed-42 results

The exact branch policy is stored in
`configs/yeast_ssl_mask_ablation_v1.yaml`:

1. If reconstruction, amplitude, and geometry all pass, run development-only
   utility evaluation without more training.
2. If reconstruction passes but geometry fails, select the mask policy with the
   largest zero-baseline improvement among nontrivial-amplitude candidates and
   compare time-only C0 with time plus VICReg C1. The VICReg global weight is
   fixed at `1.0`; the historical `0.10` term contributed only about 1-2% of
   the real objective and produced only a modest rank increase.
3. If reconstruction still fails, replace pointwise waveform prediction with a
   phase-invariant envelope/energy or local log-power target. Do not reuse the
   historical STFT implementation, which computes spectra after multiplying by
   a sparse time mask and therefore measures mask-edge artifacts.
4. If development utility fails, reject the mask-only rescue. Seeds 43 and 44
   are authorized only after seed 42 passes pretext, geometry, and utility.

## 7. Full seed-42 mask matrix and anti-collapse handoff

pfcalcul job `20260717_yeast_mask_ablation_dev_s42_v1` completed with return
code 0. All four runs and the source report were retrieved and manifest
validated. The provenance-hardened local decision report is
`artifacts/unsupervised-learning-flow-cytometry/reports/yeast-mask-ablation-dev-s42-v3`.

| Policy | Model MSE | Zero MSE | Interpolation MSE | Gain vs zero | Output/target RMS | Rank | Cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| P25 | 0.156 | 1.258 | 0.270 | 0.876 | 0.943 | 3.65 | 0.99985 |
| P10 | 0.953 | 1.398 | 0.293 | 0.318 | 0.523 | 2.28 | 0.99825 |
| PE25 | 0.135 | 1.521 | 0.289 | 0.911 | 0.978 | 2.74 | 0.99982 |
| PE10 | 0.259 | 2.350 | 0.413 | 0.890 | 0.933 | 2.06 | 0.99866 |

Conclusion: corrected masking makes the SSL reconstruction objective learn.
Every candidate beats zero with nontrivial amplitude, and PE25 also beats the
strong interpolation control by 53%. However, every global embedding still
fails both frozen geometry gates (`rank >= 8`, `cosine <= 0.95`). The central
failure has therefore moved from waveform predictability to representation
collapse. PE25 is selected by the frozen rule for the C0/C1 contrast.

The next protocol is frozen separately in
`configs/yeast_ssl_mask_collapse_v1.yaml`. C0 and C1 share initialization,
first-view masks, data order, architecture, optimizer, epochs, and a drop-last
training policy. C1 adds a second independently masked view and VICReg at
global weight 1.0. The drop-last rule fixes a pre-run defect found by an
independent audit: 7,009 training events leave a singleton batch that VICReg
cannot evaluate. CPU smokes now complete for both cells with identical initial
model hashes. These one-epoch smokes are execution checks, not results.

C1 proceeds to development utility only if it beats zero and interpolation,
has nontrivial output amplitude, reaches both absolute geometry thresholds,
and improves rank and cosine relative to C0. Otherwise the next action is the
already declared phase-invariant target contrast. No outcome from this stage
alone authorizes seeds 43 or 44 or opens a sealed split.

## 8. Conditional next-stage designs frozen while C0/C1 runs

The two downstream designs were audited before C0/C1 completed. They remain
conditional and must not be run together.

If C1 passes every gate, the primary development-utility endpoint is macro-F1
at 10% labels, averaged over probe seeds 42, 43, and 44. C1 is compared with
C0, a seed-42 untrained encoder, raw-signal features, and a 96-component PCA
fit without labels on `development_train` only. The primary comparator is the
strongest of these frozen eligible baselines. A utility pass requires a macro-F1
gain of at least 0.03 and a capture-block paired 95% interval with lower bound
above zero; C1 must also have positive point estimates versus raw, random, and
PCA individually. Label fractions 1%, 5%, 25%, and 100%, retrieval, robustness,
and calibration remain secondary. This is exploratory single-acquisition
source-group proxy evidence, not confirmatory morphology evidence. The loader
must use physically development-only metadata rather than reading a combined
metadata file before filtering.

If C1 fails, train one F1 target at seed 42 with the same PE25 masks, encoder,
optimizer, epochs, and data order as C0. F1 predicts 14 phase-invariant features
per hidden 16-sample token: relative analytic envelope and relative analytic
energy, each pooled into four 4-sample, two 8-sample, and one 16-sample bins,
then transformed with `log1p`. Features are computed from the complete signal
before any mask is applied. The first branch excludes wavelets and local
log-power because they add unresolved resolution choices; it also excludes the
historical sparse-mask STFT loss, which measures mask-edge spectra.

F1 must satisfy phase-rotation relative target error at most 0.002, mask-target
independence, at least 80% one- and eight-example overfit improvement over the
better zero/constant baseline, held-out improvement over zero, train-derived
constant, and feature-of-interpolation baselines, output/target RMS at least
0.10, effective rank at least 8, and mean pairwise cosine at most 0.95. Event
and background errors are reported separately. Utility and additional seeds
remain forbidden until the complete seed-42 gate sequence passes.

## 9. Full C1 development utility result

Pfcalcul job `20260717_yeast_collapse_utility_full_v2` completed with return
code 0 and reused the complete artifact produced during the interrupted-status
first attempt. The retrieved and manifest-validated artifact is
`artifacts/unsupervised-learning-flow-cytometry/runs/yeast-collapse-utility-full-s42-v1`.
No sealed split was opened, and all 75 logistic probes converged.

At the frozen 10% label endpoint, averaged over probe seeds 42, 43, and 44:

| Method | Macro-F1 | C1 minus method | Descriptive block interval |
|---|---:|---:|---:|
| C1 | 0.294 | - | - |
| C0 | 0.310 | -0.016 | `[-0.036, +0.013]` |
| Random encoder | 0.301 | -0.007 | `[-0.027, +0.017]` |
| Raw signal | 0.259 | +0.035 | `[-0.013, +0.080]` |
| Train-only PCA96 | 0.237 | +0.057 | `[+0.022, +0.090]` |

C1 retains the intended geometry on all development events (effective rank
`9.81`, cosine `0.935`) and beats raw and PCA point estimates. It nevertheless
fails the frozen utility gate because it does not beat either the matched C0
control or the random encoder at the primary endpoint. Seeds 43 and 44 are not
authorized for the mask-only representation.

An independent post-run audit confirmed row alignment, disjoint train and
validation records/blocks, train-only PCA fitting, matched probe subsets, and
probe convergence. It also found that the nominal interval is not
confirmatory: validation contains only 20 class-pure capture blocks, distributed
as 12 `mix`, 4 `shmoo2`, 3 `budding`, and 1 `shmoo`. The result is therefore a
post-selection, single-acquisition proxy diagnostic. It supports rejecting
mask-only C1 but cannot estimate independent `shmoo` variability or establish
morphology generalization.

The original utility gate omitted the known handcrafted baseline. A separate
post-hoc supplement is frozen in
`configs/yeast_ssl_collapse_utility_supplement_v1.yaml` before computing those
features. It compares C1 with signal-only handcrafted features, full
handcrafted detector diagnostics, and their fusion. It cannot reverse the
mask-only rejection; its purpose is to complete the baseline and
complementarity record.

If that supplement does not reveal a practical fusion gain, the one remaining
objective experiment is real-only and phase-invariant. It must preserve the C1
PE25 masking, VICReg weight 1.0, architecture, optimizer, and data order so that
only the prediction target changes. Simulation-assisted training remains
ineligible because both simulator versions failed common-support gates. The
historical masked-zero STFT loss also remains ineligible because mask edges
contaminate its spectra.

## 10. Handcrafted supplement and final target amendment

Artifact:
`artifacts/unsupervised-learning-flow-cytometry/reports/yeast-collapse-utility-supplement-v1`.

All primary probes converged. At 10% labels, averaged over probe seeds:

| Comparison | Macro-F1 difference | Descriptive block interval |
|---|---:|---:|
| C1 minus signal-only handcrafted | -0.059 | `[-0.092, -0.016]` |
| C1 minus full handcrafted | -0.098 | `[-0.124, -0.050]` |
| Full handcrafted plus C1 minus full handcrafted | -0.015 | `[-0.042, +0.024]` |

The supplement therefore confirms the mask-only rejection and finds no
practical complementarity signal. The full handcrafted point estimate is
`0.392`, compared with `0.294` for C1. The intervals remain descriptive for the
20-block support limitation stated above. No additional mask-only seed or
sealed evaluation is authorized.

The earlier conditional F1 envelope/energy design is not run unchanged. Later
evidence shows that envelope features are the weakest handcrafted family while
frequency features are materially useful. Removing VICReg from F1 would also
confound target choice with a renewed collapse mechanism. This is recorded as
a prospective amendment, not mislabeled as the earlier preregistered F1 cell.

The final objective candidate is S1: C1 with only waveform prediction replaced
by local analytic log-power prediction. S1 keeps PE25, two mask views, VICReg
weight 1.0, the encoder, optimizer, data order, epochs, and seed 42 fixed. For
each complete unmasked 1 MHz trace, it computes one 256-sample Hann-windowed
analytic FFT per token, keeps the 24 positive-frequency bins from 7.8125 to
97.65625 kHz, normalizes power by the trace-wide mean retained power, and
applies `log1p`. Only center tokens 8 through 247 are valid. The target is never
computed from masked zeros.

Before training, S1 must pass exact shape/frequency/dtype tests, phase-rotation
relative error at most 0.002, mask-target independence, finite nonzero encoder
and head gradients, and one/eight-example overfit improvement of at least 80%
over the better zero or train-constant predictor. Held-out target prediction
must beat zero, train constant, and features computed after waveform
interpolation; output/target RMS must be at least 0.10; rank must be at least 8;
and cosine must be at most 0.95. Event, background, and event-boundary frames
are reported separately. Failure of any sequential gate ends objective rescue
without simulation, another target sweep, or a sealed split.
