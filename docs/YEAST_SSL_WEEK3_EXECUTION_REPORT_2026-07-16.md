# Yeast SSL Follow-Up: Week 3 Execution Report

**Date:** 2026-07-16  
**Status:** complete controlled negative  
**Decision:** `stop_common_support_failed`  
**Representation retraining:** not authorized  
**Opening `followup_test`:** not authorized

## Question

Week 1 showed that the v1 Gaussian simulator and real yeast candidates had too
little common support for a defensible conditional domain comparison. Week 3
tested one causal correction:

> Does replacing the diffuse Gaussian event envelope with a train-calibrated,
> finite-support transit envelope make simulated and real signals sufficiently
> comparable?

This is a simulator-diagnosis experiment. It does not test a new SSL model and
cannot establish biological realism.

## Frozen Method

The correction used only `followup_train`:

1. Measure Hilbert-envelope duration above 25% of peak.
2. Store 101 quantiles over the 5th-95th percentile range.
3. Sample one duration per latent, shared by both nuisance views.
4. Generate a finite-support Tukey packet with alpha `0.25`.

Carrier frequency, component priors, position, SNR, colored noise, drift,
sensor response, RMS, and preprocessing remained unchanged. The corrected
dataset contains 7,000 latents and 14,000 views: 10,000 train, 2,000 validation,
and 2,000 held-out synthetic-test views.

The preflight records `validation_signals_accessed = 0` and
`sealed_splits_used = []`. The validation evaluation read
`followup_validation`; neither `followup_test` nor the exhausted
`in_session_test` was opened.

## Primary Result

The correction improved every primary support statistic but failed every
frozen gate.

| Simulator | Train retained | Validation retained | Train max SMD | Validation max SMD |
|---|---:|---:|---:|---:|
| v1 Gaussian | 0.146 | 0.162 | 1.946 | 2.342 |
| v2 finite support | 0.323 | 0.374 | 0.459 | 0.641 |
| Frozen requirement | >=0.500 | >=0.500 | <=0.250 | <=0.250 |

The train-to-validation direction replicates: validation retention rises by
`0.212`, while validation maximum SMD falls by `1.701`. This is a meaningful
simulator improvement, but not enough common support for the intended bridge.

## Residual Mismatch

At the primary caliper, corrected-validation SMD is:

| Observable | SMD |
|---|---:|
| Duration | 0.253 |
| Dominant frequency | 0.091 |
| RMS | 0.070 |
| SNR estimate | 0.487 |
| Spectral peak count | 0.641 |
| Event offset | 0.017 |

Duration is nearly at the bound, consistent with the intervention doing what
it was designed to do. Spectral peak count and SNR remain the limiting
families. Signal-summary permutation importance is dominated by spectral power
in 80-100 kHz (`0.420`) and 40-60 kHz (`0.180`), indicating a residual spectral
signature rather than only a duration error.

## Sensitivity

Changing the matching caliper does not rescue the result.

| Caliper | Train retained | Validation retained | Train max SMD | Validation max SMD |
|---:|---:|---:|---:|---:|
| 1.00 | 0.270 | 0.260 | 0.339 | 0.453 |
| 1.50 | 0.323 | 0.374 | 0.459 | 0.641 |
| 2.00 | 0.335 | 0.404 | 0.511 | 0.711 |

The stricter caliper improves balance but loses support; the looser caliper
gains too little support while worsening balance. The stop decision is not an
artifact of the primary caliper.

## Domain Probes

Corrected matched linear AUCs were `0.750` for observables, `0.998` for signal
summaries, and `0.515` for downsampled raw signals. The signal-summary 95%
group-bootstrap interval is `[0.996, 1.000]`.

These AUCs are exploratory because the common-support gate failed. They cannot
be reported as valid conditional domain-separability estimates. They do show
that a strong residual handcrafted spectral signature remains after matching.

## Critical Interpretation

The finite-support intervention is a successful causal ablation and a failed
bridge. It corrected most of the duration mismatch and substantially improved
overlap on unseen validation data, so this is not an implementation failure.
However, one compact wave packet still produces spectral structure unlike the
real candidate population, and the unchanged SNR/noise family remains
miscalibrated.

Adding a second correction now would be outcome-driven continuation of the same
validation study. The preregistered stop rule therefore takes precedence over
the plausible engineering next steps. A future independent simulator study may
target multimode spectral structure and acquisition-noise calibration, but it
requires a new protocol and new validation evidence.

No conclusion about CNNs, transformers, MOMENT, SSL quality, morphology, or
acquisition-OOD transfer follows from Week 3.

## Artifacts And Provenance

- protocol: `configs/yeast_ssl_followup_week3_v1.yaml`;
- corrected dataset: `yeast-passage-simulations@v2`;
- pfcalcul job: `20260716_yeast_followup_week3_v1`, return code `0`;
- evaluation: `artifacts/unsupervised-learning-flow-cytometry/runs/yeast-followup-week3-evaluation-v1/`;
- rendered decision: `artifacts/unsupervised-learning-flow-cytometry/reports/yeast-followup-week3-report-v2/`;
- metrics SHA256: `124e362e414ae358ee992b84759f5fa3fe488878ed3fbb12294b788621ded773`;
- dataset signal SHA256: `b9d7d8baab0fbafccf3bdd22331078770e32a4c40f96d9b8153777054ca94821`.

The publication figure is
`week3_simulator_comparison.png` in the rendered-decision artifact. The full
evaluation took `2661 s`; total remote orchestration took about one hour on the
shared CPU-limited Jupyter session.

## Disposition

Week 3 is complete. The month closes as a sequence of informative controlled
negatives: R3 did not pass the representation gate, and the targeted simulator
correction did not pass the common-support gate. Week 4 final-test evaluation
is not authorized, and `followup_test` remains sealed for a genuinely new
future study.
