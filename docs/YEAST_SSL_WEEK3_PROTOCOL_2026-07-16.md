# Yeast SSL Follow-Up: Week 3 Preregistered Protocol

**Frozen:** 2026-07-16, before corrected-validation generation or evaluation  
**Scope:** one simulator-family correction and conditional bridge audit  
**Final splits:** remain sealed

## Question

Does replacing the v1 diffuse Gaussian passage envelope with a train-calibrated
finite-support wave packet materially repair the measured simulation-real
common-support failure?

This is a simulator diagnostic for future studies. It cannot promote R3,
authorize representation retraining, or reopen the Week 2 decision.

## Evidence Selecting the Family

The frozen Week 1 analytic comparison retained only `14.6%` of train and
`16.2%` of validation examples. Its largest matched SMDs were spectral peak
count (`1.946` train, `2.342` validation) and duration (`1.170`, `1.081`).
Train-only distributions show median duration `0.362 ms` in reality versus
`1.278 ms` in v1, and median spectral peak count `13` versus `1`.

A bounded train-only pilot compared Tukey envelope alpha values 0.25, 0.50,
and 1.00 while leaving every other generator family unchanged.

### Pre-validation corrective addendum

The first exploratory pilot sampled target duration with the view RNG, so two
views of one latent could receive different duration factors. The production
implementation correctly samples duration once per latent. Its executable
preflight rejected the initial alpha 0.50 declaration before creating any
corrected dataset or accessing corrected validation outcomes.

With latent-level duration fixed, alpha 0.25 has the smallest train matched SMD
maximum (`0.365`) and retains `0.338`; alpha 0.50 reaches `0.411`/`0.340` and
alpha 1.00 `0.439`/`0.344`. Alpha 0.25 is therefore frozen. This addendum
corrects a paired-view implementation defect, not an outcome-dependent choice.

## Correction

1. Measure Hilbert-envelope support above 25% of peak using only real
   `followup_train` signals.
2. Store 101 quantile knots over the robust 5th-95th percentile interval.
3. Sample the target duration deterministically from those knots.
4. Generate a finite-support Tukey wave packet with alpha 0.25 and analytically
   compensate its 25%-height support.

Carrier, component, position, SNR, noise, drift, sensor-response, RMS,
preprocessing, split seeds, latent counts, and paired-view policy remain
identical to v1. No real trace is copied into simulation.

## Frozen Evaluation

Compare registered v1 and v2 simulations with the same real samples and seeds:

- 1,000 train and 500 validation examples per domain;
- primary nearest-neighbor caliper 1.50;
- descriptive sensitivity calipers 1.00 and 2.00;
- duration, dominant frequency, RMS, SNR estimate, spectral peak count, and
  event offset as matching observables;
- linear and random-forest domain probes on observables, signal summaries, and
  downsampled signals;
- capture-block and latent-group bootstrap with 1,000 repetitions.

Common support passes only if both train and validation retain at least 50% and
their maximum post-match SMD is at most 0.25. Conditional domain AUC is primary
only after that gate passes: at most 0.70 is a plausible future bridge,
0.70-0.85 is a residual gap, and above 0.85 remains a major mismatch.

If support fails, stop experimental work after documenting which residuals
remain. Thresholds, envelope alpha, nuisance priors, and probe choices may not
be changed after corrected validation is inspected. `followup_test`,
`in_session_test`, quality adaptation, CNN controls, and representation
training are forbidden in Week 3.

The executable source of truth is
[`yeast_ssl_followup_week3_v1.yaml`](../configs/yeast_ssl_followup_week3_v1.yaml).
