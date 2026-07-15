# Yeast SSL Rebuild: Execution Log

**Updated:** 2026-07-15

This is a concise evidence log for
[`YEAST_SSL_CRITIQUE_AND_REBUILD_PLAN.md`](YEAST_SSL_CRITIQUE_AND_REBUILD_PLAN.md).
It records gate decisions and failed intermediate designs; it is not a results
manuscript.

## Current Gate State

| Gate | State | Evidence | Decision |
|---|---|---|---|
| 0: provenance and ownership | Pass for required real and synthetic inputs | Registered IDs listed below; full checksum validation passed; generation code is in `particles2SNR-pipeline` | Source/model implementation may proceed |
| 1: event and split validity | Detector sub-gate pass; overall blocked | The independent v7 review gives candidate precision `48/48` and full-trace `TP=85`, `FP=1`, `FN=1`; reviewer reliability and a second acquisition are absent | Use v7 for single-acquisition development only; keep final biological and acquisition-OOD claims blocked |
| 2: input and information contract | Pass | P3 loaders, config, masking, loss, and simulator policy enforce the frozen contract | Smoke training authorized |
| 3: baseline readiness | Runtime complete, scientific matrix pending | All A0 methods completed on pfcalcul with the corrected split; the grouped multi-seed matrix is absent | Block A1-A4 promotion |
| 4: pretext validity | CUDA smoke evidence only | A1-A4 completed on pfcalcul and reconstruction beats interpolation/nearest controls; one epoch and one seed are insufficient | Block full scientific claim |
| 5: scientific promotion | Not started | No predeclared multi-seed evidence | OOD test remains unavailable/sealed |
| 6: optional methods | Not authorized | Gates 1-5 incomplete | No domain alignment or simulator inversion |

## Registered Inputs

| Dataset ID | Content | Key limitation |
|---|---|---|
| `yeast-hf-10-5-20260610@v1` | Immutable raw signals, microscopy images, and acquisition notes | One documented date/configuration |
| `yeast-source-index@v2` | Duplicate-family-safe, source-group-stratified 32-record capture-block metadata | Splits are in-session development splits only |
| `yeast-event-candidates@v5` | Same 11,099 candidates as v4, with corrected retained/rejected/missed review fields | Completed review failed Gate 1 and is now calibration evidence only |
| `yeast-event-candidates@v6` | First fresh queue for the review-calibrated detector | Superseded before annotation because 24 retained windows gave inadequate Wilson-interval power |
| `yeast-event-candidates@v7` | Frozen revised detector; 88 candidate windows and 65 full traces, excluding every v5/v6 review record | Event review passed; still from one acquisition and reliability review pending |
| `yeast-event-review-annotations@v1` | Immutable adjudicated v7 annotations, annotation audit, and Gate analysis | One reviewer; reliability subset remains pending |
| `yeast-events-representation@v2` | Frozen `10617 x 4096` real event tensor built from the candidate-identical v4 audit | Folder conditions are acquisition-level proxies |
| `yeast-events-representation@v3` | Reference `8721 x 4096` tensor built from the validated v7 detector | Single-acquisition development only; do not treat source conditions as labels |
| `yeast-passage-simulations@v1` | `14000 x 4096` paired-view identifiable passage simulations | Generic passage factors are not yeast morphology labels |

## Audit Findings

- The raw source contains 6,172 valid `float64[16384]` signals.
- There are 449 exact duplicate pairs; every pair has adjacent filename indices
  and remains within one source folder. Duplicate families are excluded before
  splitting, leaving 5,723 unique recordings.
- All source files share one documented acquisition date/configuration. Folder
  names describe acquisition conditions or concentration regimes, not
  independent sessions or event-level ground truth.
- The legacy event artifact assigned every yeast crop to `test`, used absolute
  download paths, and treated source folders as groups. It is historical only.
- A first block split allowed two duplicate pairs to cross split boundaries.
  That generated index was quarantined and rebuilt; the registered index has
  zero duplicate families and zero capture blocks crossing splits.
- The first registered split then proved unsuitable for source-group diagnostic
  probes: all `budding` and `shmoo` records fell in development train. The v2
  source index uses deterministic source-group-stratified 32-record blocks;
  every source group now appears in train, validation, and sealed in-session
  test without crossing a block or duplicate family.
- The v4 full-trace form could not distinguish false retained candidates from
  true rejected events. V5 leaves every candidate and review signal unchanged
  but records retained, rejected, and missed counts separately. Completed
  annotations must be stored as a new versioned dataset, never written into the
  registered v5 templates.
- A first candidate-audit build failed while serializing mixed event/background
  review rows. The partial dataset was quarantined; the schema now has a
  regression test.
- The legacy detector rejected 5,360 candidates because an 8,192-point centered
  crop crossed a source boundary. This incorrectly mixed physical quality with
  input geometry. Candidate quality is now crop-independent: v4 contains 9,491
  strict, 1,126 medium, and 482 width-rejected candidates.
- The completed v5 review contains `TP=95`, `FP=44`, and `FN=7` on full traces.
  Every one of the 24 reviewed width-rejected windows contains an event, so the
  old `1.6 ms` width rule was not a valid non-event criterion. In the medium
  tier, event and non-event median SNR proxies are nearly identical (`3.83`
  versus `3.92`); SNR alone does not identify all false positives.
- A count-only development sweep selected acceptance SNR `12`, boundary SNR
  `1.5`, cluster gap `0.128 ms`, maximum width `2.0 ms`, and at most five
  events. Its proxy precision/recall are both `0.931`, but this is not a
  localization metric and uses the same v5 traces. The preset is therefore
  frozen as `review-calibrated-v1`, not promoted as a validated detector.

## Frozen Input Contract

`yeast-event-8192to4096-bandpass-global-v1`:

1. clamp an 8,192-sample crop to the 16,384-sample source bounds without
   padding;
2. apply a fixed zero-phase 5-100 kHz bandpass;
3. downsample with polyphase anti-aliasing from 2 MHz to 1 MHz;
4. produce one 4,096-sample, 4.096 ms channel;
5. normalize with development-training global mean and standard deviation;
6. preserve event offset in metadata and treat position as a nuisance.

Absolute in-band amplitude is unresolved and preserved. It is not an
augmentation or synthetic target. Every downstream condition analysis must
include an RMS/amplitude shortcut baseline.

## Synthetic Redesign

The legacy hybrid simulator is not eligible because it predicted phase, event
position, and SNR while also treating them as nuisances; two-particle rows kept
only first-particle targets; pairwise distances depended on batch ranges; and
mean decimation violated the real-input contract.

The replacement simulator creates paired views sharing identifiable retained
factors (duration, Doppler, component count/separation, relative component
amplitude, and frequency separation) while independently resampling phase,
position, noise, output RMS, baseline drift, and sensor response. Morphology and
absolute physical amplitude are explicitly excluded targets.

## Immediate Next Check

The independent v7 event review is complete. The authoritative adjudicated
analysis is
`artifacts/particles2SNR-pipeline/audits/yeast-event-review-v7-analysis-20260715-v2/`:

- retained candidate precision is `48/48 = 1.000`, Wilson 95% CI
  `[0.926, 1.000]`;
- full-trace precision and recall are both `85/86 = 0.988`, Wilson 95% CI
  `[0.937, 0.998]`;
- per-group point precision and recall pass every frozen threshold;
- `6/40 = 0.150` reviewed rejected candidates contain an event, below the
  predeclared `0.25` maximum;
- one retained candidate has an acquisition artifact and one appears to merge
  two events, so the result is not evidence of a perfect detector;
- the only full-trace false negative is a true low-quality rejected candidate;
  its explicit reviewer note was adjudicated from `missed` to `true rejected`,
  with total `FN=1` unchanged.

Compared with v5 calibration, the revised detector removes the main low-SNR
false-positive mode without the catastrophic width rejection pattern. The
result validates extraction on this acquisition, not acquisition transfer.
The next human step is the frozen 20% v7 reliability review. The next data step
is a second independently acquired yeast session. No additional threshold
tuning is allowed on v7.

### Acquisition blocker audit and intake readiness

The workspace, its 2026-07-10 archive, `Downloads`, and registered dataset
catalog were searched for a second yeast acquisition. Only
`Downloads/Yeast_folder` was found; its 6,172 signal paths and counts match the
already imported `yeast-hf-10-5-20260610@v1` source. No forgotten independent
session is available locally.

The data owner now supports a future multi-acquisition v3 contract: record and
capture-block IDs are namespaced, exact traces crossing acquisitions are
rejected, every acquisition has a frozen `development` or
`sealed_ood_test` role, raw signals resolve by registered dataset ID, review
sampling and metrics are acquisition-stratified, and representation
normalization remains development-train-only. The end-to-end intake procedure
is documented in
[`YEAST_ACQUISITION_INTAKE.md`](https://github.com/juju27fun/particles2SNR-pipeline/blob/reorg/workspace-20260710/YEAST_ACQUISITION_INTAKE.md).
This closes the software-readiness risk, not Gate 1 itself.

The editable v7 queues can now be reviewed at `http://127.0.0.1:8765` through
`particles2SNR-pipeline/scripts/reports/serve_yeast_event_review.py`. The local
reviewer presents the raw trace and 7-80 kHz spectrogram, separates retained
candidate decisions from full-trace counts, validates count identities, saves
atomically, and keeps an append-only checksum audit. Desktop and mobile renders
were checked against the real v5 payload without submitting an annotation.
The protocol now also requires a stratified 20% independent double review; this
reliability evidence is still pending.

The old v5 reliability subset remains provenance for the calibration labels,
not final detector validation. The frozen v7 reliability artifact is
`artifacts/particles2SNR-pipeline/audits/yeast-event-review-v7-reliability-work/`;
it contains 18 candidate windows and 13 full traces (at least 20% of each queue)
and covers every available review stratum. It must be completed by an
independent reviewer, or as a blinded delayed repeat explicitly reported as
intra-rater evidence.

## Development Smoke Evidence

All values below are runtime/development diagnostics from
`yeast-pfcalcul-smoke-v3` on an A30 MIG GPU, not publishable estimates. Every
artifact validates, records P3 revision `9392383` and data-owner revision
`2b0370e`, and reports no use of a sealed split.

- A1-A4 satisfy the frozen v2 input contract and never open `in_session_test`.
- A1 real masked MSE is `0.890` versus interpolation `1.181`; A2/A3 simulation
  MSE is about `1.14` versus interpolation `1.965`; A4 reaches `0.871` on real
  and `1.089` on simulation.
- A3 component accuracy (`0.75`) equals its majority baseline and every
  continuous factor is worse than the constant normalized prior after one
  epoch. A4 improves two of five factors, but remains below the prior on the
  other three.
- Embedding-health instrumentation is active. The bounded 16-example runs have
  effective ranks around 2-3 and cannot decide collapse; GPU smoke must evaluate
  a larger validation sample against the random encoder control.
- In the balanced A0 smoke, handcrafted time/frequency features reach `0.351`
  macro F1 at 10% proxy labels. MOMENT reaches `0.298`; RMS, raw, random, and
  PatchTST range from `0.190` to `0.230`. Conv1D at one epoch collapses to one
  class and is runtime evidence only.
- Frozen A1-A4 checkpoint probes reach `0.188`, `0.214`, `0.222`, and `0.221`
  macro F1 at 10% proxy labels, all below the handcrafted baseline.
  Simulation-versus-real remains readily predictable for every checkpoint
  (`ROC AUC 0.921-0.927`); one A4 adaptation epoch does not close the domain gap.
- On 256 bounded real embeddings, random, A3, and A4 effective ranks are about
  `6.83`, `5.93`, and `3.72`, respectively. This is an early warning that A4 may
  concentrate the representation; full validation statistics and multiple
  seeds are required before declaring collapse.

## Remote Runtime Record

- Slurm jobs `23482036` and `23482037` failed before training because MOMENT
  attempted to resolve missing FLAN configuration and then redundant FLAN
  weights. No metrics from these attempts are scientific evidence.
- Job `23482038` completed the gate audit, A0, A1-A4, and checkpoint diagnostics
  in about two minutes with CUDA enabled. The loader now initializes the MOMENT
  architecture from its configuration and strictly loads the official MOMENT
  checkpoint without downloading a second 3 GB backbone.
- Remote and local smoke values agree to the expected deterministic tolerance.
  This closes the execution-infrastructure question, not the representation
  learning question.

## Expanded Development Evaluation

Slurm job `23482039` reused the `yeast-pfcalcul-smoke-v3` checkpoints without
training and completed the expanded evaluation as `yeast-pfcalcul-eval-v1`.
All artifacts validate and report no sealed split. Source metrics record P3
revision `db14367`; the rendered report records `5ed6d89`; both record data
owner revision `f802c2a`. The publication-shaped
development figures and tables are under
`artifacts/unsupervised-learning-flow-cytometry/reports/yeast-pfcalcul-eval-v1/`.

These remain bounded one-seed smoke diagnostics on source-condition proxy
labels:

- At 10% proxy labels, handcrafted features reach `0.351` macro F1, MOMENT
  `0.298`, and A1-A4 reach `0.188`, `0.214`, `0.222`, and `0.221`.
- Capture-block paired bootstrap differences are A4-handcrafted `-0.114`
  (`95% [-0.263, 0.021]`), A4-MOMENT `-0.057`
  (`[-0.165, 0.051]`), A4-A3 `-0.004` (`[-0.050, 0.053]`), and A3-A2 `0.014`
  (`[-0.048, 0.074]`). None supports promotion.
- A1-A4 calibration is poor at the low-label endpoint (`ECE 0.477-0.527`,
  multiclass Brier `1.129-1.193`). Conv1D's low ECE is not evidence of quality:
  it predicts one class with low confidence and reaches only `0.10` macro F1.
- Cross-recording top-1 proxy-label retrieval is `0.297-0.312` for A1-A4,
  below the untrained random encoder (`0.320`). Quality-stratum purity remains
  high (`0.69-0.72`), so retrieval does not demonstrate biological structure.
- Simulation-real domain ROC AUC remains `0.921-0.927`; A4 does not materially
  align the domains.
- Every cell linearly recovers simulated Doppler and duration, including A1,
  which never sees simulations. A3 does not improve these factors over A2.
  Component separation, frequency separation, and relative component amplitude
  remain at or below a constant prior; the physics objective has not established
  multi-component physical organization in the one-epoch smoke.
- Bounded noise and measured-IQR offsets preserve `0.92-0.99` of predictions,
  but an 8-sample (`8 us`) shift preserves only `0.50-0.57`. Very small cosine
  embedding distances therefore do not imply a stable downstream decision.

The decision remains unchanged: do not add domain alignment, new architectures,
or simulator inversion. Complete Gate 1, then run the frozen multi-seed matrix.
If the full matrix reproduces these patterns, report a controlled negative
result rather than searching post hoc for a positive A4 configuration.
