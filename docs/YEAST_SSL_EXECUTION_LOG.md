# Yeast SSL Rebuild: Execution Log

**Updated:** 2026-07-14

This is a concise evidence log for
[`YEAST_SSL_CRITIQUE_AND_REBUILD_PLAN.md`](YEAST_SSL_CRITIQUE_AND_REBUILD_PLAN.md).
It records gate decisions and failed intermediate designs; it is not a results
manuscript.

## Current Gate State

| Gate | State | Evidence | Decision |
|---|---|---|---|
| 0: provenance and ownership | Pass for required real and synthetic inputs | Registered IDs listed below; full checksum validation passed; generation code is in `particles2SNR-pipeline` | Source/model implementation may proceed |
| 1: event and split validity | Fail | Manual candidate and full-trace review are pending; only one independent acquisition is documented | No final biological or acquisition-OOD training claim |
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
| `yeast-event-candidates@v5` | Same 11,099 candidates as v4, with corrected retained/rejected/missed review fields | Manual annotation pending; registered templates are immutable |
| `yeast-events-representation@v2` | Frozen `10617 x 4096` real event tensor built from the candidate-identical v4 audit | Folder conditions are acquisition-level proxies |
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

Complete the 73 candidate-window and 73 full-trace v5 annotations, run the
predeclared review analyzer, and register the completed annotations separately.
Even a passing detector audit will not unlock the primary OOD endpoint: a
second independent acquisition is required. Full training remains blocked
until both requirements are satisfied.

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
