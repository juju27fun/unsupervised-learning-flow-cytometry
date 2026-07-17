# Yeast SSL Supervisor Presentation Plan

**Status:** superseded for delivery by
[`YEAST_SSL_GUIDED_PRESENTATION_PLAN_2026-07-17.md`](YEAST_SSL_GUIDED_PRESENTATION_PLAN_2026-07-17.md).
This file preserves the original compact 54-slide design baseline; it does not
include the final objective-rescue result.

**Presentation date:** 2026-07-17
**Audience:** internship supervisor and technically informed colleague
**Language:** English
**Format:** 16:9 ODP, approximately 25-35 minutes plus discussion
**Target:** 34 main slides plus 10 appendix slides

## Narrative Contract

The deck follows the actual evidence chain rather than presenting the latest
model as an isolated success:

> exploratory masked reconstruction -> physical diagnostics -> failed hybrid
> design -> controlled rebuild -> negative confirmation -> mechanistic
> follow-up -> next scientific decision

Every quantitative slide carries one evidence label:

- **Exploratory:** legacy inputs, incomplete controls, visualization, or
  post-hoc diagnostics;
- **Development:** frozen development split used for model comparison;
- **Confirmatory:** the one-time `in_session_test`, now exhausted;
- **Prospective follow-up:** the new train/validation protocol with
  `followup_test` still sealed.

The deck must never call source-condition proxies morphology labels, describe
multi-view signals as multimodal, claim acquisition OOD performance, or compare
legacy 512-sample particle results numerically with the final 4096-sample yeast
endpoint.

## Main Slides

| # | Slide and role | Evidence / visual | Metrics and conclusion | Transition |
|---:|---|---|---|---|
| 1 | **Learning physically useful yeast-signal representations** | LAAS/CNRS title layout | Research question: can simulation and scarce unlabeled real measurements produce a useful low-label representation? | Start from the answer, then explain how it was reached. |
| 2 | **The study produced a useful negative result** | Three-column `learned / failed / unresolved` summary | Physics recovery improved; A4 lost to handcrafted on final test; sim-real support remains invalid. | These statements come from evidence of different maturity. |
| 3 | **The evidence chain changed the scientific question** | Full June-July timeline with evidence labels | Original focus on latent appearance evolved into controlled utility and bridge validation. | Reconstruct the early reasoning before showing the final protocol. |
| 4 | **Exploration: what can unlabeled signals teach us?** | Dark section transition | No metric. | Why representation learning appeared attractive. |
| 5 | **Why SSL, simulation and physics supervision are complementary** | Diagram: unlabeled real, simulator latents, scarce labels -> representation -> tasks | Explain SSL versus physics supervision versus linear-probe supervision. | The first implementation began with the simplest SSL objective. |
| 6 | **The biological target was not yet observable in the labels** | Signal/acquisition diagram plus claim-boundary panel | Source folders are acquisition-condition proxies, not event-level morphology labels. | The first experiments therefore tested signal structure, not yeast morphology. |
| 7 | **The first model learned by reconstructing masked waveform regions** | Legacy patch-transformer diagram | Historical input and masking marked as legacy; labels absent from the loss. | Reconstruction required choices about what signal properties to preserve. |
| 8 | **The original loss mixed waveform, derivative and energy fidelity** | New loss-evolution graphic, generation 1 | `L = MSE_signal + 0.2 Huber_derivative + 0.05 Huber_energy`. Define each term pedagogically. | A lower pretext loss is not automatically a useful latent space. |
| 9 | **Reconstruction improved; usefulness remained unproven** | New training-history plot plus a reconstruction example | Train loss 13.10 -> 5.73; validation masked MSE 2.325; test 2.399. No recorded trivial-baseline result and old embedding evaluation used `split=all`. | Stronger external baselines and controlled synthetic probes were needed. |
| 10 | **Pretrained backbones already encoded useful particle structure** | Legacy particle label-efficiency curve | At 10% particle labels: Conv1D 0.799, MOMENT 0.708, PatchTST 0.619 macro F1. Mark 512-sample particle protocol as non-comparable historical motivation. | Classification scores still did not reveal which physical variables shaped the latent. |
| 11 | **One-factor sweeps exposed what each latent reacts to** | Selected physical-sweep panels for Doppler, position, width and SNR | Qualitative organization and reduction trustworthiness only; no claim of causal physical identification. | Signal quality emerged as a possible shortcut. |
| 12 | **SNR changes neighborhood structure in every legacy encoder** | Simplified legacy yeast SNR-impact curves | Present as exploratory quality sensitivity, not a calibrated physical SNR conclusion. | A yeast-specific simulator was then built to study richer signatures. |
| 13 | **A two-component generator encoded the budding hypothesis** | Formula and example signals with relative-factor labels | Two components vary separation, amplitude ratio, Doppler offset, phase, width ratio and noise. Hypothesis, not morphology validation. | Visually, generated and real budding signals appeared encouraging. |
| 14 | **Visual overlap falsely suggested domain alignment** | Historical PCA/t-SNE beside a quantitative domain-AUC callout | Unfiltered median domain AUC 0.990; filtered 0.9995 despite visible overlap. This is the central cautionary historical result. | A physics-guided objective was proposed to force a better latent organization. |
| 15 | **The first hybrid objective combined three kinds of constraints** | New loss-evolution graphic, generation 2 | Composite reconstruction + `0.10` physical contrastive + `0.05` augmentation-invariance cosine loss; 70/30 synthetic/real pretraining and real adaptation. | The ingredients were reasonable individually but incoherent together. |
| 16 | **The legacy hybrid had a contradictory information policy** | Retained-versus-nuisance contradiction matrix | Amplitude, phase, position and SNR were supervised while normalization/augmentation removed or perturbed them; two-particle targets and input semantics were incomplete. No eligible full-run result. | The study had to be rebuilt before more training. |
| 17 | **Controlled rebuild: from attractive figures to testable claims** | Dark section transition | No metric. | The rebuild froze data meaning, causal comparisons and stopping rules. |
| 18 | **The bottleneck moved from architecture to evidence quality** | Audit-to-rebuild flow: detector review, duplicate-safe split, input contract, gates | Detector v7: retained precision 48/48; full-trace precision and recall 85/86; single acquisition and single reviewer remain limitations. | With events bounded, one model input could be defined physically. |
| 19 | **Every method now receives the same 4.096 ms signal** | Current preprocessing pipeline | Raw 2 MHz -> reviewed event -> 8192 crop -> 5-100 kHz filter -> anti-aliased 1 MHz -> global train normalization -> 4096 samples. | Simulation must produce the same observable while separating retained factors from nuisances. |
| 20 | **Paired simulation views separate physics from acquisition nuisance** | New multi-view simulator diagram | Shared: duration, Doppler, components/separations, relative amplitude. Resampled: phase, position, noise, RMS, drift, sensor response. | The encoder and losses implement this information policy. |
| 21 | **One compact Transformer supports all controlled cells** | Current architecture diagram | 349,367 parameters; 256 patches of 16; 3 layers, 4 heads; 96-D pooled embedding. | The controlled cells change supervision, not architecture. |
| 22 | **Four objective generations clarify what was actually tested** | Full loss-evolution figure | Legacy reconstruction -> rejected hybrid -> retained-factor A1-A4 -> Week-2 spectral/VICReg ablation. Explicit SSL/physics-supervised labels. | A0-A4 isolate the causal contribution of each retained ingredient. |
| 23 | **A0-A4 ask three causal questions under one architecture** | Matrix of data stage and losses | A3-A2: physical supervision; A4-A3: real adaptation; A4-best baseline: practical utility. Seeds 42/43/44. | These questions require downstream and mechanism metrics. |
| 24 | **Evaluation separates utility, physics, geometry and domain** | Four-axis evaluation diagram | Primary: frozen linear-probe macro F1 at 10% labels. Secondary: label-efficiency AUC, physical MSE reduction, effective rank/cosine, domain ROC AUC, robustness and retrieval. | Begin with practical utility on development. |
| 25 | **A4 improves—but handcrafted features stay competitive** | Development label-efficiency curve | At 10%: handcrafted 0.356, MOMENT 0.329, A3 0.335, A4 0.351. No practical winner. | Paired contrasts show which internal steps nevertheless had real effects. |
| 26 | **The controlled mechanisms improve as intended** | Paired-difference and physical-fidelity panels | A3-A2 +0.0457 [0.0230, 0.0584]; A4-A3 +0.0160 [0.0008, 0.0281], below 0.03 practical threshold. A4 improves all five retained-factor means and component balanced accuracy to about 0.90. | Better factor recovery did not guarantee a healthy or transferable representation. |
| 27 | **The latent stays concentrated and domain-specific** | Embedding-health and domain-versus-physics panels | A1/A2 effective rank 1.2-1.7/96; A3/A4 2.7-4.2/96. Domain AUC reaches 0.9990-0.9998 for A3/A4. | The final test decides whether the complete system is still practically useful. |
| 28 | **The one-time test confirms the handcrafted baseline is better** | Final macro-F1 and paired difference | Handcrafted 0.433, A3 0.329, A4 0.353; A4-handcrafted -0.0805 [-0.1040, -0.0484]. Controlled negative, no promotion. | The follow-up therefore studies failure mechanisms without reopening the test. |
| 29 | **Follow-up: diagnose the bridge, not another architecture** | Dark section transition | `in_session_test` exhausted; `followup_test` sealed. | Week 1 determines whether another objective or a simulator correction is justified. |
| 30 | **Weeks 1-3 progressively localize the remaining failure** | Three-column Week 1/2/3 evidence chain | Week 1: handcrafted family ceiling and support failure. Week 2: R0-R3 no robust primary gain and 2/3 convergence. Week 3: finite-support envelope improves retention/SMD but misses frozen gates. | Inspect each gate before deciding. |
| 31 | **Week 1: handcrafted frequency cues remain hard to replace** | Feature-family and domain-bridge figures | All handcrafted features reach 0.381 macro F1; analytic matching fails common support. | Test whether objective design can repair representation quality. |
| 32 | **Week 2: spectral reconstruction and VICReg do not help** | Full R0-R3 comparison | Macro F1 remains 0.314-0.335; R3 fails primary, rank and all-seed gates. | Correct one simulator mechanism instead of another loss. |
| 33 | **Week 3: finite support improves; bridge gates still fail** | Full simulator-comparison figure | Validation retention 0.162 to 0.374; max SMD 2.342 to 0.641; frozen gates still fail. | Convert the sequence into a decision. |
| 34 | **Next: observability and simulation, not model scale** | `supported / rejected / open / act next` table | Supported: physics heads and adaptation have mechanism effects. Rejected: practical promotion, domain invariance, morphology/OOD claims, more same-protocol architecture search. Open: new biological labels/acquisition and a better noise/SNR/spectral simulator. | End with questions for the supervisor. |

## Appendix Slides

| # | Appendix content | Purpose |
|---:|---|---|
| A1 | Evidence-maturity and compatibility matrix | Prevent accidental comparison across 512/4096, particles/yeast and exploratory/confirmatory protocols. |
| A2 | Complete legacy particle baseline table | Preserve historical MOMENT/PatchTST/Conv1D evidence without crowding the main narrative. |
| A3 | Complete one-factor latent sweeps | Show all six variables and all available encoders. |
| A4 | SNR analyses and detector-quality distinction | Separate physical SNR dB, yeast `snr_proxy`, and detector false-positive observations. |
| A5 | Detector review and dataset counts | Provide Wilson intervals, review semantics and single-reviewer limitation. |
| A6 | Exact old and new objective formulas | Support technical discussion of reconstruction, contrastive, factor, consistency, spectral and VICReg terms. |
| A7 | Full A0 baseline and paired-development tables | Preserve all systems and uncertainty. |
| A8 | Per-proxy recall, robustness and retrieval | Show the shmoo failure, perturbation worst case and shortcut warning. |
| A9 | Week 2 and Week 3 gates | Give seed convergence, rank, retention, SMD and sensitivity details. |
| A10 | Literature, artifact, split and claim provenance | Cite primary papers, map every figure/number to a local artifact and record sealed-test status. |

## Figure Production

New figures will be written under
`artifacts/unsupervised-learning-flow-cytometry/reports/yeast-ssl-supervisor-presentation-20260717/`
with a valid `run.json`:

1. `study_timeline.png`: chronology and evidence maturity;
2. `legacy_reconstruction_history.png`: 60-epoch losses and held-out metrics;
3. `loss_evolution.png`: four objective generations and supervision types;
4. `legacy_overlap_caution.png`: visual overlap paired with domain metrics;
5. `retained_nuisance_contradictions.png`: legacy hybrid information conflict;
6. `followup_evidence_chain.png`: Week 1-3 gates and decisions;
7. `scientific_balance_sheet.png`: supported, rejected, open and next actions.

Existing publication figures will be embedded from their manifested report
directories. They may be cropped for presentation but their numerical content
will not be altered.

## Missing-Analysis Decision

No new GPU training is scientifically justified for this presentation. The
available evidence already answers the claims used in the narrative. The only
new computation is deterministic secondary reporting from existing JSON/CSV
artifacts. In particular:

- do not access `in_session_test` again;
- do not open `followup_test`;
- do not rerun the rejected legacy hybrid pipeline;
- do not add a post-hoc architecture or loss cell;
- do not treat PCA/t-SNE as primary evidence.

## Presentation Design

The deck duplicates the official layouts from `/home/intern/Dropbox/Intern_JLB.odp`:

- 16:9 page geometry;
- LAAS/CNRS and University of Toulouse footer;
- navy titles, cyan/red title rule and white background;
- navy transition slides;
- restrained evidence colors: cyan for exploratory, amber for development,
  pink for confirmatory and green for prospective follow-up;
- one scientific message per slide, stated in the title;
- concise notes of approximately 40-90 words per main slide.

## Acceptance Checks

- ODP opens and exports through LibreOffice without repair warnings.
- All 44 slides render at 16:9 with no overlap or clipped text.
- Every numerical claim matches a source JSON/CSV/report.
- Every slide has an evidence level or is clearly conceptual.
- All images are embedded; no absolute local path is needed at presentation time.
- Speaker notes describe interpretation and limitations without duplicating the
  visible slide.
- A skeptical review confirms that negative evidence is not visually minimized
  and that the final decision follows from the displayed results.
