# Guided Yeast SSL Presentation Plan

**Audience:** deep-learning practitioner who has not followed the project
**Format:** 16:9 ODP, 48 main slides plus 10 appendix slides
**Companion:** `YEAST_SSL_GUIDED_EXPLANATION_2026-07-17.pdf`

## Teaching Contract

The original deck was built as a compact scientific audit. This guided version
keeps its evidence and claim boundaries but exposes the reasoning that was
previously implicit in speaker notes.

Every experiment is presented using the same sequence:

1. **Problem:** what was missing or unreliable?
2. **Intuition:** what simple idea might help?
3. **Theory:** why might it work?
4. **Test:** what controlled comparison was run?
5. **Result:** what did the evidence show?
6. **Next:** which assumption changed as a consequence?

Terminology is introduced before it is used. Equations are interpreted term by
term. A figure is never shown without stating the question it answers and the
claim it cannot support.

## Narrative

| Block | Slides | Plain-language question | Main conclusion |
|---|---:|---|---|
| Bottom line and reading key | 1-4 | What is the project and how should I read it? | The model learned intended factors but did not become the best practical representation. |
| Why representation learning | 5-8 | What is a representation and what supervision was available? | We had unlabeled signals, simulator latents and scarce proxy labels, but no event-level morphology truth. |
| Experiment 1: reconstruction | 9-12 | Can missing waveform regions teach useful structure? | Reconstruction was learnable, but pretext improvement did not validate representation utility. |
| External baselines and latent sweeps | 13-16 | What information did existing encoders already contain? | Public backbones encoded particle structure; one-factor sweeps generated hypotheses, not real-world proof. |
| Signal quality and yeast simulation | 17-20 | Could a yeast-specific simulator make the latent physically meaningful? | SNR was a shortcut risk and the two-component generator remained a morphology hypothesis. |
| Domain-gap trap | 21-22 | Did visual simulated-real overlap mean alignment? | No. Domain AUC near 1 showed that origin remained almost perfectly recoverable. |
| Experiment 4: rejected hybrid | 23-26 | Would reconstruction + physics + invariance solve the bridge? | The objective asked the model to preserve and erase the same variables; more training was not justified. |
| Controlled rebuild | 27-35 | How do we make the comparison interpretable? | Freeze event quality, input meaning, retained factors, nuisances, architecture and causal contrasts. |
| Development and final evidence | 36-40 | Did physics supervision and real adaptation help enough? | Both helped internally, but A4 stayed narrow/domain-specific and lost to handcrafted features on the one-time test. |
| Mechanistic follow-up | 41-43 | Was the problem features, broad loss design or the simulator? | Broad loss additions did not fix utility; finite support improved the bridge but failed frozen support gates. |
| Final objective rescue | 44-47 | Why did A1 look flat, and can a well-controlled SSL objective work? | Corrected masking makes prediction learn, VICReg repairs collapse, but S1 remains 2.34x worse than interpolation. |
| Decision | 48 | What should happen next? | The bottleneck is useful information beyond signal priors; improve observability or simulation before another SSL search. |
| Technical appendix | 49-58 | Where are exact values, formulas, caveats and references? | Complete numerical and provenance support without interrupting the teaching narrative. |

## Added Guided Slides

Fourteen slides are added to the rigorous deck:

1. the repeated reasoning loop;
2. representation learning in plain language;
3. why reconstruction is useful but insufficient;
4. how PCA/t-SNE, sweeps and probes answer different questions;
5. domain gap in plain language;
6. three design rules learned from the rejected hybrid;
7. the final loss interpreted as an information contract;
8. how to read macro F1, factor recovery, effective rank and domain AUC;
9. Weeks 1-3 as one debugging tree;
10. the three decisions to discuss with the supervisor;
11. the final objective-rescue question and gate order;
12. corrected masking as a separation between predictability and collapse;
13. VICReg geometry repair followed by the controlled utility failure;
14. the S1 local-spectral result against zero, constant and interpolation.

## Objective-Rescue Teaching Contract

The new block must answer four different questions in order:

1. **Optimization:** can the production encoder and head reduce the intended
   target on fixed and full data?
2. **Non-triviality:** does the model beat zero and a train-derived constant
   while producing realistic output amplitude?
3. **Geometry:** does the embedding avoid near-rank-one concentration?
4. **Added value:** does learned prediction beat a strong deterministic signal
   prior and lead to useful frozen features?

The result is deliberately mixed rather than summarized as "SSL failed". S1
passes optimization, non-triviality and geometry, but fails added value:
development-validation MSE is `0.0597` for S1 and `0.0255` for interpolation.
The failure also holds in event, boundary and background regions. This closes
the frozen objective/head package without claiming that spectral SSL is
universally ineffective.

## Figure Policy

- Keep the original measured figures when they directly answer the question.
- Add diagrams only for concepts and reasoning transitions.
- Keep detailed full tables and dense multi-panel figures in the appendix.
- Mark every result as exploratory, development, confirmatory or follow-up.
- Never compare values from incompatible 512-sample particle and 4096-sample
  yeast protocols as one leaderboard.

## Scientific Boundaries

- `in_session_test` is exhausted and is not reused for selection.
- `followup_test` remains sealed.
- The objective rescue uses development evidence only; no S1 utility,
  additional representation seed or sealed evaluation was authorized.
- The source folders are acquisition-condition proxies, not trusted morphology
  labels.
- Domain AUC diagnoses recoverable origin; it does not identify the causal
  mismatch by itself.
- No new training is needed for this communication revision.
