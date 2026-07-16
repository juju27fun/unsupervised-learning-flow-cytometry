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
