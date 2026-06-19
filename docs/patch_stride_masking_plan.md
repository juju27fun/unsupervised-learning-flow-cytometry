# Patch, Stride, and Masking Audit

The first P3_SSL gate is visual, not a training score. Before accepting overlap,
generate a PDF that shows where tokens fall on the decimated signal.

The audit must display:

- decimated signal of length 2048
- patch spans for each candidate patch/stride pair
- sampled masked time blocks
- guard bands around masked blocks
- tokens hidden from the model
- optional event spans from labels, only for diagnostics

Default candidates:

| Patch | Stride | Tokens | Intent |
|---:|---:|---:|---|
| 4 | 4 | 512 | Primary no-overlap MOMENT-like run |
| 4 | 2 | 1023 | Fine alignment with heavy overlap risk |
| 8 | 8 | 256 | Coarser no-overlap baseline |
| 8 | 4 | 511 | Moderate overlap baseline |
| 16 | 8 | 255 | Wider context baseline |

Overlap is accepted only if it improves event-start alignment without making
masked regions directly recoverable from neighboring visible tokens. Guard-band
masking is mandatory for overlap experiments.

