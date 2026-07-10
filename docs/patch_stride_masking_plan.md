# Patch, Stride, and Masking Audit

The first P3_SSL gate is visual, not a training score. Before accepting overlap,
generate a PDF that shows where tokens fall on the decimated signal.

The audit must display:

- decimated signal of length 4096, the primary full-window representation
- patch spans for each candidate patch/stride pair
- sampled masked time blocks
- guard bands around masked blocks
- tokens hidden from the model
- optional event spans from labels, only for diagnostics

Default candidates:

| Patch | Stride | Tokens | Intent |
|---:|---:|---:|---|
| 4 | 4 | 1024 | Primary no-overlap MOMENT-like run |
| 4 | 2 | 2047 | Fine alignment with heavy overlap risk |
| 8 | 8 | 512 | Coarser no-overlap baseline |
| 8 | 4 | 1023 | Moderate overlap baseline |
| 16 | 8 | 511 | Wider context baseline |

Overlap is accepted only if it improves event-start alignment without making
masked regions directly recoverable from neighboring visible tokens. Guard-band
masking is mandatory for overlap experiments.
