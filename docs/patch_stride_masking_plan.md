# Patch, Stride, and Masking Audit

> **Historical audit notice (updated 2026-07-15).** The completed rebuilt study
> froze non-overlapping 16-sample patches, 25% contiguous masks of
> 0.128-0.512 ms, and a 0.016 ms guard on the 1 MHz event-crop contract. The
> candidate grid below documents the earlier visual audit, not an open
> hyperparameter search. See
> [`YEAST_SSL_REBUILT_STUDY_REPORT.md`](YEAST_SSL_REBUILT_STUDY_REPORT.md).

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
