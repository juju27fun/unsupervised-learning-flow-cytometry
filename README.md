# Unsupervised Learning for Flow Cytometry

Self-supervised particle-signal representation learning, pretrained backbone
comparisons, and physical latent-space validation. Reusable code is in
`p3_ssl/`; user-facing commands are in `scripts/`.

## Research status

The original P3 design is under methodological reconstruction. Do not launch a
large SSL or hybrid-physics run from the legacy configurations until the data,
split, physical-input, nuisance-variable, and baseline gates in
[`docs/YEAST_SSL_CRITIQUE_AND_REBUILD_PLAN.md`](docs/YEAST_SSL_CRITIQUE_AND_REBUILD_PLAN.md)
pass. Existing runs remain exploratory evidence and are not deleted.

## Workspace contract

- Environment: `../.venv`
- Registered inputs: `../datasets/`
- Run artifacts: `../artifacts/unsupervised-learning-flow-cytometry/`
- Hugging Face cache: `../.cache/huggingface/`

`p3_ssl.paths` defines these roots and supports `INTERNSHIP_WORKSPACE_ROOT`,
`INTERNSHIP_DATASETS_ROOT`, and `P3_SSL_ARTIFACT_ROOT` overrides. Dependencies,
including the pinned MOMENT package, are installed into the shared environment;
the former `vendor/` copies are not runtime dependencies.

## Development

From the workspace root:

```bash
.venv/bin/python -m pip install -e unsupervised-learning-flow-cytometry
.venv/bin/python -m pytest -q unsupervised-learning-flow-cytometry/tests
.venv/bin/python unsupervised-learning-flow-cytometry/scripts/build_ssl_manifest.py --help
.venv/bin/python unsupervised-learning-flow-cytometry/scripts/train_ssl_reconstruction.py --help
```

The current implementation reference uses a centered 4096-sample 1D input, but
its physical contract is being re-audited against the historical decimated-4096
path. See [`docs/p3_4096_pipeline.md`](docs/p3_4096_pipeline.md) for the existing
workflow and the rebuild plan above for experiment authorization. Every
generated manifest, checkpoint, metric table, and figure belongs in the project
artifact root.
