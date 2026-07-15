# Unsupervised Learning for Flow Cytometry

Self-supervised particle-signal representation learning, pretrained backbone
comparisons, and physical latent-space validation. Reusable code is in
`p3_ssl/`; user-facing commands are in `scripts/`.

## Research status

The rebuilt yeast study is complete under a documented single-acquisition
scope. A4 (physics-informed pretraining plus real SSL adaptation) improved its
A3 predecessor but did not outperform handcrafted features on the one-time
in-session test. The frozen decision is a controlled negative result; no
morphology or acquisition-OOD claim is made. Start with
[`docs/YEAST_SSL_REBUILT_STUDY_REPORT.md`](docs/YEAST_SSL_REBUILT_STUDY_REPORT.md),
then consult the
[`critique and rebuild plan`](docs/YEAST_SSL_CRITIQUE_AND_REBUILD_PLAN.md) and
[`execution log`](docs/YEAST_SSL_EXECUTION_LOG.md) for design history.

Do not relaunch legacy configurations or tune against the exhausted final split.
Existing runs remain evidence and are not deleted.

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

The frozen study contract is one centered, filtered, anti-aliased 4096-sample
channel at 1 MHz (4.096 ms), normalized from development-training statistics.
The historical decimated-4096 path is not interchangeable with it. See
[`docs/p3_4096_pipeline.md`](docs/p3_4096_pipeline.md) for the legacy workflow.
Every generated manifest, checkpoint, metric table, and figure belongs in the
project artifact root.
