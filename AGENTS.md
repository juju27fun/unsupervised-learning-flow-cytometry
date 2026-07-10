# Unsupervised Learning for Flow Cytometry agent context

- Treat this as an independent nested repository and obey the parent workspace
  `AGENTS.md`.
- Put reusable logic in `p3_ssl/`; scripts must be thin CLIs and must not inject
  source or vendored directories into `sys.path`.
- Resolve datasets through the workspace registry. Write all run material to
  `artifacts/unsupervised-learning-flow-cytometry/<run-id>/`.
- Keep model caches in `.cache/huggingface`; do not recreate `outputs/`,
  `vendor/python*`, or a project-local environment.
- Import `p0`, `detseg`, and MOMENT from the shared installed environment.
- Verify with `.venv/bin/python -m pytest -q unsupervised-learning-flow-cytometry/tests`
  and avoid downloading models or launching GPU training for a source-only check.
- Use the workspace pfcalcul orchestration and runbook for remote execution.
