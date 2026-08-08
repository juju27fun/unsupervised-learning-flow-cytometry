# P3 self-supervised learning contract

- If `../workspace-repos.lock` exists, read `../AGENTS.md` first; Git roots do
  not inherit parent instructions.
- Rebuilt v1 and follow-up are frozen negative results: never relaunch legacy
  configs, retrain, open `followup_test`, or tune the exhausted final split;
  preserve runs.
- Put reusable logic in `p3_ssl/` and thin CLIs in `scripts/`. Import installed
  `p0`, `detseg`, and MOMENT; never inject or vendor source paths.
- Resolve registered inputs. Put manifested runs under
  `artifacts/unsupervised-learning-flow-cytometry/<run-id>/` and model caches in
  `.cache/huggingface/`; create no local outputs, vendors, or environments.
- Verify with `.venv/bin/python -m pytest -q unsupervised-learning-flow-cytometry/tests`
  without downloads or GPU work. Use root pfcalcul orchestration and runbook.
