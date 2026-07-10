# P3 SSL Codex Instructions

## PF Calcul

- For remote pfcalcul GPU work, read `PF_CALCUL_JUPYTER_FIRST_AGENT_PROMPT.md`
  from this directory before launching anything.
- For interactive/fast pfcalcul GPU execution, use the Jupyter RTX PRO runner.
- Do not launch GPU training directly over SSH. SSH is for enqueueing runner
  jobs, monitoring, collecting results, and explicit Slurm submissions.
- Use Slurm only when the user explicitly asks for batch execution or when
  reliability across Jupyter/session interruptions matters.
- Do not delete datasets/checkpoints or cancel running long jobs without
  explicit confirmation.
