# Yeast SSL Scope Decision

**Decision date:** 2026-07-15  
**Decision owner:** project owner

## Decision

The study proceeds with the available yeast acquisition because no additional
yeast material or independent acquisition can be obtained within the project
timeline. The completed v7 detector review is accepted as sufficient for this
restricted study. Independent reviewer reliability and acquisition-OOD
validation are waived as execution requirements, not silently treated as
satisfied evidence.

## Authorized Scope

- Use `yeast-event-candidates@v7` and `yeast-events-representation@v3`.
- Run the frozen A0-A4 matrix with representation seeds `42`, `43`, and `44`.
- Use `development_train` and `development_validation` for fitting and model
  selection.
- Open `in_session_test` once only after the method and probe protocol freeze.
- Report source-group proxy discrimination, label efficiency, retained-factor
  recovery, robustness, retrieval, collapse diagnostics, and simulation-real
  separability.
- Accept either a positive or a controlled negative result.

## Prohibited Claims

- No acquisition-OOD or cross-session generalization claim.
- No biological morphology classification claim from source-folder proxies.
- No independent-reviewer reliability claim.
- No post-v7 detector tuning, post-hoc architecture search, or test-set tuning.

## Consequence For Gates

Gate 1 may authorize full **in-session** development after the reviewed detector,
split-integrity, duplicate, and information-contract checks pass. The original
acquisition-OOD component of Gate 5 remains unavailable. The final decision is
therefore an in-session scientific comparison, and every final table and figure
must carry that scope.

## Execution Outcome

The authorized matrix completed on pfcalcul with representation seeds 42, 43,
and 44. The method and probe protocol were frozen, then `in_session_test` was
opened once in Slurm job `23487080`.

A4 improved over A3 by `+0.0235` macro F1, but remained below handcrafted
features by `-0.0805`, with a 95% hierarchical bootstrap interval entirely
below zero (`[-0.1040, -0.0484]`). A4 is not promoted. No OOD, morphology, or
independent-reviewer claim was introduced. The complete interpretation is in
[`YEAST_SSL_REBUILT_STUDY_REPORT.md`](YEAST_SSL_REBUILT_STUDY_REPORT.md).
