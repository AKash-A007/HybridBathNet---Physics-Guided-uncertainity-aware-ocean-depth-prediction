# Project working instructions

- Read [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) before edits, experiment launches, or claims about project status. Use its artifact links to verify current results; the root results JSON is historical.
- After every modification to code, configuration, data, experiment state, results, or plans, update PROJECT_CONTEXT.md and the current-status section of README.md in the same work session. Record what changed, why, checks performed, actual results, active/stopped/completed jobs, and next steps. Link new artifacts and distinguish hypotheses from findings.
- Preserve historical checkpoints/results and completed studies. Put new experiments in distinct named directories. Never silently reuse stale features or overwrite a comparator.
- Do not automatically start training merely from this handoff. Check the latest user instruction and PROJECT_CONTEXT.md for the currently authorized experiment scope; subsequent explicit training requests supersede earlier stop/review-only instructions. Avoid duplicate jobs.
- Use the existing venv and verify CUDA. Do not train tabular methods unless the user explicitly requests them.
- Report validation and test metrics separately. Six versus sixteen refers to verified input-channel counts; do not assert network layer counts without checking source.
- Keep spatial/temporal grouping intact, fit transformations on training data only, and never optimize test splits or remove difficult held-out examples to improve reported scores.
- Keep credentials out of code, logs, commits, and documentation.
