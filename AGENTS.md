# SleepyDreamyV3 Agent Guide

## Purpose

SleepyDreamyV3 is a PyTorch DreamerV3 research implementation. Treat it as an
ML reliability project: preserve reproducibility, make failures visible, and do
not present an experiment as successful without deterministic evaluation.

## Repository Map

- `src/dreamer/models/`: encoder, RSSM world model, losses, imagination, and
  actor/critic initialization.
- `src/dreamer/runtime/`: environments, collection processes, and replay.
- `src/dreamer/trainer/`: training orchestration, checkpoints, and logging.
- `src/dreamer/conf/`: Hydra configuration.
- `tests/`: fast tests intended for local runs and CI.
- `scripts/`: diagnostic research probes.

Generated experiment directories, checkpoints, MLflow stores, inspection
outputs, and `.lavish/` review artifacts are not source code and should remain
untracked.

## Supported Commands

```bash
uv sync --frozen --extra cpu
uv run --extra cpu pytest -q
uv run --extra cpu python -m compileall -q src tests scripts
uv run --extra cpu pyright src/dreamer/main.py src/dreamer/run_manifest.py \
  src/dreamer/trainer/checkpoints.py src/dreamer/runtime/replay_buffer.py
uv run --extra cpu dreamer-train --help
uv run --extra cpu dreamer-inspect --help
uv run --extra cpu python scripts/index_runs.py
```

Use `UV_PROJECT_ENVIRONMENT=.venv-rocm uv sync --extra rocm` for the separate
ROCm 6.4 training environment. Never install CPU and ROCm extras together.

For a one-update CPU smoke test:

```bash
uv run --extra cpu dreamer-train \
  general.dry_run=true general.device=cpu \
  train.max_train_steps=1 train.min_buffer_episodes=2 \
  train.batch_size=2 train.sequence_length=4 \
  train.replay_burn_in=1 train.eval_every=0
```

## Change Rules

- Reproduce runtime bugs through a supported CLI or a focused integration test.
- A trainer or collector subprocess failure must make the parent command fail.
- Treat episode delivery as fan-in, but model updates as fan-out: every collector
  must have an independent bounded weight mailbox.
- On normal completion, keep the trainer process alive until the parent has
  stopped and joined every collector.
- Preserve periodic, final, and best-checkpoint semantics as separate concepts.
- Every non-dry run must retain `run_manifest.json`; checkpoint artifacts must
  carry its run ID and preserve the configured evaluation metric on resume.
- Hydra YAML is the authored training configuration. Reject invalid runtime
  projections before starting MLflow, collectors, or the trainer.
- Run indexing is non-destructive: do not load pickle checkpoint contents or
  delete raw evidence while generating `reports/runs.csv`.
- Keep long training out of hosted CI; use fast deterministic environments for
  integration coverage.
- Do not start broad sweeps without a hypothesis, primary metric, fixed seed
  set, sample budget, and stop rule.
- Record research conclusions in a concise report or manifest rather than
  relying only on chronological notes.

## Validation Order

Run focused tests first, then the full fast suite, compile/type checks, and an
end-to-end CPU smoke test when process or training code changes. Preserve logs
or summaries that demonstrate the original user-visible behavior.
