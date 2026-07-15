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
uv sync --frozen
uv run pytest -q
uv run python -m compileall -q src tests
uv run dreamer-train --help
uv run dreamer-inspect --help
```

For a one-update CPU smoke test:

```bash
uv run dreamer-train \
  general.dry_run=true general.device=cpu \
  train.max_train_steps=1 train.min_buffer_episodes=2 \
  train.batch_size=2 train.sequence_length=4 \
  train.replay_burn_in=1 train.eval_every=0
```

## Change Rules

- Reproduce runtime bugs through a supported CLI or a focused integration test.
- A trainer or collector subprocess failure must make the parent command fail.
- Preserve periodic, final, and best-checkpoint semantics as separate concepts.
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
