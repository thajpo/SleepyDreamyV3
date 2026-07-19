# Current CartPole Instability Diagnosis

Date: 2026-07-18

Source commit: `12e748b6eecd1e4db64b210e03ced4cb537ca895`

Run evidence: `runs/cartpole_current_10k_12e748b_r2/`

## Decision

The current warmup/reinforce configuration does not reliably learn CartPole.
The planned 10,000-update screen was stopped by user decision after every seed
had produced repeated deterministic evaluations near the minimum episode
length. This is an intentionally truncated screen, not a completed 10,000-step
benchmark.

The first clearly broken boundary is imagined action-value construction. The
learned latent and decoded one-step transition contain useful control
information, but the continuation rollout creates a biased action preference
and the critic bootstrap usually amplifies it in the wrong direction. The
trained actor then turns that unreliable target into a constant or random-like
policy.

## Three-seed result

All runs used the same code and configuration. Only the deterministic seed
differed.

| Seed | Last update observed | Deterministic evaluation rewards | Outcome |
|---:|---:|---|---|
| 0 | 5,300 | 9.30, 9.30, 9.25, 9.30, 9.30 | actor entropy collapsed |
| 1 | 5,500 | 18.25, 13.40, 18.65, 9.30, 9.30 | transient random-like behavior, then constant actor |
| 2 | 8,500 | 9.30, 11.40, 9.25, 9.25, 9.35, 9.30, 9.40, 9.40 | actor remained high-entropy/random-like |

The runs were operationally healthy. Collector logs show every published
weight version delivered and loaded. Per-run USS stabilized near 3.2 GiB,
available host RAM remained roughly 15-17 GiB, and no child process failed.
This rules out the repaired collector broadcast path and resource exhaustion as
causes of this learning result.

## Within-seed collapse probe

Seed 1 provides a controlled comparison between its best checkpoint at step
3,000 and its collapsed periodic checkpoint at step 5,000. Both probes used the
same 512 simulator states and probe seed.

| Measurement | Step 3,000 | Step 5,000 |
|---|---:|---:|
| Deterministic evaluation reward | 18.65 | 9.30 |
| Actor entropy | 0.693 | 0.451 |
| Mean actor probability of action 1 | 0.499 | 0.822 |
| Actor preference histogram | action 0: 435; action 1: 77 | action 0: 0; action 1: 512 |
| Actor vs real-rollout action accuracy | 0.490 | 0.510 |
| Model/critic Q vs real-rollout accuracy | 0.442 | 0.442 |
| Model/critic Q vs real-return correlation | -0.353 | -0.009 |
| Mean one-step state MSE | 0.109 | 0.355 |
| One-step action-effect sign accuracy | 1.0 | 1.0 |

The step-3,000 score was not evidence of a competent policy. Its action
probabilities were almost exactly uniform, so its modest return was
random-like. By step 5,000 the actor preferred action 1 on every probed state,
while its agreement with real action values remained at chance.

One-step state prediction became less accurate on the independent probe, but
the model retained the correct direction of both action effects. The critic's
multi-step action ranking remained uninformative at both checkpoints.

## Frozen-latent supervision probe

A fresh actor was trained against the same deterministic heuristic labels on
4,096 posterior latent states from each checkpoint. Actor initialization,
dataset seed, training budget, and evaluation episodes were fixed.

| Frozen world model | Train accuracy | Mean deterministic return |
|---|---:|---:|
| Step 3,000 | 0.611 | 22.85 |
| Step 5,000 | 0.666 | 54.65 |

The later latent was more useful to a clean supervised control target even
while the repository's trained actor had collapsed. Representation quality is
still imperfect and world-model rollout fidelity remains a secondary concern,
but observation encoding is not the first failed boundary.

## Mechanism hypothesis

`actor_warmup_steps` currently suppresses both actor and critic updates:

```python
in_actor_warmup = self.train_step < self.config.actor_warmup_steps
skip_ac_batch = in_actor_warmup or self.should_skip_ac_update()
```

At step 3,000, actor and critic training therefore start together. Across all
three seeds, the first logged imagined critic distribution had mean `23.505`
and standard deviation `0.00000165`; actor entropy was approximately
`log(2) = 0.693`. Imagined reward variance was also nearly zero.

The reinforce loss centers and scales advantages by their batch standard
deviation. When the critic and imagined returns contain almost no trustworthy
action-dependent variation, this can promote tiny model/value differences to
unit-scale policy gradients. The entropy coefficient is only `0.001`. Seeds 0
and 1 then developed arbitrary action commitments; seed 2 retained high entropy
without learning control.

This is the leading causal hypothesis, not yet a demonstrated cause. The
intervention below is designed to falsify it.

## One evidence-selected next experiment

Decouple critic warmup from actor warmup:

1. Keep the collector random and the actor frozen through update 3,000.
2. Train the world model and critic during that period instead of suppressing
   both actor and critic.
3. Start actor optimization at update 3,000 with the existing actor loss and
   all other settings unchanged.

Use a staged stop rule:

1. First run one seed slightly beyond the 3,000-step boundary. Verify that
   critic loss is active during warmup, actor parameters remain frozen, and a
   step-3,000 Q probe is no longer uniform/constant and improves agreement with
   real simulator action values. Stop if it does not.
2. Only if that gate passes, run seeds 0, 1, and 2 concurrently to 5,000 updates.
   Reject the change if deterministic behavior remains near 9-10 or actors
   become constant. Extend toward the existing 10,000-step contract only for a
   clear cross-seed behavioral improvement.

Do not add entropy changes, a new actor objective, or world-model auxiliaries
to this experiment. Those would prevent attribution. Do not return to Pong
until a state-dependent CartPole policy is stable across seeds.

## Critic-warmup intervention result

The isolated intervention was implemented in `99e543a` and tested with seed 1
in `runs/cartpole_critic_warmup_canary_99e543a_seed1_3200/`. The run completed
3,200 updates normally in 12 minutes 36 seconds. The actor remained frozen and
the critic trained through step 3,000, followed by 200 actor updates.

The mechanical hypothesis was correct: critic warmup removed the uniform cold
critic. Imagined critic-value standard deviation was `0.915` at step 1,000 and
`2.107` at step 3,000, rather than `0.00000165` when actor and critic previously
started together.

The learning hypothesis failed its predeclared gate:

| Probe | Q vs real accuracy | Q/real correlation | Q preference | Actor preference |
|---|---:|---:|---|---|
| Old seed 1, step 3,000 | 0.442 | -0.353 | action 1: 442/512 | near-uniform probabilities |
| Critic warmup, step 3,000 | 0.529 | -0.201 | action 1: 500/512 | near-uniform probabilities |
| Critic warmup, step 3,200 | 0.385 | -0.332 | action 1: 420/512 | action 1: 512/512 |

After only 200 actor updates, mean action-1 probability reached `0.641` and the
actor preferred action 1 on every probe state. Critic pretraining therefore
made scalar state values non-uniform without producing reliable
counterfactual action ranking. The actor still amplified the bad ranking into
a constant deterministic policy.

The three-seed extension was not launched. The intervention was reverted in
`987f747` because it failed the scientific gate and made the 3,000-step warmup
substantially more expensive. This moves the first broken boundary one step
earlier than the original hypothesis: the main problem is not merely a cold
critic at actor startup, but failure to learn a trustworthy state-dependent
`Q(s, a)` or equivalent policy-improvement target.

## One-step hybrid boundary probe

The Q probe was extended to separate learned transition quality from learned
multi-step value quality. For each real state and first action, it decodes the
world model's predicted next physical state, injects that state into the real
CartPole simulator, and uses the trusted heuristic for the remainder of the
30-step horizon. A second hybrid also applies the learned continuation
probability to the real downstream survival return.

With a perfect one-step transition, this hybrid exactly reproduces the fully
real rollout score and action preference; focused tests enforce that invariant.
All checkpoint comparisons used the same 512 states and probe seed 17.

| Checkpoint | State hybrid decisive states | State hybrid accuracy when decisive | State + continue accuracy | State + continue correlation | Full learned Q accuracy | Full learned Q correlation |
|---|---:|---:|---:|---:|---:|---:|
| Old seed 1, step 3,000 | 29/104 | 0.897 | 0.740 | 0.412 | 0.442 | -0.353 |
| Old seed 1, step 5,000 | 61/104 | 0.705 | 0.606 | 0.347 | 0.442 | -0.009 |
| Critic warmup, step 3,000 | 6/104 | 1.000 | 0.548 | 0.134 | 0.529 | -0.201 |
| Critic warmup, step 3,200 | 37/104 | 0.865 | 0.500 | 0.304 | 0.385 | -0.332 |

The decoded one-step state usually ranks actions correctly when it creates a
non-tied downstream survival difference, consistent with the existing perfect
one-step action-effect sign results. However, it often does not separate the
actions at all. The continuation head supplies a preference on almost every
actionable state, but that preference is not consistently trustworthy; in the
critic-warmup step-3,000 checkpoint it preferred action 1 on 497/512 states.

Most importantly, every hybrid has positive correlation with real action
advantage (`0.134` to `0.412`), while every full model/critic Q estimate is
non-positive (`-0.353` to `-0.009`). Useful local counterfactual information is
therefore present but weak, then degraded by learned continuation, longer
latent rollout, and/or terminal critic value. The next investigation should
audit that post-transition value path before changing the actor again.

## Imagined value-path decomposition

The same frozen checkpoints were probed at horizons 1, 2, and 3 with the
terminal critic bootstrap enabled and disabled. A survival-only objective
isolated the continuation head. The table reports Pearson correlation between
each learned action difference and the trusted 30-step simulator difference;
the magnitude column is the mean absolute learned action difference over all
512 states.

| Checkpoint | Reward only h1 corr / magnitude | Continue h1 corr / magnitude | Survival h3 corr / magnitude | Critic h1 corr / magnitude | Full Q h3 corr / magnitude |
|---|---:|---:|---:|---:|---:|
| Old seed 1, step 3,000 | 0.315 / 0.000006 | -0.113 / 0.000535 | -0.289 / 0.002111 | -0.113 / 0.012540 | -0.353 / 0.022466 |
| Old seed 1, step 5,000 | 0.262 / 0.000004 | 0.112 / 0.000309 | 0.176 / 0.001158 | -0.294 / 0.027216 | -0.009 / 0.022478 |
| Critic warmup, step 3,000 | 0.100 / 0.000001 | -0.137 / 0.000711 | -0.047 / 0.004252 | -0.185 / 0.058060 | -0.201 / 0.026104 |
| Critic warmup, step 3,200 | 0.035 / 0.000001 | -0.115 / 0.000473 | -0.200 / 0.002006 | -0.262 / 0.071846 | -0.332 / 0.018846 |

The immediate reward head is directionally positive in these checkpoints but
effectively action-neutral: its differences are only `1e-6` to `6e-6`. That is
not inherently surprising for CartPole, whose non-terminal reward is almost
always one; useful control ranking must instead come from predicting how the
first action changes future survival.

That survival signal is already unreliable at the first learned transition.
Three of four continuation correlations are negative, and its preferences are
strongly action-biased. Recursive rollout grows the magnitude but does not make
the ranking reliable. Finally, the horizon-1 critic bootstrap is 24 to 152
times larger than the continuation difference and is negatively correlated at
every checkpoint. It can therefore dominate the weak local signal before a
long rollout even begins.

The failure is compound rather than one isolated actor bug:

1. The decoded next state contains useful action effects, but the continuation
   head does not turn them into a calibrated survival difference.
2. The state critic assigns much larger values without reliable local ordering,
   overwhelming what little counterfactual signal exists.
3. The actor then faithfully optimizes this malformed imagined target and can
   collapse to one action.

The next bounded investigation should audit continuation-target construction
and state-value calibration against exact CartPole remaining lifetime. It
should use frozen replay/checkpoints first, so any subsequent training change
is selected by evidence rather than combining another set of speculative actor
or entropy changes.

## Reliability follow-up

Interrupted manifests correctly record `status: interrupted` and evaluation
history, but incorrectly retain `progress.train_step: 0` and `env_steps: 0`.
Fix this bookkeeping issue separately from the learning experiment so it does
not alter the scientific intervention.
