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

This initial diagnosis was superseded by the later critic-warmup and supervised
critic probes below. At this stage, the first apparently broken boundary was
imagined action-value construction. The
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

## Replay critic target audit

The subsequent code audit found a concrete off-by-one error in replay critic
grounding. A collected replay row contains the action and reward that produced
the observation stored in that same row. Consequently, posterior row `t`
represents the state *after* reward `t`, and its value target must begin with
reward `t + 1`.

The grounding path instead paired posterior row `t` with reward `t`. It also
trained the last posterior in a sampled sequence even though no following
replay transition was present, bootstrapping that state from its own imagined
annotation. The optional direct replay-return path had the same shift.
CartPole's nearly constant reward makes the error difficult to see in aggregate
loss curves, but moving the terminal boundary by one state corrupts precisely
the remaining-lifetime ordering needed for control.

Both paths now form targets only from adjacent real replay rows: posterior
states `[:-1]` are paired with rewards and continuation flags `[1:]`, and a
pair mask requires both rows to be real and in the same episode. Focused tests
use deliberately distinct rewards to enforce this temporal contract.

This defect was introduced with replay critic grounding in `1372c6d` on
2026-01-25. The documented CartPole runs that reached deterministic return 500
were on 2026-01-19 and 2026-01-20, before that path existed. That history is
supporting evidence, not proof of causality; the aligned implementation still
requires a fixed-seed canary under the current training stack.

### Alignment-fix canary

The aligned implementation in `baf5846` was run with seed 1 under the frozen
current configuration. The run was stopped after the step-4,000 gate rather
than spending its full 5,000-update budget:

| Checkpoint | Deterministic return | Q/real correlation | Actor preference |
|---|---:|---:|---|
| Step 3,000, before actor/critic updates | 11.25 | -0.298 | near-uniform |
| Step 4,000 | 9.15 | -0.010 | action 1: 512/512 |

Replay alignment moved Q correlation toward zero, but did not create a useful
positive ranking or prevent policy collapse. The change remains a correctness
fix, while its isolated learning hypothesis is rejected. No additional seeds
were launched.

## Duplicate continuation discount audit

The same audit found a second mismatch against the [reference DreamerV3
implementation](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/agent.py).
With `contdisc: true`, this repository trains the continuation head
on `(1 - terminal) * (1 - 1 / horizon)`. Its output therefore already contains
the task discount. The imagination path nevertheless multiplied that learned
probability by `gamma` again in lambda returns, actor/critic trajectory weights,
enumerated action values, and Q-critic targets. Replay targets similarly
multiplied real continuation flags by both the horizon discount and `gamma`.

At the current values, every non-terminal transition was discounted by
`0.997 * 0.997` rather than `0.997`. The reference implementation explicitly
uses an additional multiplier of one when `contdisc` is enabled, and applies
the horizon discount separately only when the continuation target does not
contain it.

The local implementation now follows that contract: learned discounted
continuation is consumed with multiplier one, while undiscounted real replay
continuation is consumed with `gamma`. A focused test fixes both modes at the
API boundary. This is the next isolated canary; replay alignment remains in
place because the reference replay equation independently confirms its
successor-reward indexing.

### Single-discount canary

The corrected equation in `32f27af` was tested with the same seed-1 contract
and stopped at the step-4,000 gate:

| Checkpoint | Deterministic return | Q/real correlation | Q preference | Actor preference |
|---|---:|---:|---|---|
| Step 3,000, before actor/critic updates | 28.35 | -0.392 | action 1: 352/512 | mixed, near-uniform probabilities |
| Step 4,000 | 9.15 | 0.176 | action 0: 479/512 | action 1: 512/512 |

The full Q correlation became positive within the run, unlike the previous
canary, but its action accuracy was still only `0.500` and its preference was
strongly biased. More importantly, the actor moved in the opposite direction
from the checkpoint's enumerated Q ranking and collapsed to the minimum-return
policy. The return-equation correction is retained for conformance, but the
isolated learning hypothesis again fails and no additional seeds are justified.

This run's pre-actor deterministic return was 28.35, versus 11.25 in the prior
same-seed canary, even though the discount change cannot affect world-model-only
training. Asynchronous collection/replay and GPU sampling therefore prevent a
same-seed run from being treated as a perfectly paired replicate. Within-run
checkpoint changes remain the stronger evidence.

The next bounded experiment should revisit critic-only warmup now that both
replay indexing and continuation discounting are correct. It must gate Q
ranking at step 3,000 before the first actor update. If the critic is not
positive, state-dependent, and materially above chance at that boundary, stop
without training the actor; otherwise allow only a short actor canary and test
whether actor/Q agreement improves rather than merely checking return.

### Corrected critic-warmup gate

Commit `c76d60d` decoupled critic training from actor warmup and was run to
exactly 3,000 updates in
`experiments/2026-07-18_213401_CartPole-v1/`. Checkpoint optimizer state
confirmed zero actor updates and active critic updates. The collector remained
on random actions for the entire run.

The predeclared Q gate failed:

| Measurement | Step 3,000 result |
|---|---:|
| Q vs real-rollout accuracy | 0.423 |
| Q/real correlation | -0.137 |
| Q preference histogram | action 0: 354; action 1: 158 |
| Survival-only h3 correlation | 0.067 |
| Critic-bootstrap h1 correlation | -0.157 |
| Actor optimizer states | 0 |
| Critic optimizer states | 6 |

The corrected survival rollout was weakly positive, but the trained critic
again flipped the local action ranking negative. Because the gate failed before
policy learning, no actor continuation or additional seeds were run. The
critic-warmup intervention was reverted in `c59899a`; the replay-indexing and
single-discount correctness fixes remain.

This isolates the next question: can the existing posterior latent and critic
architecture fit a trusted real remaining-lifetime target at all? A frozen,
supervised critic probe can answer that without another online training run. If
it succeeds, the missing boundary is target grounding; if it fails, critic
observability/capacity is the earlier problem.

### Supervised critic representability probe

A frozen probe collected 4,096 states across 184 random-policy episodes from
the corrected critic-warmup checkpoint. For every physical state it computed
the exact discounted remaining lifetime under a deterministic constant-action
policy. Whole episodes, not individual rows, were assigned to the train/test
split; the held-out set contained 897 states. Fresh distributional critics with
the repository's normal architecture were trained either on posterior latents
or directly on symlogged physical state.

| Target policy | Input | Held-out correlation | Held-out MAE |
|---|---|---:|---:|
| Always action 0 | Posterior latent | 0.645 | 2.45 |
| Always action 0 | True state | 0.942 | 0.67 |
| Always action 1 | Posterior latent | 0.387 | 3.00 |
| Always action 1 | True state | 0.948 | 0.62 |

The critic architecture easily fits the trusted target from true state, so
output capacity and the distributional loss are not the primary limitation.
The frozen posterior latent also supports meaningful held-out value prediction
in both directions, although with a clear information gap and action asymmetry.
That makes representation quality a secondary constraint rather than an excuse
for the online critic's negative correlation.

The first failed boundary is therefore target grounding: the online critic is
trained mostly from imagined reward/continuation and its own bootstrapped
annotations, even though a real return target is learnable from the same latent.
The next intervention should deliver exact episode return-to-go with replay
rows and train the critic against it at a controlled scale. It should not use
the existing finite-window `critic_real_return_scale` target unchanged, because
random subsequence boundaries truncate that target before the episode ends.

## Slow critic target audit

A subsequent comparison with the reference
[DreamerV3 agent](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/agent.py)
found another concrete value-target discrepancy. The reference computes
imagined lambda returns from its slow value model and uses that same slow value
as the policy baseline. The local trainer maintained and checkpointed an EMA
critic, but used it only as a distributional regularizer. Its lambda-return
bootstrap and actor baseline both came from the online critic being optimized.

That makes the main target network self-referential: an online value error
immediately changes the target used to train that same value model, while the
EMA copy can only pull the output distribution toward its older shape. This is
consistent with the critic bootstrap overwhelming the smaller learned survival
signal and becoming negatively ordered during critic-only warmup.

The trainer now decodes the EMA critic for lambda-return construction and the
actor baseline while retaining the online critic for the trainable value loss.
The full-episode replay-return path remains disabled by default, so the next
critic-only seed-1 gate isolates this slow-target correction against the prior
failed gate. Exact real-return grounding remains the next bounded intervention
if the slow-target gate fails.

## Critic target and free-bits gates

Four seed-1 critic-only runs used the same 3,000-update configuration and the
same 512-state, seed-17 simulator counterfactual probe. The actor optimizer had
zero state in every checkpoint. Each row changed only the named intervention
from the preceding corrected baseline.

| Gate | Q accuracy | Q/real correlation | Q preference | Result |
|---|---:|---:|---|---|
| Corrected replay/discount baseline | 0.423 | -0.137 | a0 354 / a1 158 | fail |
| Slow critic target | 0.510 | -0.192 | a0 116 / a1 396 | fail |
| Preserve replay scale 0.3 | 0.510 | -0.303 | a1 512 | fail |
| Exact full-episode return, scale 1.0 | 0.490 | -0.274 | a0 284 / a1 228 | fail |
| Hard one-nat free bits | **0.865** | **0.540** | a0 189 / a1 323 | **pass** |

The hard-free-bits checkpoint is
`experiments/2026-07-19_154109_CartPole-v1/checkpoints/checkpoint_final.pt`.
Its continuation-only action ranking also reached `0.875` accuracy and `0.664`
correlation. This is the first intervention to make both the learned survival
signal and the critic bootstrap positive, state-dependent, and materially above
chance before policy learning.

The causal defect was the interaction of two individually motivated changes.
The implementation had moved from averaging categorical KL across latent
variables to summing it, matching the reference scale, but retained a
straight-through free-bits estimator introduced for the older averaged loss.
Thus the summed KL continued pushing the posterior toward the prior even after
falling below the one-nat budget. In the two preceding canaries, the logged KL
hit the `1.0` floor by update 25 and remained there for 115/120 and 117/120
measurements while gradients continued through the floor.

Reference Dreamer uses the summed KL together with a hard one-nat clamp. The
default now matches that pair. Straight-through behavior remains available as
an explicit research override, but is no longer the authored configuration.
The next bounded gate is a short actor continuation from the passing checkpoint;
it must improve actor/Q agreement and deterministic behavior before any
multi-seed extension.

## Reliability follow-up

Interrupted manifests correctly record `status: interrupted` and evaluation
history, but incorrectly retain `progress.train_step: 0` and `env_steps: 0`.
Fix this bookkeeping issue separately from the learning experiment so it does
not alter the scientific intervention.
