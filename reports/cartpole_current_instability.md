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

**Correction, 2026-07-21:** the source claim above was wrong. At reference
commit [`e3f0224`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/configs.yaml#L108-L110),
both `imag_loss.slowtar` and `repl_loss.slowtar` are `False`. The implementation
selects the online value when that flag is false and uses the slow value only
as a regularization target
([`agent.py` lines 397--422](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py#L397-L422)).
The local July 19 change therefore moved away from, rather than toward, that
reference contract. Its historical experiments remain valid measurements of
the slow-target variant, but their source-conformance interpretation is
withdrawn.

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

### Short actor continuation

The passing checkpoint was resumed from update 3,000 to 3,500 in
`experiments/2026-07-19_155543_CartPole-v1/`. The collector loaded checkpoint
weights before filling replay, and deterministic evaluation used 20 episodes
every 100 updates.

| Update | Mean deterministic return |
|---:|---:|
| 3,100 | 9.20 |
| 3,200 | 21.90 |
| 3,300 | 21.85 |
| 3,400 | 26.15 |
| 3,500 | **28.30** |

At update 3,500, the fixed counterfactual probe retained `0.846` Q accuracy
and positive `0.349` Q/real correlation. Actor-vs-real accuracy reached `0.875`,
with state-dependent preferences (action 0 on 142 states and action 1 on 370)
rather than the previous single-action collapse. Mean probe-state actor entropy
was `0.366`; training entropy was still `0.594` at the last logged pre-final
point.

This clears the policy-transfer gate for one seed: a trustworthy pre-actor Q
signal remained useful under policy-driven replay and produced improving real
behavior. It does not establish that CartPole is solved or stable. Multi-seed
and longer-horizon testing should wait until the separately identified RSSM
sequence-gradient and posterior-unimix correctness defects are fixed and the
same frozen contract is rerun.

## RSSM correctness gate

The follow-up audit found two independent RSSM implementation defects:

1. The recurrent deterministic state and sampled stochastic state were
   detached before being saved for the next observed timestep. This severed
   backpropagation through time: a loss at timestep `t + 1` could not train how
   timestep `t` produced its recurrent or stochastic state. Commit `25a94ef`
   retains that graph during training; collection and evaluation already run
   under `no_grad`.
2. Posterior logits were mixed with 1% uniform probability before sampling and
   the already-mixed logits were returned to the loss, which mixed them a
   second time. The KL loss therefore saw approximately 1.99% posterior unimix
   while the prior and deployed inference path saw 1%. Commit `a82cf7e` now
   returns raw posterior logits to the loss and uses a separate once-mixed copy
   only for sampling.

Focused regressions prove that a timestep-2 loss now produces nonzero gradients
through timestep-1 deterministic and stochastic states, and that the posterior
head's raw logits reach the loss unchanged. The full fast suite passed with 69
tests, along with compile, scoped type checking, and the multiprocess CPU smoke
test.

The combined seed-1 gate used the same critic-only 3,000-update contract as the
hard-free-bits canary. Relative to the hard-free-bits-only checkpoint, one-step
state MSE fell from `0.0939` to `0.00844`. The fixed 512-state counterfactual
probe reported:

| Measurement | Step 3,000 result |
|---|---:|
| Q vs real-rollout accuracy | 0.875 |
| Q/real correlation | 0.559 |
| Survival-only h3 correlation | 0.663 |
| Q preference histogram | action 0: 288; action 1: 224 |
| Actor optimizer states | 0 |

The checkpoint is
`experiments/2026-07-19_160309_CartPole-v1/checkpoints/checkpoint_final.pt`.
Retaining sequence gradients increased peak CartPole VRAM only from roughly 18%
to 19%; trainer RSS was approximately 3.9 GiB, with no observed growth trend.

### Combined actor continuation

The passing combined checkpoint was resumed for 500 actor updates in
`experiments/2026-07-19_161541_CartPole-v1/`. Deterministic evaluation again
used 20 episodes every 100 updates.

| Update | Mean deterministic return |
|---:|---:|
| 3,000 | 44.05 |
| 3,100 | 281.25 |
| 3,200 | 394.70 |
| 3,300 | 437.35 |
| 3,400 | **481.00** |
| 3,500 | 464.70 |

At update 3,500, the fixed probe retained `0.865` Q accuracy and improved to
`0.807` Q/real correlation. Actor-vs-real accuracy was also `0.865`. The actor
remained state-dependent, preferring action 0 on 234 probe states and action 1
on 278, with mean probe-state entropy `0.141`. Thus useful model knowledge now
crosses the critic-to-actor boundary and produces near-solved real behavior in
this seed.

This is strong causal evidence for the combined corrected training stack, but
it is still one seed and must not be reported as stable CartPole learning. The
frozen replication contract is seeds 0, 1, and 2; 3,000 critic-only updates;
the same fixed seed-17 Q probe at that boundary; and only for passing Q gates,
500 actor updates with deterministic 20-episode evaluations every 100 updates.
The primary behavioral metric is final deterministic episode return. The
boundary metrics are Q/real correlation, Q accuracy, actor/real accuracy, and
actor preference balance. A seed fails early if its step-3,000 Q correlation is
not positive and its accuracy is not materially above chance. No Pong run is
justified until this contract establishes whether the correction is robust to
training seed.

### Three-seed boundary replication

The missing seed-0 and seed-2 critic gates were run under the frozen contract.
All three seeds passed before actor training:

| Seed | Q accuracy | Q/real correlation | Q preference |
|---:|---:|---:|---|
| 0 | 0.856 | 0.396 | a0 252 / a1 260 |
| 1 | 0.875 | 0.559 | a0 288 / a1 224 |
| 2 | 0.875 | 0.557 | a0 253 / a1 259 |

The staged 500-update actor continuations ended at deterministic returns 153.6,
464.7, and 500.0 for seeds 0, 1, and 2. Every final actor agreed with trusted
real-rollout preferences on `0.865` of actionable states, and every actor used
both actions. This confirms that the original critic-to-actor boundary failure
is repaired across seeds, but the seed-0 return remained too low to call the
behavior stable.

The staged continuations also exposed a research-protocol confound. Checkpoints
restore model, optimizer, normalization, and step state, but not replay contents
or random-number-generator state. Each continuation therefore started from a
new 16-episode replay population. Fixed random-policy-state model MSE rose from
`0.0078` to `0.173` in seed 0 and from `0.0091` to `0.278` in seed 2, while seed
2 nevertheless solved the task. This MSE is best interpreted as loss of
off-policy support, not a direct behavioral failure metric.

Two uninterrupted pre-optimizer-fix baselines then kept the same replay buffer
across the step-3,000 actor boundary:

| Seed | Step 3,100 | Step 3,200 | Step 3,300 | Step 3,400 | Step 3,500 | Best |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 461.70 | 466.50 | 496.25 | 437.75 | 300.40 | 496.25 |
| 1 | 465.85 | 289.25 | 500.00 | 500.00 | 500.00 | 500.00 |

Replay continuity materially improved seed 0, so a resumed continuation is not
an exact substitute for uninterrupted training. The result is still unstable:
seed 0 lost 196 points between its best and final evaluation, and seed 1 had a
211-point transient drop. Their final Q accuracies remained `0.827` and `0.875`
and final actor/real accuracy was `0.875` for both, so this is no longer the old
constant-policy or malformed-Q failure.

### LaProp bias-correction audit

The optimizer audit found a concrete mismatch in the next boundary. Local
LaProp maintained exponential RMS and momentum accumulators but applied neither
bias correction, despite a comment claiming this matched the source. The
reference DreamerV3 optimizer bias-corrects both accumulators before applying
the learning rate.

For the default betas, the local cold optimizer's first parameter update was
`3.16` times the configured learning rate. The RMS bias remains material during
the short actor window: after 500 steps it still makes the normalized update
about `1.6` times too large. This is selective evidence because the world-model
and critic optimizers already have 3,000 updates at actor release, whereas the
actor optimizer has zero state and receives the full cold-start error exactly
where the return curves overshoot.

The implementation now bias-corrects both stages and has a numerical two-step
regression against the reference equations. Pre-fix optimizer checkpoints must
not be resumed into the corrected experiment because their accumulator history
was produced under different update semantics. The isolated validation is a
fresh uninterrupted seed set with the same 3,500-update budget and metrics; no
learning-rate, entropy, replay, or architecture changes are allowed.

#### Corrected-LaProp result at the frozen learning rate

The fresh uninterrupted seed-0 and seed-1 runs rejected the stability
hypothesis at `actor_lr=3e-5`:

| Seed | Step 3,100 | Step 3,200 | Step 3,300 | Step 3,400 | Step 3,500 | Best |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 167.65 | 316.00 | 321.50 | 256.70 | 165.55 | 321.50 |
| 1 | 113.50 | 178.65 | 158.65 | 132.85 | 126.75 | 178.65 |

Bias correction is retained as a source-conformance fix, but it removes update
amplification that the current CartPole configuration had implicitly relied on.
Both final critics remained trustworthy (`0.865`/`0.894` Q accuracy and
`0.667`/`0.799` Q correlation), and both actors reached `0.875` real-action
agreement without becoming constant. Their probe-state entropies remained
`0.225` and `0.275`, roughly twice the pre-fix values after the same 500 actor
updates. This is a policy optimization rate problem, not recurrence of the old
representation or critic failure.

The next isolated configuration gate doubles only `actor_lr` to `6e-5`. This is
inside the old optimizer's effective late-window range while avoiding its
`3.16`-times first-step spike. Accept it only if fresh uninterrupted seeds learn
faster and reduce best-to-final collapse; otherwise reject the rate hypothesis
instead of stacking entropy or objective changes.

#### Actor-rate gate

Doubling only `actor_lr` restored fast initial learning but failed the stability
gate:

| Seed | Step 3,100 | Step 3,200 | Step 3,300 | Step 3,400 | Step 3,500 | Best |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 402.20 | 138.35 | 120.50 | 127.80 | 70.25 | 402.20 |
| 1 | 224.20 | 340.10 | 256.80 | 297.25 | 174.35 | 340.10 |

Final Q accuracy remained `0.865`/`0.875` and actor/real agreement remained
`0.875`/`0.865`; the faster actor did not fail because its critic lost action
ranking. Seed 1's fixed random-state decoder MSE reached `6.45`, but action-
effect signs and survival ranking remained useful, so that number primarily
shows severe off-policy support drift. The scalar-rate hypothesis is rejected.

The next audit found a more direct policy-loss mismatch. Reference DreamerV3
divides advantages by a running return-percentile scale and configures its
separate advantage normalizer as `none`. The local loss already divides by its
return scale, then—when `normalize_advantages=true`—also centers and unit-scales
every imagined batch. This local z-score was added after the historical failure
runs. It guarantees a full-strength policy gradient even when current imagined
advantages are tiny or unstable, which can make a still-correct actor boundary
oscillate as on-policy replay changes.

The next isolated gate returns to corrected `actor_lr=3e-5` and changes only
`normalize_advantages=false`. Keep the same uninterrupted 3,500-update contract.
If it fails, do not add entropy tuning; inspect running return normalization and
actor-gradient magnitudes before selecting another objective change.

#### Advantage-normalization gate

Disabling per-batch advantage z-scoring delayed policy confidence but did not
stabilize behavior:

| Seed | Step 3,100 | Step 3,200 | Step 3,300 | Step 3,400 | Step 3,500 | Best |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 331.95 | 366.25 | 353.55 | 192.50 | 144.75 | 366.25 |
| 1 | 172.55 | 294.50 | 331.60 | 211.65 | 195.85 | 331.60 |

The reference-conforming setting is still preferable, but the isolated causal
hypothesis is rejected. Final random-policy-state probes still looked healthy:
Q accuracy was `0.856`/`0.827`, Q correlation was `0.529`/`0.420`, and actor
accuracy was `0.865`/`0.875`. Those aggregate metrics therefore did not explain
the best-to-final collapse.

### On-policy boundary probe

A new deterministic probe follows each checkpoint's actor in the real
simulator and evaluates both actions at every visited state using the same
30-step real-dynamics counterfactual as the fixed random-state probe. It also
computes the learned three-step model/critic Q values from the matching latent
history. Best and final checkpoints were compared on the same 20 initial-state
seeds:

| Seed/checkpoint | Return | Actor vs real | Q vs real | Q/real corr | Actor vs Q |
|---|---:|---:|---:|---:|---:|
| 0 best, step 3,200 | 374.05 | 0.487 | 0.465 | 0.001 | 0.827 |
| 0 final, step 3,500 | 147.90 | 0.468 | 0.437 | 0.046 | 0.969 |
| 1 best, step 3,300 | 371.90 | 0.475 | 0.485 | 0.012 | 0.884 |
| 1 final, step 3,500 | 204.05 | 0.484 | 0.378 | 0.107 | 0.878 |

The old random-state result and the new on-policy result are not contradictory.
They measure different state distributions. On states induced by the actor,
learned Q is at or below chance and essentially uncorrelated with real action
advantage. The actor nevertheless follows that Q preference `83%` to `97%` of
the time. The first broken boundary is therefore no longer actor optimization:
the model/value objective fails on the policy-induced distribution, and the
actor increasingly follows it with high confidence.

### Collection-to-learning ratio failure

Runtime metrics explain how those on-policy states escape learning. The trainer
enforced `replay_ratio` only when learning was ahead of collection. Nothing
stopped a fast CartPole collector from running far ahead of the learner. The
background replay thread continuously drained the bounded process queue into a
512-episode circular buffer, so the queue's intended backpressure never reached
the collector.

For the two no-z-score runs, update 3,000 had already received 755k/784k raw
environment steps. From update 3,000 to 3,100 alone, each collector added about
60k steps and 2,600 episodes: more than five complete replay-buffer turnovers
for only 100 learner updates. Across the full run, the configured ratio was 16
replayed non-burn-in transitions per raw environment frame, while the measured
full-run ratio was only about `0.31`, roughly 52 times below target. During the
critical updates 3,000 to 3,100 it was about `0.16`, 100 times below target.
Policy data was generated and evicted much faster than it could be learned,
often under stale weight snapshots.

The runtime now applies symmetric backpressure. Startup collection remains
unrestricted until the minimum replay population exists. Afterwards, each
completed learner update releases exactly
`batch_size * non_burn_in_sequence_length * action_repeat / replay_ratio` raw
environment frames of collection budget. The replay drain waits at whole-
episode boundaries when that budget is exhausted, allowing the bounded queue to
fill and block collectors. The authored `recent_fraction` is also now passed to
sampling instead of silently using the helper's default.

A 20-update multiprocess CPU smoke test completed with 187 admitted environment
steps: exactly its 67-step startup population plus the 120-step learner-issued
budget in that run. In general, admitted collection can temporarily exceed the
budget by at most one complete episode. The prior implementation could admit
thousands during the same small run.

The first two-seed training attempt exposed why that episode tolerance is
necessary. Seed 1 completed with best return `451.70` at update 3,300 and final
return `252.10`; seed 0 reached `457.75` at update 3,200 and then stopped making
progress. Both maintained approximately `16` replayed non-burn-in transitions
per admitted environment step. The stop was a pacing deadlock, not a model
failure: the exact limiter required the next roughly 500-step episode to fit
inside a smaller remaining allowance, while the learner could not perform
enough additional updates on already admitted data to earn that allowance.

The limiter therefore admits a whole episode whenever any positive collection
budget exists, then records the overshoot as debt. Further episodes remain
blocked until learner updates repay the debt. This preserves bounded
backpressure without requiring indivisible episodes to exactly fit a fractional
step allowance. Because the pacing implementation changed during the attempt,
neither run is the final causal benchmark; both seeds must be restarted from
the corrected commit. Success still requires improved on-policy Q ranking and
reduced best-to-final return collapse, not merely a strong random-state probe.

### Corrected pacing benchmark

Two fresh seeds completed from clean source commit `a2e7890` with the
episode-debt limiter. The runs used the frozen no-advantage-normalization
configuration and changed only collection pacing relative to the prior
controlled baseline.

| Seed | Admitted frames | Effective replay ratio | Best/final return | On-policy probe return | Actor/real | Q/real | Q-real correlation | Actor/Q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21,118 | 15.91 | 338.70 / 338.70 | 311.20 | 0.492 | 0.491 | -0.011 | 0.780 |
| 1 | 21,328 | 15.75 | 398.15 / 398.15 | 424.70 | 0.485 | 0.567 | 0.057 | 0.686 |

The effective ratio is
`updates / admitted_frames * batch_size * non_burn_in_length`; its small
shortfall from 16 is the expected startup population and final episode debt.
Both runs completed normally without another pacing stall. Each final
checkpoint was also its best checkpoint, so the best and final probe results
are identical.

This is a meaningful causal improvement. The prior controlled final returns
were `147.90` and `204.05`, whereas the paced runs finished at `338.70` and
`398.15` while consuming only about 21k admitted frames rather than roughly
1.1 million. Their action histograms were balanced (`49%/51%` and `50%/50%`),
so the prior constant-action deployment collapse did not recur. The run curves
were not monotonic, however: seed 1 fell from `281.50` at update 3,300 to
`193.40` at 3,400 before recovering to `398.15` at 3,500.

The pacing fix is therefore necessary but not sufficient. On actor-induced
states, actor-versus-real counterfactual agreement remains at chance. Learned
model/critic Q ranking is also at chance for seed 0 and only modestly above it
for seed 1, with essentially zero correlation to the real counterfactual value
margin. The actor no longer merely copies that learned Q ranking, and useful
closed-loop behavior emerges despite weak local 30-step action ranking. The
next diagnostic should split the combined learned Q estimate into model-rollout
error and critic-bootstrap error on these same frozen checkpoints; extending
training or changing another loss before that split would confound the first
remaining boundary.

### On-policy model/value decomposition

The split probe first corrected a diagnostic validity problem. The original
enumerated Q helper propagated categorical prior probabilities as continuous
latent vectors, whereas RSSM training and normal imagination use sampled
one-hot latents. Because the decoder is trained on one-hot forward values, a
soft latent can decode to an unsupported average state. The expanded probe
therefore reports both the historical probability-vector estimate and a
categorical-mode one-hot estimate. The mode result did not rescue the boundary,
so the failure below is not an artifact of that probe mismatch.

| Seed | Actor/real balanced acc. | Full Q h3 balanced acc. | Critic bootstrap h1 balanced acc. | Model-only h3 balanced acc. | Q/real corr. |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.539 | 0.487 | 0.538 | 0.485 | -0.011 |
| 1 | 0.494 | 0.599 | 0.503 | 0.508 | 0.057 |

Balanced accuracy is required here because the trusted preferred action is
strongly imbalanced along each actor trajectory; raw accuracy can make an
almost-constant preference look useful. Neither the model-only return nor the
exact horizon-one critic bootstrap contribution is robust across seeds, and
their value-margin correlations remain near zero.

The physical error is more specific. Every non-time-limit probe failure—19 of
19 for seed 0 and 13 of 13 for seed 1—ended at the cart-position boundary, not
the pole-angle boundary. The one-step model preserved the sign of both actions'
effects on cart and pole velocity on 100% of visited states, so it learned the
direction of the local control physics. However, decoding either action and
then handing that state to the real simulator produced an exact 30-step tie on
essentially every state. Mean one-step state MSE was `0.352`/`0.231`, dominated
by cart position (`0.795`/`0.613`) and cart velocity (`0.565`/`0.258`), while
pole angle error was only `0.0038`/`0.0014`. Conditioning the next latent on the
real next observation did not materially improve those errors, so the gap is
not isolated to the dynamics prior. The agents learned to balance the pole but
not to keep the cart centered, matching their observed termination modes.

Replay sampling explains how this selective on-policy gap can persist after
collection pacing is fixed. The local buffer samples an episode uniformly and
then samples one subsequence inside it. A 500-step policy episode therefore has
the same base probability as a roughly 20-step random episode. At actor release
the two runs had 840/846 collected episodes; at completion they had 878/883.
Only 38/37 of the 512 retained episodes were therefore generated during actor
training. With the configured recent mixture, only about 11% of sampled batch
sequences came from that policy era. Weighting episodes by valid sequence starts
would put the same data near 40–45%.

Reference Dreamer replay inserts each valid sequence start as a separately
sampleable item and applies uniform selection over those items. The local
episode-uniform policy is therefore not a harmless storage detail: it
systematically underweights the long, cart-boundary-reaching trajectories that
the model most needs after the actor improves. The next isolated fix should
sample episodes in proportion to their number of valid sequence starts while
preserving the existing recent/uniform mixture and collection budget.

### Sequence-start sampling result

Commit `b4cd42c` changed episode selection probability to the number of valid
sequence starts, independently within the recent and global pools. Sampling is
with replacement so a long episode can contribute multiple windows to one
batch. The focused contract uses three- and six-row episodes with sequence
length three and verifies weights of one and four. The full suite passed with
78 tests, along with compile, scoped type checking, and a 20-update,
two-collector CPU smoke run.

The frozen warmup benchmark completed from that commit:

| Seed | Manifest frames | Best return (step) | Final return |
|---:|---:|---:|---:|
| 0 | 21,338 | 347.05 (3,500) | 347.05 |
| 1 | 21,464 | 156.80 (3,300) | 136.55 |

Compared with the episode-uniform paced results of `338.70` and `398.15`, this
does not support sequence-start weighting as a sufficient learning fix: seed 0
was essentially unchanged and seed 1 was much worse. The implementation is
still retained because it gives every stored training window equal base
probability and matches the reference replay semantics. Its causal performance
hypothesis is rejected rather than promoted from a single favorable seed.

### Actor-warmup removal result

The warmup contract also exposed a poor training abstraction. With
`actor_warmup_steps=3000` in a 3,500-update run, the collector remained random
and the actor optimizer remained inactive for 86% of the budget. The critic was
trained against dreams from a frozen randomly initialized actor, followed by an
abrupt policy/distribution change and only 500 joint updates. Commit `05b2d96`
removed this option from authored and runtime configuration. Actor and critic
now train together from the first scheduled AC update. Historical config
snapshots containing the retired field remain inspectable because the field
does not affect model construction. The full suite passed with 79 tests,
compile and scoped type checks passed, and a two-collector CPU smoke run
completed normally.

Two fresh seeds then ran the same sequence-weighted, paced, no-advantage-
normalization contract with joint training from update zero:

| Seed | Manifest frames | Best return (step) | Final return | Probe best/final |
|---:|---:|---:|---:|---:|
| 0 | 21,446 | 500.00 (2,600) | 241.60 | 500.0 / 239.0 |
| 1 | 21,554 | 275.25 (2,800) | 169.70 | 204.1 / 184.2 |

Seed 0 first exceeded 200 at update 1,500 and achieved deterministic 500 at
updates 2,600 and 2,800, before regressing. This is substantially earlier than
the old update-3,000 actor release and justifies removing the freeze. Seed 1
learned less and both final checkpoints remained below their best checkpoint,
so joint training improves time-to-learning but does not solve stability.

The deterministic on-policy decomposition localizes the best-to-final change:

| Seed/checkpoint | Probe return | Actor/real balanced | Q/real balanced | Q/real corr. | Actor/Q | One-step MSE | Cart-position MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 best, step 2,600 | 500.0 | 0.651 | 0.656 | 0.162 | 0.764 | 0.172 | 0.079 |
| 0 final, step 3,500 | 239.0 | 0.556 | 0.564 | -0.019 | 0.974 | 0.847 | 3.141 |
| 1 best, step 2,800 | 204.1 | 0.480 | 0.416 | -0.123 | 0.901 | 0.116 | 0.304 |
| 1 final, step 3,500 | 184.2 | 0.518 | 0.562 | 0.013 | 0.869 | 0.120 | 0.294 |

For seed 0, the solved checkpoint proves that the implementation can cross the
model-to-value-to-actor boundary. The later failure is not constant-action
deployment: the final action histogram remains balanced. Instead, one-step
cart-position MSE grows by about 40 times, posterior-conditioned position MSE
also grows from `0.102` to `3.156`, three-step Q correlation falls to zero, and
the actor follows that degraded Q preference `97.4%` of the time. Seed 1 never
develops robust on-policy Q ordering in the first place.

The next bounded investigation should therefore be offline and checkpoint-
local: measure replay cart-position coverage and per-state reconstruction/Q
metrics across seed 0 checkpoints 2,500 through 3,500 to identify the first
update at which position representation degrades. Do not change another loss
or return to Pong until that transition is characterized.

#### Preregistered checkpoint-localization contract

- **Hypothesis:** seed 0's best-to-final behavioral collapse begins when its
  posterior/decoder loses absolute cart position on policy-relevant histories;
  three-step Q ordering should degrade at the same checkpoint or afterward.
- **Frozen models:** seed 0 periodic checkpoints at updates 2,500, 3,000, and
  3,500 plus the best checkpoint at update 2,600, all from commit `05b2d96`.
- **Fixed data:** deterministic histories generated by the step-2,600 best
  actor and step-3,500 final actor, using ten episodes each and reset seeds
  17--26. Every target checkpoint must process identical observations and
  source-policy actions for a given history set.
- **Primary metrics:** current-state posterior cart-position MSE, one-step prior
  and posterior cart-position MSE, three-step Q/real balanced accuracy and
  value-margin correlation, actor/Q agreement, and the same metrics stratified
  by absolute cart position.
- **Stop rule:** identify the first available checkpoint interval containing
  degradation, document uncertainty, and stop without changing training.
- **Limitation:** historical replay contents were not persisted. Exact replay
  cart-position coverage cannot be recovered from these runs; fixed policy
  histories are an evaluation distribution, not a reconstruction of what each
  learner batch contained. This missing evidence is itself an observability
  finding and must not be papered over with inferred replay statistics.

#### Checkpoint-localization execution and result

`scripts/probe_cartpole_checkpoint_drift.py` was added to drive the real
environment with one frozen source actor while every target checkpoint consumes
the identical observation/action history. It reports current posterior
reconstruction, both-action one-step prior/posterior reconstruction, and
three-step Q ranking overall and by absolute-position bin. Summary contracts
and bin-edge tests passed, followed by a one-episode end-to-end smoke test. The
full repository finished with 82 passing tests; compile, the supported scoped
type gate, and a direct type check of the new probe also passed.

The two full commands used one CPU numerical thread each and ran concurrently:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 uv run --frozen --extra cpu python \
  scripts/probe_cartpole_checkpoint_drift.py \
  --source-checkpoint <checkpoint_best.pt-or-checkpoint_final.pt> \
  checkpoint_step_2500.pt checkpoint_best.pt \
  checkpoint_step_3000.pt checkpoint_final.pt \
  --device cpu --episodes 10 --seed 17 \
  --rollout-horizon 30 --model-horizon 3
```

The two source substitutions were the seed-0 `checkpoint_best.pt` and
`checkpoint_final.pt`; the four targets came from the same run's checkpoint
directory. Outputs, including per-state CSV evidence and console logs, are in
`experiments/2026-07-20_cartpole_checkpoint_drift_best_source/` and
`experiments/2026-07-20_cartpole_checkpoint_drift_final_source/`.

On the solved step-2,600 source histories, all four targets see the same 5,000
states, mean source return 500, mean absolute position `0.587`, and maximum
absolute position `1.253`:

| Target step | Current posterior x MSE | One-step prior x MSE | Q/real balanced | Q/real corr. | Actor/Q |
|---:|---:|---:|---:|---:|---:|
| 2,500 | 0.066 | 0.055 | 0.600 | 0.119 | 0.774 |
| 2,600 best | 0.081 | 0.075 | 0.642 | 0.116 | 0.744 |
| 3,000 | 0.060 | 0.049 | 0.646 | 0.131 | 0.999 |
| 3,500 final | 0.110 | 0.097 | 0.654 | 0.145 | 0.986 |

The final-policy source histories cover 2,572 identical states, mean source
return `257.2`, mean absolute position `0.635`, 90th-percentile position
`1.854`, and maximum position `2.400`:

| Target step | Current posterior x MSE | One-step prior x MSE | Q/real balanced | Q/real corr. | Actor/Q |
|---:|---:|---:|---:|---:|---:|
| 2,500 | 1.442 | 0.788 | 0.848 | 0.078 | 0.416 |
| 2,600 best | 0.318 | 0.163 | 0.735 | 0.087 | 0.302 |
| 3,000 | 1.281 | 0.982 | 0.566 | -0.008 | 0.961 |
| 3,500 final | 3.573 | 2.889 | 0.563 | -0.003 | 0.962 |

The self-source rows reproduce the earlier independently generated on-policy
measurements within the expected ten-versus-twenty-episode sampling difference:
step 2,600 prior x MSE is `0.075` versus `0.079`, while step 3,500 is `2.889`
versus `3.141`; final Q balanced accuracy is `0.563` in both probes. This is a
useful end-to-end check that fixed-history latent alignment is correct.

Position stratification shows where the change occurs. The table below uses
the fixed final-policy histories and current posterior reconstruction:

| Absolute x | States | Step 2,600 MSE | Step 3,000 MSE | Step 3,500 MSE |
|---:|---:|---:|---:|---:|
| 0.0--0.5 | 1,610 | 0.051 | 0.152 | 0.119 |
| 0.5--1.0 | 280 | 0.617 | 1.187 | 1.913 |
| 1.0--1.5 | 245 | 0.843 | 2.852 | 6.380 |
| 1.5--2.0 | 268 | 0.900 | 3.988 | 11.208 |
| 2.0+ | 169 | 0.686 | 5.617 | 23.044 |

The preregistered hypothesis is therefore refined, not simply accepted. The
model does not globally forget position: all checkpoints remain useful on the
central solved-policy distribution. It loses position-dependent
generalization outside that distribution, with the first available degradation
interval between updates 2,600 and 3,000. During the same interval, fixed-
history Q balanced accuracy falls from `0.735` to `0.566`, Q-margin correlation
falls to zero, and actor/Q agreement jumps from `0.302` to `0.961`. The actor at
step 3,000 still performs well on its own central trajectories, but the system
has already lost a reliable recovery signal on the future failure states.

The causal mechanism cannot yet be chosen. Long solved episodes may dominate
valid replay windows while short boundary failures contribute fewer starts,
but replay contents were not retained, so that remains a hypothesis. The next
evidence requirement is prospective logging of replay absolute-position
coverage and per-position-bin reconstruction error during a bounded rerun. No
loss or sampling intervention should be selected before that measurement.

#### Preregistered prospective replay-coverage contract

- **Hypothesis:** before seed 0 loses boundary-state position reconstruction,
  sampled learner batches lose representation of large absolute cart positions.
- **Intervention:** observability only. At existing scalar-log steps, detach the
  sampled post-burn-in states and their already-computed posterior decoder
  outputs. Log valid-row absolute-position coverage, physical reconstruction
  MSE for all four state components, and cart-position MSE within the same five
  absolute-position bins used by the offline probe. Do not alter losses,
  sampling, gradients, optimizer order, or policy behavior.
- **Frozen run:** one seed-0, 3,500-update replication of commit `05b2d96`'s
  joint-training contract after the telemetry commit: batch 8, sequence length
  16, burn-in 4, replay ratio 16, sequence-start weighting, no advantage
  normalization, and the same learning rates/evaluation protocol.
- **Primary comparison:** coverage and decoder error before update 2,600,
  during 2,600--3,000, and after 3,000. A coverage decline must precede or
  coincide with binned error growth to support the hypothesis. Stable coverage
  with rising error rejects it.
- **Stop rule:** complete one seed, compare the telemetry with evaluation and
  actor entropy, document the result, and stop before changing training.

The implementation stores detached raw states, physical-space posterior
reconstructions, and validity masks for post-burn-in learner rows in the
per-step metrics accumulator. The CartPole logger emits mean, 90th percentile,
maximum, and five-bin fractions for absolute cart position; physical MSE for
`x`, `x_dot`, `theta`, and `theta_dot`; and cart-position MSE for every populated
position bin. Padded sequence rows are excluded. These tensors do not enter a
loss and no stochastic operation was added.

Two focused summary tests cover padding and every bin edge. The full suite
passed with 84 tests, compile and the supported scoped type gate passed, and
the modified logging module passed a direct type check. A one-update non-dry
CPU integration run completed with manifest `75b7c15bbacd49ccbea73f7be7323928`
and MLflow run `4ebff07d94174d668cf9a90cd69f0cf1`; its persisted metric files contain
all coverage fractions and the populated central-bin reconstruction metric.
The smoke source was intentionally dirty because it preceded the telemetry
commit and is validation evidence, not an exact-comparison research run.

#### Prospective replay-coverage execution and result

The preregistered seed-0 run completed normally from clean telemetry commit
`5c02f3b`:

- output: `experiments/2026-07-20_cartpole_replay_coverage_seed0_3500/`
- manifest: `64f4a60775f743ed85f206fce59291ca`
- MLflow: `975541a26aec4a639a2c2df5de8bcabd`
- budget: 3,500 learner updates and 21,772 environment frames in 979.9 seconds
- result: best deterministic return `500.0` at update 2,400; final return
  `391.2` at update 3,500
- disposition: completed at `max_train_steps`; source was clean and the parent
  stopped and joined the collector normally

The exact command was:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
NUMEXPR_NUM_THREADS=2 UV_PROJECT_ENVIRONMENT=.venv-rocm \
uv run --frozen --extra rocm dreamer-train \
  hydra.run.dir=experiments/2026-07-20_cartpole_replay_coverage_seed0_3500 \
  general.device=cuda models.d_hidden=128 train.max_train_steps=3500 \
  train.batch_size=8 train.sequence_length=16 train.wm_lr=3e-4 \
  train.actor_lr=3e-5 train.critic_lr=8e-5 \
  train.actor_entropy_coef=1e-3 train.normalize_advantages=false \
  train.eval_every=100 train.eval_episodes=20 train.checkpoint_interval=500 \
  train.num_collectors=1 train.replay_buffer_size=512 \
  train.min_buffer_episodes=16 train.replay_burn_in=4 train.replay_ratio=16 \
  train.early_stop_ep_length=0 train.critic_real_return_scale=0.0 \
  train.free_bits_straight_through=false general.seed=0 \
  general.experiment_name=cartpole_replay_coverage_seed0_3500
```

`scripts/summarize_cartpole_replay_coverage.py` reads the persisted MLflow
filesystem metrics, retains the final value for each step, derives coverage
above absolute-position thresholds, and produces both the aligned 25-update
time series and half-open interval summaries. The checked output is
`replay_coverage_summary.json` inside the experiment directory. The primary
comparison is:

| Updates | Eval mean (range) | Mean fraction \|x\| >= 1 | Mean fraction \|x\| >= 1.5 | Mean replay x p90 | Mean decoder x MSE | Mean x MSE at 1.5--2.0 (batches) | Mean x MSE at 2.0+ (batches) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,400--2,599 | 496.33 (492.65--500.0) | 0.194 | 0.102 | 1.457 | 0.071 | 0.045 (4/8) | 0.109 (3/8) |
| 2,600--2,999 | 333.86 (241.35--500.0) | 0.269 | 0.132 | 1.518 | 0.129 | 0.259 (8/16) | 0.593 (6/16) |
| 3,000--3,500 | 410.81 (306.05--475.55) | 0.241 | 0.104 | 1.499 | 0.088 | 0.208 (13/20) | 0.184 (4/20) |

The evaluation curve is oscillatory rather than a single collapse: `500.0` at
2,400, `294.85` at 2,600, `500.0` at 2,700, `241.35` at 2,900, `447.0` at
3,000, `306.05` at 3,200, `475.55` at 3,300, and `391.2` at 3,500. Each point
is a 20-episode deterministic evaluation. Mean actor entropy only declined from
`0.470` in the solved window to `0.447` during 2,600--3,000 and `0.444`
afterward, so this run did not undergo a simple entropy or constant-action
collapse.

The replay-coverage hypothesis is rejected for this replication. Boundary
coverage did not decline before or during the performance loss. It increased
while decoder error and evaluation both worsened, then remained high while
decoder error and evaluation partly recovered. In particular, mean sampled
coverage at `|x| >= 1` rose by about 39% from the solved interval to
2,600--3,000. The result rules out insufficient *frequency of sampled boundary
states* as the immediate cause under sequence-start weighting. It instead
localizes the next question to why the learner intermittently fits or uses
those observed states poorly: representation interference, target/value drift,
or actor sensitivity remain live mechanisms.

This conclusion is deliberately bounded. Each telemetry point is one sampled
batch with at most 96 valid post-burn-in rows, so individual points are noisy.
Bin-conditional MSE is only emitted when a bin is populated; the table reports
the number of contributing batches and does not treat missing bins as zero.
The summaries average batches rather than individual states. The asynchronous
collector makes the exact trajectory scheduling-sensitive even with the same
seed, so this run replicates instability but is not a bitwise continuation of
the earlier step-2,600-to-3,500 collapse. Finally, one seed rejects this causal
account for the observed replication but does not estimate a population
effect.

No training intervention follows from this result yet. The next bounded
diagnostic should separate representation optimization from representation
interference on the same prospective run: determine whether boundary decoder
error rises because those samples receive too little effective gradient, or
because subsequent central-state updates undo that fit. Only after measuring
that distinction should a replay stratification, auxiliary state loss, or
optimizer change be selected.

#### On-policy baseline and reference replay-value gradient audit

Before selecting an intervention, the telemetry run's best and final
checkpoints were evaluated with the existing 20-episode, reset-seed-17--36,
three-step on-policy counterfactual probe. Artifacts are in
`experiments/2026-07-20_cartpole_replay_coverage_seed0_on_policy_probe/`.

| Checkpoint | Probe return | Actor/real balanced | Q/real balanced | Q/real correlation | Actor/Q | One-step MSE | Prior x MSE | Prior x-dot MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Best, step 2,400 | 500.0 | 0.598 | 0.655 | 0.008 | 0.923 | 0.166 | 0.423 | 0.206 |
| Final, step 3,500 | 378.6 | 0.510 | 0.496 | 0.073 | 0.914 | 0.199 | 0.161 | 0.566 |

This reproduces the important boundary even though this run did not end in a
full collapse. The actor continues to follow the learned three-step Q ordering,
but that ordering falls from useful to chance-level balanced accuracy. The
error is not simply cart-position reconstruction in this replication: prior
position MSE improves while prior cart-velocity MSE grows by 2.7 times.

A source audit then found a concrete semantic difference from the official
DreamerV3 implementation at reference commit
[`e3f0224`](https://github.com/danijar/dreamerv3/tree/e3f02248693a79dc8b0ebd62c93683888ddaccfe).
The reference configuration enables both replay value loss and
[`repval_grad: True`](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/configs.yaml#L108-L116).
Its replay path conditionally preserves the observed feature graph before the
value head
([`agent.py` lines 218--233](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py#L218-L233)).
Thus the real-sequence value objective shapes both the value head and the
control-relevant representation, while imagined actor/critic trajectories
remain stopped at the model boundary.

The local path does the opposite at two points:

```python
metrics.replay_posterior_states.append(h_z_joined.detach())
...
replay_logits = critic(replay_posterior[:, :-1].detach())
```

Consequently `critic_replay_scale=0.3` currently means "fit the critic head to
fixed latent features," not the reference algorithm's replay-value grounding.
This disconnect is directly relevant to the observed failure: boundary states
are sampled, but their latent/Q ordering can degrade without any value gradient
asking the representation to retain control-relevant distinctions.

The audit also found reference differences that are *not* part of the next
intervention: the current reference uses a 1,000-update optimizer learning-rate
ramp, online rather than slow value targets by default, and one joint optimizer.
Those remain separate candidates; combining any of them with representation
gradient reachability would make the result uninterpretable.

#### Preregistered replay-value representation-gradient gate

- **Hypothesis:** allowing only the existing replay value loss to update the
  observed encoder/RSSM representation will preserve on-policy action ordering
  after the policy first solves CartPole. Imagined actor and critic gradients
  must remain stopped at the world-model boundary.
- **Code intervention:** retain the graph for post-burn-in replay posterior
  states, remove the detach only on the configured replay-value critic input,
  and backpropagate the world-model and critic losses jointly so their shared
  graph is traversed once. Keep replay targets, slow-value targets, imagined
  states, actor inputs, sampling, loss scales, and optimizer settings unchanged.
  The disabled `critic_real_return_scale` path remains detached.
- **Mechanical acceptance:** a focused regression test must show that the
  replay-value term produces gradients in both the critic and observed latent,
  while its targets remain detached; the full suite, compile/type gates, and a
  process smoke run must pass.
- **Frozen run:** one clean seed-0, 3,500-update run with exactly the telemetry
  benchmark configuration. Preserve the prospective replay metrics and
  deterministic 20-episode evaluation every 100 updates.
- **Primary behavioral gate:** after the first evaluation at 500, no later
  evaluation may fall below 300, final evaluation must be at least 450, and the
  best-to-final gap must be at most 50. These thresholds distinguish stability
  from the baseline's repeated 241--299 regressions and final gap of 108.8.
- **Primary boundary gate:** rerun the same best/final on-policy probe. Final
  Q/real balanced accuracy must exceed the baseline `0.496`, actor/real balanced
  accuracy must exceed `0.510`, and actor/Q agreement must remain above `0.8`.
- **Stop rule:** stop after code validation, one seed, and its best/final probe.
  Revert the intervention if gradient reachability or runtime safety fails.
  Treat a behavioral miss as a rejected causal fix; only a clear pass justifies
  replication on seeds 1 and 2.

#### Replay-value representation-gradient result: rejected

The intervention was implemented in `c83c93c`. It extracted the replay-value
calculation into a focused helper, connected its online critic input to observed
post-burn-in features, and combined world-model and critic losses into one
shared-graph backward pass. Replay targets, the slow critic, imagined
trajectories, and the optional full-return path remained detached. A focused
test verified gradients in both the critic and non-bootstrap replay features,
zero gradient in the final bootstrap-only feature, and no gradients in target
annotations or slow-critic parameters.

Mechanical validation passed: 90 tests, bytecode compilation, the supported
scoped type gate, a non-dry one-update CPU process smoke, and a 20-update,
two-collector dry process run. The non-dry smoke used manifest
`675b8556b6234ca1a68633e7a77a8712` and MLflow run
`1360e6f518554d0f8a6c1b698eac536e`. Directly expanding Pyright to all of
`forward.py` and `core.py` still exposed their pre-existing dynamic-model and
possibly-unbound errors; this experiment did not claim a broader clean type
baseline.

The frozen seed-0 experiment then completed normally from clean commit
`c83c93c`:

- output: `experiments/2026-07-20_cartpole_repval_grad_seed0_3500/`
- manifest: `5e8f4dc5a71f4358acc95da334fc8a61`
- MLflow run: `cbc7767bc49b421a89852ba38e141f0e`
- budget: 3,500 learner updates and 21,571 environment frames
- elapsed time: 952.09 seconds
- best: 500.0 at update 2,200; final: 399.6; best-to-final gap: 100.4

The behavioral gate failed decisively. After scoring 500.0 at updates 2,200 and
2,300, the evaluations at updates 2,400--2,800 were respectively 14.2, 35.05,
13.05, 15.5, and 11.75. Performance later recovered to 343.25 at update 3,000,
fell again to 160.3 at 3,300, and recovered to 399.6 at 3,500. This is severe
oscillation rather than permanent erasure, but it violates all three stability
criteria: post-solve minimum at least 300, final at least 450, and gap at most
50. Seeds 1 and 2 were therefore not run.

Prospective replay telemetry again rules out boundary starvation during this
collapse:

| Update interval | Evaluation mean (range) | Replay fraction abs(x) >= 1 | Replay abs(x) p90 | Decoder x MSE | Actor entropy |
|---|---:|---:|---:|---:|---:|
| 0--2,199 | 88.06 (9.25--316.5) | 0.045 | 0.515 | 0.109 | 0.524 |
| 2,200--2,399 | 500.0 (500.0--500.0) | 0.164 | 1.374 | 0.105 | 0.529 |
| 2,400--2,999 | 34.67 (11.75--118.45) | 0.191 | 1.425 | 0.158 | 0.477 |
| 3,000--3,500 | 243.37 (160.3--399.6) | 0.197 | 1.449 | 0.153 | 0.468 |

The collapse interval contains slightly *more* boundary data than the solved
interval. Decoder x error rises by about 50%, but actor entropy remains
moderate, so this is neither missing boundary experience nor a constant-action
policy collapse.

The preregistered best/final on-policy probe is in
`experiments/2026-07-20_cartpole_repval_grad_seed0_on_policy_probe/`:

| Run/checkpoint | Probe return | Actor/real balanced | Q/real balanced | Q/real correlation | Actor/Q | One-step MSE | Prior x MSE | Prior x-dot MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Detached baseline, best 2,400 | 500.0 | 0.598 | 0.655 | 0.008 | 0.923 | 0.166 | 0.423 | 0.206 |
| Detached baseline, final 3,500 | 378.6 | 0.510 | 0.496 | 0.073 | 0.914 | 0.199 | 0.161 | 0.566 |
| Representation gradient, best 2,200 | 500.0 | 0.577 | 0.561 | -0.034 | 0.723 | 0.502 | 0.098 | 1.403 |
| Representation gradient, final 3,500 | 382.25 | 0.551 | 0.551 | 0.058 | 0.880 | 0.102 | 0.084 | 0.310 |

The final checkpoint technically clears the three boundary thresholds, but the
behavioral gate dominates: a fix that improves final action-ranking metrics
slightly while permitting a 500-to-11.75 collapse is not a stable solution.
At the solved checkpoint, one-step prediction error triples and prior
cart-velocity error is almost seven times the detached baseline. By the final
checkpoint those errors recover, consistent with competing objectives moving
the shared representation back and forth.

This result establishes two different facts. First, the original detach really
does prevent replay value loss from shaping the representation. Second,
removing it alone is not a valid local fix. In this repository, the shared
representation is updated by the world-model optimizer at `3e-4`, whereas the
critic head uses `8e-5`; the audited reference couples these losses through one
optimizer and also uses a learning-rate ramp. Thus the isolated transplant
changes not only gradient reachability but effectively applies the value
gradient to shared parameters at 3.75 times the critic learning rate. The
experiment does not prove that reference-style representation grounding is
wrong; it shows that it cannot safely be inserted into the local split-optimizer
training contract as a one-line semantic correction.

The intervention was reverted in `2161be3`. Its run directory, manifest,
checkpoints, probe, and summary remain as evidence. The next bounded step should
be observational: measure world-model and replay-value gradient norms and their
cosine similarity on the same representation parameters, then audit the full
optimizer contract (loss ownership, learning rates, clipping, and ramp) before
testing another coupled-gradient change.

#### Optimizer-contract audit and preregistered gradient diagnostic

The local optimizer contract differs from the audited reference in ways that
make raw loss-scale comparisons insufficient:

| Property | Frozen local CartPole contract | Official reference `e3f0224` |
|---|---|---|
| Optimizer ownership | separate world-model, actor, and critic LaProp instances | one optimizer over encoder, dynamics, heads, policy, and value |
| Learning rate | `3e-4` WM, `3e-5` actor, `8e-5` critic | `4e-5` for all modules |
| Startup schedule | full rate from update 0 | linear 0-to-`4e-5` ramp over 1,000 updates |
| Replay-value representation gradient | detached | enabled |
| Gradient clipping | AGC per local optimizer parameter set | AGC on the joint gradient before LaProp transforms |

The local rates are benchmark-specific overrides rather than defaults. With the
replay representation connected, its gradient reaches encoder/RSSM parameters
owned by the WM optimizer; it is therefore transformed using the WM optimizer's
state and `3e-4` step size, not the critic optimizer's `8e-5` rate. LaProp
normalizes each coordinate using its accumulated second moment, so the 3.75x
rate ratio does not alone predict a 3.75x parameter displacement. It does prove
that the rejected trial was not equivalent to the reference's joint optimizer.

The next diagnostic is preregistered as follows:

- **Question:** on an otherwise detached baseline run, are the hypothetical
  replay-value representation gradient and the authored world-model gradient
  opposed or badly imbalanced on the shared parameters, and does that relation
  worsen at or before behavioral degradation?
- **Observational implementation:** only on existing scalar-log updates, retain
  a live copy of the replay posterior and construct the same scaled replay-value
  objective used in `c83c93c`. Use `torch.autograd.grad` to inspect it and the
  world-model objective separately without writing `.grad` or adding the replay
  gradient to any optimizer update. Log raw L2 norms, norm ratio, and cosine
  similarity over parameters reached by both objectives, globally and for the
  encoder, recurrent dynamics, and posterior head.
- **Invariance requirements:** diagnostics default off; the normal replay-value
  training input remains detached; targets and EMA outputs remain detached;
  enabling diagnostics must not change model gradients or one-step parameter
  updates under a fixed RNG state. A focused test must establish this before a
  research run.
- **Frozen run:** one clean seed-0, 3,500-update run with the exact command and
  deterministic evaluation protocol used for the prospective replay-coverage
  baseline, plus the diagnostic flag. Sample diagnostics every 25 updates.
- **Primary analysis:** align diagnostic points to the evaluation curve. Compare
  the period before the first evaluation at least 450, periods at least 450,
  and subsequent periods below 300. Support for gradient interference as a
  *proximate* instability mechanism requires the degraded period's global mean
  cosine to be at least 0.10 lower or its median replay/WM norm ratio at least
  2x higher than the solved period, with the change present no later than the
  first degraded evaluation.
- **Stop rule:** run one seed only. If it never produces both a solved and a
  degraded interval, report the run as mechanically valid but behaviorally
  inconclusive. Do not change learning rates, add a ramp, or reconnect the
  gradient until these measurements are analyzed.

The diagnostic implementation is read-only and opt-in through
`general.research_gradient_diagnostics=true`. On scalar-log updates only, the
forward pass retains a second live reference to the already-computed replay
posterior and reconstructs the same target and slow-regularized replay value
loss at its authored sequence and `0.3` scale. The normal critic loss continues
to use detached features. `torch.autograd.grad` measures the two component
gradients without populating parameter `.grad`; normal backward and all three
optimizer steps remain unchanged. Metrics cover the global shared subspace and
the encoder, recurrent, and posterior-head groups. A checked summarizer aligns
each 25-update sample with the most recent deterministic evaluation and
classifies it as pre-solve, solved, degraded, or intermediate.

A focused invariance test verifies that diagnostic measurement leaves `.grad`
empty and that the later world-model and critic training gradients exactly
match an unmeasured control, including the resulting parameter updates. The
full suite passes with 92 tests, bytecode
compilation passes, and both the supported type gate and direct checks of the
new diagnostic/logger/summarizer modules pass. As before, broad direct checking
of `core.py` and `forward.py` reports their pre-existing dynamic-model and
possibly-unbound errors; the new diagnostic path introduces no distinct error
class or clean broad-type claim.

A 12-update non-dry CPU process smoke completed in 3.57 seconds with manifest
`287642c64b6b43f3b683a1023b22c5b2` and MLflow run
`dabe1cf229804bb6a6697e03c1b6f580`. It published weights, persisted diagnostic
metrics at updates 0 and 10, saved a final checkpoint, and shut down normally.
At update 10, global cosine was `-0.030` and the scaled replay/WM raw norm ratio
was `0.00041`. This tiny early-batch observation validates plumbing only and is
not treated as evidence for or against the research hypothesis.

#### Gradient-alignment execution and result

The preregistered run completed normally from clean commit `20ebf54`:

- output: `experiments/2026-07-20_cartpole_gradient_alignment_seed0_3500/`
- manifest: `3067e0ab16664fc59cd7351925c263c3`
- MLflow run: `6207c5ea8df748acbaf98144ee3abf5f`
- budget: 3,500 learner updates and 21,484 environment frames
- elapsed time: 950.71 seconds
- result: best evaluation 421.95 at update 1,900; final 134.6

The run never met the preregistered solved threshold of 450, so the primary
solved-versus-degraded association test is **inconclusive by its stop rule**.
The checked primary summary is `gradient_alignment_summary.json`; all 140
telemetry batches are correctly classified as pre-solve. Across the run, the
global gradient cosine has mean `-0.001`, median `-0.016`, and range
`-0.658--0.837`. The scaled replay/WM norm ratio has mean `1.322`, median
`0.721`, 90th percentile `3.026`, maximum `10.344`, and final logged value
`7.502`. Thus the rejected representation-gradient intervention was not a
small auxiliary perturbation: on many batches its hypothetical representation
gradient matched or exceeded the entire authored WM gradient, while its
direction varied from strongly opposed to strongly aligned.

For exploration only, a second checked summary lowers the performance bands to
at least 300 versus below 200; it is saved as
`gradient_alignment_exploratory_300_200.json` and does not replace the
preregistered gate.

| Exploratory band | Batches | Global cosine mean (median) | Replay/WM ratio mean (median) |
|---|---:|---:|---:|
| at least 300 | 16 | 0.037 (-0.009) | 1.639 (0.731) |
| below 200 after first >=300 | 32 | -0.012 (-0.040) | 1.050 (0.782) |

Neither global criterion is met: cosine is only 0.049 lower, not 0.10, and the
median norm ratio is 1.07x, not 2x. Encoder cosine shows a larger exploratory
shift (mean `0.215` to `-0.017`), but this was not the primary test and the
hypothetical gradient is detached in this baseline, so it cannot itself cause
the observed baseline degradation. The defensible conclusion is narrower:
objective scale and direction explain why reconnecting replay value loss was a
materially risky intervention, but this run does not identify replay/WM
gradient interference as the original instability mechanism.

This run also exposes a benchmark confound in `evaluate_policy()`. It uses
`seed + 1_000_000 + step * 1000`, so every checkpoint is evaluated on a
different set of 20 CartPole initial states. Each score is reproducible, and
different runs at the same update share seeds, but changes across updates mix
model change with test-set change. This affects the research curve and best
checkpoint selection. It does not erase prior fixed-seed evidence of real
degradation: the earlier best/final on-policy probe used seeds 17--36 for both
checkpoints and fell from 500.0 to 378.6.

Before another training intervention, evaluate every saved checkpoint from the
new run on the same fixed seeds 17--36, using posterior and action argmax exactly
as deployed evaluation does. Include periodic checkpoints 500--3,500 and the
best checkpoint at 1,900. Record all 20 returns plus mean, range, and solved
fraction. This is an offline measurement only. If the fixed-seed curve is much
smoother, repair the benchmark before interpreting more training changes; if it
retains the large reversals, the instability is genuinely in the learned
policy/model rather than primarily evaluation sampling noise.

The fixed-seed audit was run from clean evaluator commit `9987361`; its complete
per-episode output is `fixed_seed_checkpoint_evaluation.json` in the experiment
directory.

| Checkpoint update | Original changing-seed mean | Fixed seeds 17--36 mean | Fixed range | Fixed solved fraction |
|---:|---:|---:|---:|---:|
| 500 | 9.30 | 9.10 | 8--11 | 0.00 |
| 1,000 | 9.45 | 9.40 | 8--10 | 0.00 |
| 1,500 | 381.00 | 395.85 | 226--500 | 0.25 |
| 1,900 (best artifact) | 421.95 | 396.20 | 269--500 | 0.40 |
| 2,000 | 201.30 | 220.80 | 162--419 | 0.00 |
| 2,500 | 105.65 | 100.85 | 81--163 | 0.00 |
| 3,000 | 218.15 | 197.15 | 144--429 | 0.00 |
| 3,500 | 134.60 | 137.80 | 98--191 | 0.00 |

The fixed cohort preserves the large reversals and closely tracks the original
means. Evaluation sampling noise is therefore not the cause of this run's
instability: the same 20 initial states lose about 75% of their mean return from
update 1,900 to 2,500 and do not recover by 3,500. Nevertheless, changing seeds
across updates remains an avoidable benchmark and checkpoint-selection
confound. Future in-training evaluations will use one fixed seed cohort per run;
the update number remains a logging coordinate, not part of environment reset
state.

The fixed-cohort change is isolated in `evaluation_episode_seed()`: run seed 0
always evaluates episodes on reset seeds 1,000,000 onward, regardless of
checkpoint update. Validation passes with 95 tests, bytecode compilation, and
the supported scoped type gate. A one-update CPU process smoke with evaluation
enabled at update 1 completed normally, including collector shutdown. This
changes evaluation and best-checkpoint selection only; it does not alter the
training data, losses, or optimizer updates.

#### Preregistered exact-return critic-grounding gate

The next intervention revisits `critic_real_return_scale`, but only after the
hard-free-bits, RSSM sequence-gradient, posterior-unimix, LaProp, collection
pacing, sequence sampling, joint-training, and fixed-evaluation corrections.
The earlier scale-1 canary predates the first three model correctness fixes and
cannot answer the question on the current training stack.

- **Hypothesis:** adding an exact full-episode return-to-go loss to the critic
  head will prevent its on-policy action ordering from drifting after useful
  behavior appears. Complete episodes already carry the required targets, so a
  sampled subsequence's label includes rewards beyond its right boundary.
- **Causal variable:** change only `train.critic_real_return_scale` from `0.0`
  to `1.0`. Retain `critic_replay_scale=0.3`, detached replay features, current
  model/actor/critic learning rates, fixed collection ratio, sequence-start
  sampling, no advantage z-score, and all architecture/loss settings. Do not
  enable gradient-alignment diagnostics in this behavioral run.
- **Source:** code source `75024df`; launch from its clean docs-only descendant
  containing this preregistration. The run manifest must record the resolved
  commit and `dirty=false`.
- **Frozen run:** seed 0, 3,500 learner updates, batch 8, sequence length 16,
  burn-in 4, replay ratio 16, one collector, and fixed 20-episode evaluation
  every 100 updates. Use the same 21k-frame pacing contract as the detached
  baseline.
- **Primary behavioral gate:** after the first evaluation at least 450, no
  later evaluation may fall below 300; final must be at least 450 and the
  best-to-final gap at most 50. If the run never reaches 450, it fails rather
  than becoming an unbounded tuning search.
- **Boundary gate:** run the fixed seeds 17--36 on-policy best/final probe.
  Final Q/real balanced accuracy must exceed the detached telemetry baseline
  `0.496`, actor/real must exceed `0.510`, actor/Q must remain above `0.8`, and
  exact-return loss must remain finite.
- **Stop rule:** one seed and its best/final probe only. A clear behavioral and
  boundary pass may be replicated on seeds 1 and 2; otherwise reject scale-1
  head-only grounding and do not stack representation gradients or optimizer
  changes onto it.

#### Exact-return critic-grounding result: rejected

The frozen seed-0 run completed normally from clean preregistration commit
`48744f1`:

- output: `experiments/2026-07-21_cartpole_exact_return_seed0_3500/`
- manifest: `1c6e55604b8a405c966c0497e29c4df0`
- MLflow run: `b6b8d355ffc94a18b88163b914c4c3be`
- causal variable: `critic_real_return_scale=1.0`; all other frozen benchmark
  settings matched the detached baseline contract
- budget: 3,500 learner updates and 21,491 environment frames
- elapsed time: 1,165.98 seconds
- disposition: completed at `max_train_steps`; source was clean, final and best
  checkpoints were retained, and the collector stopped normally
- result: best deterministic return `292.3` at update 1,600; final return
  `41.2`; best-to-final gap `251.1`

The full fixed-cohort evaluation curve is:

```text
100:11.60  200:9.40   300:9.35   400:9.35   500:9.35
600:9.40   700:9.40   800:9.40   900:20.85  1000:9.35
1100:63.10 1200:156.00 1300:139.30 1400:154.85 1500:199.05
1600:292.30 1700:193.05 1800:159.35 1900:138.80 2000:115.20
2100:137.55 2200:123.00 2300:149.50 2400:134.25 2500:102.85
2600:163.60 2700:128.60 2800:129.10 2900:103.45 3000:139.90
3100:129.85 3200:109.80 3300:114.40 3400:105.65 3500:41.20
```

The behavioral gate fails every applicable criterion. The policy never reached
450, finished far below 450, and lost 251.1 return from its modest peak. Seeds
1 and 2 are therefore not authorized by the stop rule. The exact-return path
was numerically healthy: all 140 logged `loss.critic.replay_mc_return` values
were finite, with mean `4.544` and range `3.741--5.541`. This is a learning-
quality failure rather than target overflow, process failure, or missing loss
execution.

The evidence-selected replay intervals are summarized in
`replay_coverage_summary.json`:

| Updates | Eval mean (range) | Mean actor entropy | Mean replay fraction abs(x) >= 1 | Mean decoder x MSE | Mean decoder x-dot MSE |
|---:|---:|---:|---:|---:|---:|
| 0--1,499 | 44.34 (9.35--156.0) | 0.614 | 0.005 | 0.020 | 0.150 |
| 1,500--1,699 | 245.68 (199.05--292.3) | 0.426 | 0.108 | 0.275 | 0.313 |
| 1,700--2,499 | 143.84 (115.2--193.05) | 0.405 | 0.180 | 0.124 | 0.078 |
| 2,500--3,500 | 115.31 (41.2--163.6) | 0.388 | 0.102 | 0.041 | 0.038 |

The policy degrades while sampled boundary coverage remains material and both
state-decoder errors improve sharply. Entropy stays moderate rather than
approaching zero. This repeats the earlier conclusion that the immediate
failure is not boundary starvation, a numerically broken world-model loss, or
a simple constant-action collapse.

The preregistered best/final fixed-seeds-17--36 probe is in
`experiments/2026-07-21_cartpole_exact_return_seed0_on_policy_probe/`:

| Checkpoint | Probe return | Actor/real balanced | Q/real balanced | Q/real correlation | Actor/Q | One-step MSE |
|---:|---:|---:|---:|---:|---:|---:|
| Best, step 1,600 | 284.15 | 0.488 | 0.446 | -0.036 | 0.711 | 0.199 |
| Final, step 3,500 | 43.30 | 0.627 | 0.638 | 0.296 | 0.613 | 0.015 |

The final checkpoint clears the actor/real and Q/real thresholds but fails the
actor/Q requirement badly. This is not a successful intervention: final
behavior is poor, and the actor is no longer reliably choosing the action that
the same checkpoint's three-step critic/model planner ranks higher. The late
planner ordering and one-step model accuracy improve while behavior gets worse,
which localizes the remaining question to policy improvement or to a mismatch
between the planner diagnostic and the actor's actual learning target.

This conclusion is bounded. The probe's true-action label is a 30-step real-
simulator comparison, while its learned Q comparison uses a three-step model
horizon. The actor is trained from 15-step imagined lambda returns, so low
actor/Q agreement could partly reflect horizon mismatch rather than failure to
follow its own training objective. Before another training intervention, run
one read-only final-checkpoint probe with model horizon 15 on the same episodes
and seeds. If actor/Q agreement rises above 0.8, audit horizon-dependent model
value error; if it remains below 0.8, treat actor-to-value transfer as genuinely
broken. Do not change training or launch another seed for this check.

The read-only horizon-15 check is saved in
`experiments/2026-07-21_cartpole_exact_return_seed0_final_h15_probe/`. On the
same 20 episodes and 248 actionable states, learned Q retained `0.628` balanced
accuracy against the real rollout preference and `0.294` delta correlation,
but actor/Q agreement was only `0.601`. This is slightly worse than the
three-step agreement of `0.613`, not a recovery above `0.8`.

The subsequent source audit found that this does **not** yet establish a broken
actor-to-value transfer. `enumerate_first_action_values()` expands every future
action branch and returns the maximum branch for each forced first action. It is
a model-predictive planner value. The REINFORCE actor instead samples future
actions from its own policy and is trained on the resulting 15-step lambda
return. Matching the horizon does not remove this max-planner versus on-policy
objective difference. The existing `actor/Q` field is still useful as a
planner-distillation diagnostic, but it is not a direct test of whether the
actor follows its authored loss.

Before another training run, add one opt-in, read-only Monte Carlo diagnostic
that forces each first action and then reproduces the training rollout:
categorical prior sampling, future actions sampled from the 1% unimixed actor,
learned reward and continuation, slow-critic bootstrap, lambda `0.95`, and the
15-step horizon. Average 64 rollouts per first action on the final checkpoint's
same seed-17--36 actor trajectory and report the standard error of each action
difference. The causal variable is measurement only; no weights or training
configuration may change. If actor agreement with statistically separated
policy-conditioned action values is at least `0.8`, the gap is planner versus
policy improvement. If it remains below `0.8` on at least 100 separated states,
the actor optimizer is not tracking its own target. If fewer than 100 states
separate, the sampled target itself is too weak/noisy for a strong transfer
conclusion. Stop after this final-checkpoint diagnostic and its tests.

The opt-in diagnostic adds `--policy-q-samples` to the on-policy probe. Its
vectorized estimator leaves all models under `no_grad`, forces each candidate
first action, samples subsequent latent states and 1%-unimixed actor actions,
and reuses the authored lambda-return implementation. It reports per-action
Monte Carlo standard errors and a 95%-separated actor-agreement subset. With the
option disabled, the historical probe output and computation are unchanged.
Focused estimator and summary tests pass, as do all 99 fast tests, bytecode
compilation, the supported type gate plus direct checks of the diagnostic
modules, and a one-episode CUDA probe smoke.

The preregistered final-checkpoint result is saved in
`experiments/2026-07-21_cartpole_exact_return_seed0_final_policy_q_probe/`:

- 64 rollouts per first action, 15-step horizon, and the same 20 deployed actor
  episodes produced 866 states.
- 562 states had a policy-Q difference larger than 1.96 standard errors; actor
  agreement on them was `0.683`, failing the `0.8` gate with ample sample size.
- Across the 248 real-actionable states, policy-Q/real balanced accuracy was
  `0.632` and delta correlation was `0.287`. Among the 160 both actionable and
  statistically separated states, balanced accuracy was `0.679`.
- Mean absolute policy-Q difference was `1.316`. The 178 confident actor/Q
  mismatches still had mean absolute difference `1.057` and mean deployed actor
  confidence `0.885`, so the miss is not confined to near-ties or an uncertain
  actor.

This establishes a current-target mismatch at the final exact-return
checkpoint: the actor does not reliably select the action favored by the
sampled lambda-return objective it is authored to optimize. It does not yet
distinguish an actor implementation error from optimizer lag under moving
world-model/critic targets. The evidence-selected control is the detached
baseline's final checkpoint, measured with the identical 64-sample protocol.
If baseline agreement is at least `0.8`, attribute the new mismatch to the
exact-return intervention's effect on target consistency; if baseline also
fails, treat this as a systemic actor-optimization boundary. This is one
read-only checkpoint measurement; do not train or tune from it.

The detached control was run from clean diagnostic commit `32a5ad8` and is
saved in
`experiments/2026-07-21_cartpole_detached_baseline_final_policy_q_probe/`:

| Final checkpoint | Return | Confident states | Actor/policy-Q | Confident actionable states | Policy-Q/real balanced | Policy-Q/real correlation |
|---|---:|---:|---:|---:|---:|---:|
| Detached baseline | 378.6 | 4,131 | 0.949 | 2,130 | 0.438 | 0.074 |
| Exact-return scale 1 | 43.3 | 562 | 0.683 | 160 | 0.679 | 0.287 |

The control passes the `0.8` actor-agreement threshold by a wide margin. The
exact-return auxiliary therefore introduced the actor/current-target mismatch;
it is not a general action-index, sign, or REINFORCE plumbing error. At the same
time, the control exposes why the detached baseline remains unreliable: its
actor faithfully follows a policy-conditioned target whose real-action ranking
is worse than chance on the confident actionable subset. Exact-return grounding
improves that target ranking but changes it faster or less consistently than
the actor tracks it. This is a critic/actor time-scale and target-consistency
tradeoff, not a successful fix.

One bounded temporal audit will distinguish whether mismatch appears before or
after the exact-return run's behavioral peak. Reuse the identical 64-sample,
15-step, seeds-17--36 protocol on the exact-return best checkpoint at update
1,600 and periodic checkpoints 2,000 and 3,000; combine them with the existing
final result. Record return, confident actor/policy-Q agreement, and policy-Q/
real metrics. If agreement is at least `0.8` at the peak and later falls, target
drift/actor lag is the proximate mechanism. If it is already below `0.8` at the
peak, head-only grounding creates an inconsistent policy-improvement target
throughout useful behavior. Stop after these three read-only checkpoints; do
not alter training.

The temporal audit is saved in
`experiments/2026-07-21_cartpole_exact_return_policy_q_temporal_probe/`:

| Update | Probe return | Confident states | Actor/policy-Q | Confident actionable states | Policy-Q/real balanced | Policy-Q/real correlation | Mean abs policy-Q delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,600 best | 284.15 | 3,842 | 0.804 | 1,483 | 0.425 | -0.038 | 1.118 |
| 2,000 | 113.60 | 1,482 | 0.815 | 1,060 | 0.439 | 0.005 | 0.533 |
| 3,000 | 137.75 | 2,072 | 0.582 | 1,009 | 0.500 | 0.003 | 2.280 |
| 3,500 final | 43.30 | 562 | 0.683 | 160 | 0.679 | 0.287 | 1.316 |

The first behavioral degradation occurs while the actor still passes the
agreement gate: from update 1,600 to 2,000, return falls by 60% while actor/
policy-Q agreement remains `0.804--0.815` and the policy-conditioned target is
worse than chance against real action values. Thus actor lag is not the cause
of the initial collapse. The actor is following bad learned advice.

A second failure appears later. By update 3,000 the target's mean action margin
has quadrupled while actor agreement falls to `0.582`. At the final checkpoint
the target becomes materially more correct against real rollouts, but agreement
recovers only to `0.683` and behavior reaches its minimum. Exact-return
grounding therefore improves critic/model action ordering too late and with a
time scale the actor does not track. It neither prevents the original target-
quality collapse nor yields a stable later policy improvement.

The bounded conclusion is now stronger: reject scale-1 head-only exact-return
grounding as a training fix, retain the policy-Q diagnostic, and do not launch
replication seeds. The original first broken boundary remains learned on-policy
action-value quality; once that target changes, actor/target consistency becomes
a second boundary. A next intervention must address target construction and
policy improvement together or demonstrate, offline, that a frozen improved
target can produce a better deployed actor before another end-to-end run.

#### Preregistered frozen policy-improvement probe

- **Question:** is the exact-return final checkpoint's improved policy-Q target
  actionable, or does it fail as soon as a new actor changes the visited-state
  distribution?
- **Frozen sources:** exact-return final encoder, RSSM, reward/continue heads,
  and slow critic. Start from a copy of its deployed actor; never write back to
  the checkpoint or resume the trainer.
- **Dataset:** follow the original deterministic actor for reset seeds 17--36.
  At each visited posterior latent, estimate both 15-step policy-conditioned
  lambda returns with 64 rollouts per action. Retain only labels whose action
  difference exceeds 1.96 standard errors. Use a deterministic 80/20 train/
  validation split.
- **Actor fit:** fine-tune only the copied actor for 20 epochs with AdamW,
  learning rate `1e-3`, batch size 128, and ordinary cross-entropy. These are
  the existing frozen-latent supervision-probe settings, used as a capacity
  diagnostic rather than a proposed online optimizer.
- **Evaluation:** compare the original and fitted deterministic actors on the
  same 20 unseen reset seeds 10,017--10,036 with the frozen encoder/RSSM.
- **Gate:** the target is actionable only if held-out label accuracy is at least
  `0.8` and mean real return improves by at least 50 over the original actor.
  If fitting succeeds without behavioral gain, reject one-step frozen target
  distillation because its labels do not survive the induced distribution
  shift. If fitting itself fails, actor representation/capacity remains the
  nearer boundary.
- **Stop rule:** one checkpoint, split, optimizer setting, and seed. Do not tune
  epochs, learning rate, confidence cutoff, or labels after seeing the result.

The probe implementation was committed as `6659b11` after focused tests,
direct type checking, bytecode compilation, CLI validation, and a one-episode
CUDA smoke. The preregistered result is saved in
`experiments/2026-07-21_cartpole_exact_return_frozen_policy_improvement/`:

- collection visited 866 states and retained 581 confident labels, split into
  465 training and 116 validation examples; the label histogram was 260/321;
- the copied actor reached `0.998` train accuracy and `0.991` held-out accuracy,
  with final cross-entropy `0.0010`;
- on unseen reset seeds 10,017--10,036, the original actor scored `39.0` mean
  return and the fitted actor scored `44.55`, an improvement of only `5.55`;
- disposition: failed the preregistered `+50` behavioral gate despite easily
  passing the `0.8` held-out-fit gate. No hyperparameter or label iteration was
  run.

Actor representation capacity and ordinary supervised optimization are
therefore not the blocker at this checkpoint. The final policy-Q target is
locally learnable on the old actor's state distribution, but one-step
distillation does not produce robust closed-loop control when the fitted actor
changes that distribution. The earlier `0.679` policy-Q/real balanced accuracy
is a local diagnostic, not evidence of a globally useful policy-improvement
operator.

This rejects “make the actor catch up to the late exact-return target” as the
next fix. The primary boundary remains action-value target robustness under
policy-induced distribution shift; actor lag is secondary. Do not iterate
distillation or stack it into training. The next investigation should explain
why imagined action preference fails off the actor trajectory despite accurate
one-step state prediction—most directly by separating prior-latent rollout
error, learned continuation/value bootstrap error, and compounding horizon
error on matched real action sequences.

#### Preregistered matched-sequence rollout-fidelity audit

- **Question:** when the actor's real action sequence is held fixed, does
  multi-step return error come primarily from critic calibration at real
  posterior states, learned reward/continuation along the prefix, or carrying
  the critic through prior-latent dynamics?
- **Frozen checkpoints:** exact-return final and detached replay-coverage final.
  Use each checkpoint's deployed deterministic actor and slow critic without
  updating any parameters.
- **Data contract:** 20 episodes on reset seeds 17--36. Record each real state,
  chosen action, reward, posterior latent, and discounted real return-to-go.
  For every start with a complete matched prefix, replay the exact real actions
  through 64 categorical-prior samples at horizons 1, 3, 5, 10, and 15.
- **Decomposition:** compare the model's prefix plus prior-state critic bootstrap
  against real return-to-go. Decompose its signed error into (a) an oracle
  prefix plus critic at the real future posterior, (b) learned reward error
  under the real discount, (c) continuation error within the prefix and on the
  final bootstrap, and (d) prior-latent critic transport after controlling for
  the predicted discount. The five signed terms must sum to total error on
  every row. Also record decoded prior-state mean/sample MSE and predicted
  versus actual survival discount.
- **Primary decision:** at horizon 15 on the exact-return checkpoint, compare
  RMS oracle critic error with RMS model-minus-oracle rollout error. The larger
  term is the dominant boundary. Within rollout error, compare RMS reward,
  continuation-prefix, final-discount, and prior-latent critic-transport terms.
  Horizon trends and the detached checkpoint are required context, not
  additional interventions.
- **Limitations:** horizon rows require the real episode to survive the whole
  matched prefix, so terminal-transition continuation calibration is reported
  separately and long horizons describe pre-terminal states. Time-limit-
  truncated episodes must be identified rather than treated as physical
  failures.
- **Stop rule:** two frozen checkpoints, the five fixed horizons, and one
  64-sample seed stream. Do not train, tune thresholds, or add another model
  variant after observing the decomposition.

#### Matched-sequence rollout-fidelity result

The frozen audit completed normally from clean diagnostic commit `3d230b6`.
Its summaries and per-state decomposition rows are retained in
`experiments/2026-07-21_cartpole_matched_rollout_fidelity/`. Both checkpoints
used reset seeds 17--36, 20 deployed deterministic episodes, 64 categorical-
prior samples per start, and the preregistered horizons. No parameter was
updated.

| Checkpoint | Return | Real-state critic error at h15, RMS / mean | Additional rollout error at h15, RMS / mean | Reward RMS | Continue-prefix RMS | Final-discount RMS | Prior critic-transport RMS | Predicted / real h15 discount |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exact-return final | 43.3 | 15.910 / +14.070 | 10.572 / -7.922 | 0.007 | 2.232 | 8.803 | 3.560 | 0.692 / 0.956 |
| Detached baseline final | 378.6 | 78.946 / -54.005 | 11.169 / -7.039 | 0.102 | 0.687 | 8.279 | 3.904 | 0.879 / 0.956 |

The exact-return checkpoint satisfies the preregistered decision rule:
real-posterior critic calibration against the deployed trajectory is the first
and larger error boundary (`15.910` RMS versus `10.572` added by rollout).
Within rollout, the dominant component is continuation discount, not reward or
decoded dynamics. Its learned reward path is effectively exact, while the
final-discount term alone contributes `8.803` RMS. Decoded mean-state MSE grows
from `0.0148` at one step to `0.0749` at 15 steps, but the corresponding prior
critic-transport term is only `3.560` RMS.

The decomposition also reveals cancellation that an aggregate return metric
would conceal. At horizon 15 the exact-return critic overestimates the deployed
trajectory by `+14.070` on average, while model rollout subtracts `7.922` on
average, principally through excessive termination discount. The resulting
full-return error therefore looks smaller than either broken component.

The detached checkpoint provides context, not a controlled causal comparison:
its actor visits much longer and different trajectories. It has a similarly
sized rollout error and the same dominant final-discount term, but its slow
critic underestimates the single deployed trajectory by `54.005` on average.
That absolute error does not by itself imply a malformed critic optimizer. The
critic is trained for the stochastic imagined actor, whereas this audit uses
one deterministic deployed trajectory; CartPole's unobserved 500-step time
limit also makes finite realized return-to-go non-Markov near truncation. Two
of its 20 episodes truncated, versus none for the exact-return checkpoint.

The exact-return auxiliary has an additional semantic limitation: it fits
returns generated by historical replay behavior, not the continually changing
current actor. Its positive current-policy bias is therefore consistent with
the already-rejected off-policy head-only intervention and is not evidence of
a newly discovered return-indexing bug. What does transfer across both
checkpoints is the rollout result: reward prediction is not the blocker,
continuation removes too much future value, and prior-latent critic transport
is secondary at the actor's 15-step horizon.

One-step continuation calibration sharpens the next question. On the
exact-return trajectory, the prior predicts mean continuation `0.965` on real
nonterminal transitions and `0.942` on terminal transitions; the corresponding
detached values are `0.989` and `0.973`. The head separates the classes only
weakly and assigns very high continuation even to actual failures. Before any
training change, compare the continuation head on each real next-state
posterior with the same prediction after one prior transition. If posterior
calibration is good but prior calibration is poor, latent transport is the
boundary. If both are weak, continuation supervision/imbalance or head capacity
is the boundary. Do not change the actor, critic, or end-to-end configuration
until this frozen distinction is measured.

#### Preregistered posterior-versus-prior continuation audit

- **Question:** is excessive imagined termination discount caused by the
  continuation head itself, or by the transition from observation-conditioned
  posterior latents to imagination-only prior latents?
- **Frozen checkpoints and trajectories:** reuse the exact-return and detached
  final checkpoints, their deployed deterministic actors, 20 episodes, and
  reset seeds 17--36. Do not update parameters.
- **Matched transition contract:** for every real action, encode the actual next
  observation and evaluate continuation under 64 samples from its posterior.
  From the identical pre-action latent and action, evaluate the same head under
  64 samples from the learned prior. Retain the posterior-mode prediction used
  by deterministic deployment, physical terminal/truncation flags, distance to
  physical termination, posterior/prior KL, and prior probability assigned to
  posterior modal categories.
- **Metrics:** decompose prior effective-discount error exactly into posterior
  label error plus prior-minus-posterior transport error. Report RMS/mean on all
  transitions and terminal transitions, failure ROC AUC, terminal/nonterminal
  means, effective-discount Brier score, and distance-to-terminal strata.
- **Primary decision:** on the exact-return checkpoint's terminal transitions,
  compare posterior-label-error RMS with transport-error RMS. If transport is
  larger and posterior failure AUC is at least `0.8`, the prior/posterior
  dynamics gap is primary. If posterior error is larger or posterior AUC is
  below `0.8`, continuation prediction/supervision is primary. The detached
  checkpoint tests whether the classification generalizes.
- **Stop rule:** two checkpoints, the fixed 20 episodes and 64 samples, one
  sampling stream, and no training or threshold tuning. Choose the next
  intervention from this boundary result only.

#### Posterior-versus-prior continuation result

The frozen audit completed normally from clean diagnostic commit `a116e57`.
Artifacts are in
`experiments/2026-07-21_cartpole_continuation_posterior_prior/`; the exact
signed decomposition has zero residual on every transition.

| Checkpoint | Terminal rows | Posterior failure AUC | Posterior terminal / nonterminal discount | Posterior terminal-error RMS | Prior terminal-error RMS | Posterior-to-prior transport RMS | Terminal KL q/p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact-return final | 20 | 0.812 | 0.942 / 0.965 | 0.942 | 0.943 | 0.0020 | 0.496 |
| Detached baseline final | 18 | 0.902 | 0.972 / 0.988 | 0.972 | 0.973 | 0.0019 | 0.884 |

The primary decision classifies continuation prediction/supervision as the
broken boundary on both checkpoints. On exact-return terminal transitions,
posterior label error is about 462 times the RMS transport error; on the
detached checkpoint it is about 524 times larger. Supplying the real next
observation therefore does not make the head predict termination. Prior and
posterior expectations are nearly identical even though their categorical
latent KL is nonzero, so the earlier excessive rollout discount cannot be
attributed to losing an otherwise-good terminal prediction in latent
transport.

The head does retain weak risk ordering. Posterior failure AUC exceeds `0.8`
on both checkpoints, and continuation falls as trajectories enter the last ten
steps. But no posterior or prior prediction crosses half of the task discount;
terminal/nonterminal balanced accuracy at that decision point is exactly
`0.5`. The model acts like a slightly state-dependent average hazard estimator,
not a classifier that recognizes an observed terminal state.

This result supersedes historical measurements from older checkpoints where
posterior continuation on terminal rows reached `0.25--0.42`; it does not
contradict them. The present conclusion is scoped to the current training
stack and two current final checkpoints. Terminal sample sizes are 20 and 18,
but the effect is too large and consistent across all rows to be sampling
noise.

The source audit found a matching architectural divergence. The current
[reference DreamerV3 configuration](https://github.com/danijar/dreamerv3/blob/main/dreamerv3/configs.yaml)
uses a one-hidden-layer continuation MLP with RMS normalization and SiLU; its
size-1M override uses 64 hidden units. This repository's continuation predictor
has been one linear projection since `b4eb689`. That restriction is especially
material for CartPole's non-linear, two-sided terminal set: failure is the
union of positive and negative cart-position or pole-angle thresholds, not one
linearly separable half-space.

#### Preregistered continuation-head conformance canary

- **Hypothesis:** replacing only the linear continuation projection with the
  reference-style one-hidden-layer, `d_hidden`-wide RMSNorm/SiLU MLP will turn
  the existing terminal-risk ordering into calibrated imagined survival and
  prevent the policy/value collapse caused by weak continuation differences.
- **Causal variable:** add one hidden continuation layer. Keep the reward head,
  encoder, RSSM, actor, critic, losses, optimizer rates, replay sampling,
  collection ratio, and detached critic representation path unchanged. New
  Hydra runs use one layer; legacy checkpoint configs default to zero layers so
  all existing evidence remains loadable with its authored architecture.
- **Frozen run:** seed 0 for 3,500 updates under the exact prospective replay-
  coverage command: `d_hidden=128`, batch 8, sequence 16, burn-in 4, replay
  ratio 16, 16 startup episodes, 512-episode buffer, learning rates `3e-4`,
  `3e-5`, and `8e-5`, fixed 20-episode evaluation every 100 updates, no exact-
  return auxiliary, no actor warmup, and no advantage normalization. Retain
  replay coverage telemetry but add no other training intervention.
- **Mechanical gate:** on the final fixed seeds 17--36, posterior terminal
  effective discount must fall below `0.5`, posterior failure AUC remain at
  least `0.8`, and posterior-to-prior transport must remain smaller than
  posterior label error. This tests the intended mechanism rather than merely
  loss execution.
- **Behavioral gate:** the run must reach fixed-cohort mean return `450`, never
  fall below `300` afterward, finish at least `400`, and have best-to-final gap
  at most `100`. If it never reaches `450`, it fails rather than triggering
  tuning.
- **Stop rule:** one seed and its frozen continuation/rollout probes. Replicate
  seeds 1 and 2 only if both mechanical and behavioral gates pass. Otherwise
  reject the isolated head-capacity hypothesis and do not add class weighting
  or reward-head changes to the failed run.

#### Continuation-head conformance canary result

The canary completed normally at source commit `4648035` with manifest run ID
`4a87926bbf7141a3b67454653b968ce0` and MLflow run ID
`0604a4a67ac5496bbd77fcf1f7f3e517`. It performed 3,500 learner updates and
21,677 environment steps in 1,180 seconds. The run and its generated probes are
retained under
`experiments/2026-07-21_cartpole_continue_mlp_seed0_3500/`,
`experiments/2026-07-21_cartpole_continue_mlp_continuation_probe/`, and
`experiments/2026-07-21_cartpole_continue_mlp_rollout_fidelity/`.

The behavioral result was reachable control followed by collapse, not failure
to learn. Fixed-cohort evaluation remained near `9.35` through update 1,500,
then rose through `31.8`, `156.35`, `332.95`, and `437.60` at updates
1,600--2,000. It first exceeded `450` at update 2,100, reached `500.0` at
update 2,600, and immediately became unstable: `372.35`, `200.85`, `214.0`,
and `284.6` at updates 2,700--3,000. The final score was `247.4`. It therefore
failed all three preservation conditions: it fell below `300` after reaching
`450`, finished below `400`, and had a best-to-final gap of `252.6`. Actor
entropy remained moderate around the collapse, so this was not a deterministic
single-action entropy collapse.

The mechanical gate also failed. The best checkpoint solved all 20 fixed-seed
episodes to the 500-step truncation and consequently exposed no physical
terminal rows. At the final checkpoint, the same fixed seeds averaged `207.0`
and supplied 20 physical failures. Mean posterior effective discount was
`0.9597` on terminal transitions versus `0.9794` on nonterminal transitions;
failure AUC was only `0.7558`, and terminal balanced accuracy at half discount
was `0.5`. Terminal posterior label-error RMS was `0.9635`, while posterior-to-
prior transport RMS was only `0.0103`. The larger head therefore still learned
weak risk ranking rather than a calibrated termination decision.

Matched-action rollout decomposition preserves the earlier diagnosis while
showing that simple observation reconstruction is not the collapse boundary.
At horizon 15, learned reward error remained negligible at both best and final
checkpoints (`0.0173` and `0.0546` RMS). Final-discount error worsened from
`5.525` to `9.745` RMS, while mean decoded-state MSE improved from `0.1379` to
`0.0561`. Model-target correlation was effectively absent at both checkpoints
(`-0.111` and `0.034`). Absolute critic errors are not interpreted as a direct
implementation defect because the probe compares deterministic finite-horizon
realized returns with a critic trained for the stochastic imagined policy and
because CartPole's 500-step time limit is hidden from the observation.

The isolated capacity hypothesis is rejected, so seeds 1 and 2 and further
head tuning are stopped by the preregistered rule. The nonlinear head remains a
reference-conformance correction, but it is neither the mechanical nor the
behavioral fix. This single seed demonstrates that the current stack can learn
a solved controller and then trains away from it; the next question is where
the learned action ordering changes on an unchanged set of solved-policy
histories.

#### Preregistered fixed-history best-to-final drift audit

- **Hypothesis:** after the solved update-2,600 checkpoint, learned action-value
  ordering degrades on the same histories before or together with the actor.
  If true, changing state visitation is not required to explain the collapse.
- **Fixed histories:** drive seeds 17--36 deterministically with the best
  checkpoint's actor. Evaluate those identical observations and preceding
  actions with the best, update-3,000, and final update-3,500 checkpoints.
- **Frozen probe:** use `scripts/probe_cartpole_checkpoint_drift.py` on CUDA
  with 20 episodes, seed 17, real counterfactual rollout horizon 30, and model
  horizon 3. Retain per-state rows and summaries under
  `experiments/2026-07-21_cartpole_continue_mlp_fixed_history_drift/`.
- **Primary metrics:** actor-versus-real and learned-Q-versus-real balanced
  accuracy, correlation between learned and real action deltas, and actor-
  versus-Q agreement. Secondary metrics are posterior and one-step state MSE,
  with results stratified by cart position.
- **Decision rule:** classify the earliest measurable boundary as value/target
  drift if Q-versus-real accuracy or delta correlation falls while the actor
  continues to agree with Q; classify it as actor fitting/lag if Q remains
  useful while actor-versus-Q agreement falls; classify representation as
  primary only if fixed-history current-posterior or one-step errors degrade
  sufficiently to explain the ordering loss. Mixed movement remains a coupled
  instability rather than being forced into one class.
- **Stop rule:** this is a read-only three-checkpoint comparison. Do not change
  training from its result alone unless the fixed-history evidence identifies a
  specific boundary and a single bounded intervention.

The solved-history half of this audit completed with 10,000 identical states,
of which 794 had different 30-step real outcomes for the two first actions.
The best, update-3,000, and final checkpoints respectively achieved actor-
versus-real balanced accuracies `0.598`, `0.630`, and `0.628`; Q-versus-real
accuracies `0.625`, `0.626`, and `0.643`; and actor-versus-Q agreements
`0.914`, `0.990`, and `0.926`. The final checkpoint therefore has not forgotten
the solved checkpoint's familiar action ordering. Current-posterior cart-
position MSE also improves from `0.0724` at best to `0.0533` at final. Despite
that, each checkpoint's learned-versus-real action-delta correlation is weak
(`0.057`, `0.147`, and `0.056`), so the solution does not rest on a robust
global value margin.

This rejects simple fixed-distribution value forgetting and triggers one
preregistered symmetric extension before selecting a training intervention:

- **Hypothesis:** the final policy's closed-loop trajectories enter a different,
  harder state/history distribution on which learned action ordering is poor.
  The old best controller either retains a recovery advantage there, indicating
  policy drift, or also fails, indicating that the apparently solved controller
  never acquired a robust recovery landscape.
- **Fixed histories:** drive the same seeds 17--36 with the final checkpoint's
  deterministic actor; evaluate the best, update-3,000, and final checkpoints
  on those identical histories with the same horizons and metrics.
- **Decision rule:** a best-checkpoint actor/real advantage over final supports
  actor-policy drift; poor and similar best/final accuracy plus a broader cart-
  position distribution supports a shared out-of-distribution recovery gap;
  useful final Q with poor final actor/Q agreement supports actor fitting. Stop
  after this symmetric comparison and choose at most one intervention.

#### Fixed-history drift result

The generated evidence is retained under
`experiments/2026-07-21_cartpole_continue_mlp_fixed_history_drift/` and
`experiments/2026-07-21_cartpole_continue_mlp_fixed_history_drift_final_source/`.
The symmetric extension confirms a closed-loop recovery-distribution failure.
Final-policy histories average absolute cart position `0.750`, versus `0.387`
under the solved policy; `32.5%` of final-policy states have `|x| >= 1`, versus
`6.0%`, and `59.5%` are action-critical under the 30-step counterfactual,
versus `7.9%`.

On final-policy histories, best/update-3,000/final actor-versus-real balanced
accuracies are `0.334`, `0.468`, and `0.506`; Q-versus-real accuracies are
`0.448`, `0.493`, and `0.521`; and learned-versus-real action-delta
correlations are `-0.069`, `0.018`, and `-0.072`. The final actor agrees with
its Q preference on `96.2%` of action-critical rows, so actor fitting or lag is
not the first broken boundary. At `|x| >= 1.5`, final Q/real accuracy is near
chance or worse and delta correlation is strongly negative. The best checkpoint
also fails on those histories, so it learned a narrow stabilizing orbit rather
than a robust recovery controller that the final policy merely forgot.

The final model's current-posterior cart-position MSE on its own histories is
only `0.0471`, and its one-step prior MSE is `0.0339`; those aggregate errors are
slightly lower than the best model's values on the same histories. Although
errors grow in the rare `|x| >= 2` bin, visual-state reconstruction does not
track the much larger and earlier action-ordering failure. Combined with the
matched-rollout result, this classifies the first actionable boundary as
imagined value/target quality in recovery states. The actor faithfully follows
that bad ordering, and closed-loop action changes then move it from the narrow
solved orbit into an action-critical distribution.

No training intervention is selected from correlation alone. The next bounded
step is a source audit of imagination, lambda-return construction, critic
training, and actor objectives against the reference algorithm. A change is
permitted only if that audit finds a concrete semantic divergence capable of
producing wrong action ordering; otherwise the next experiment must target
recovery-state coverage explicitly and remain a single-variable canary.

#### Preregistered online-value-target conformance canary

The source audit found one concrete semantic divergence in the live value path.
Reference commit `e3f0224`, which remains upstream `main`, configures both
imagined and replay lambda returns with `slowtar: False`. The online value
prediction supplies the lambda-return bootstrap and policy baseline; the slow
value remains a distributional regularizer. Local commit `8bd2ad9` instead
changed both bootstrap and baseline to the slow critic based on the now-
corrected opposite source claim.

- **Hypothesis:** the slow target cannot adapt quickly enough when the policy
  begins visiting recovery states, so stale action ordering helps move the
  closed-loop controller out of its narrow solved orbit. Restoring the online
  target while retaining slow-value regularization will improve post-solve
  stability and recovery-state Q ordering.
- **Code intervention:** add an explicit authored
  `train.critic_slow_target=false`; use detached online value predictions for
  imagined lambda-return construction and the actor baseline when false. Keep
  the EMA critic, its `0.98` update decay, and its scale-1 distributional
  regularizer unchanged. Historical configs lacking the field default to true,
  preserving the semantics of their checkpoints and reports.
- **Not changed:** continuation/reward/world-model heads, replay-value scale and
  detach behavior, replay sampling and pacing, optimizer ownership and rates,
  actor objective and entropy, advantage normalization, horizons, and
  architecture sizes. Missing RMSNorm/depth and optimizer-ramp differences are
  recorded but deliberately excluded.
- **Frozen run:** seed 0 under the exact 3,500-update continuation-head canary
  command and fixed 20-episode evaluation cohort: hidden 128, batch 8, sequence
  16, burn-in 4, replay ratio 16, buffer 512/start 16, learning rates `3e-4`,
  `3e-5`, and `8e-5`, entropy `1e-3`, no advantage z-score, no exact-return
  auxiliary, and no actor warmup.
- **Behavioral gate:** reach mean return `450`; never fall below `300`
  afterward; final at least `400`; best-to-final gap at most `100`.
- **Boundary gate:** rerun fixed-seed best/final on-policy and symmetric fixed-
  history probes. Final Q-versus-real balanced accuracy on final-policy
  histories must exceed the slow-target canary's `0.521`, learned-versus-real
  action-delta correlation must become positive, and actor-versus-Q agreement
  must remain above `0.8`.
- **Stop rule:** validate and run seed 0 only. Replicate seeds 1 and 2 only if
  both behavioral and boundary gates pass. A failure rejects the stale-target
  causal hypothesis; it does not justify combining head-depth, optimizer-ramp,
  or replay-coverage changes.

#### Online-value-target canary result

The frozen seed-0 canary completed normally at source commit `43d43a1` with run
ID `d9c14b1ba1d74173aaa85a7d8531f05c` and MLflow run ID
`3de8bb86ef06442b9df38ab72edc104d`. It used 3,500 learner updates, 21,353
environment steps, and 1,208.5 seconds on ROCm. The run retained an atomic
manifest plus best and final checkpoints under
`experiments/2026-07-21_cartpole_online_target_seed0_3500/`.

The intervention makes learning earlier but does not make it stable. Evaluation
remained near `9.4` through update 1,200, then reached `362.95` at 1,700,
fell to `149.25` at 2,300, recovered to `441.05` at 2,700, fell to `156.35`
at 2,900, and reached a best of `493.30` at 3,100. It immediately violated
the post-solve floor with `284.80` at 3,200 and ended at `163.20`; the
best-to-final gap is `330.10`. The behavioral gate therefore fails, and seeds
1 and 2 are stopped by the preregistered rule.

This rejects stale slow targets as a sufficient cause of collapse. It does not
reverse the source-conformance correction: online targets match the audited
reference and materially change learning dynamics. The already-preregistered
boundary diagnostics now compare best and final on-policy behavior and run the
symmetric fixed-history probe from both source policies, using episodes/seeds
17--36, real horizon 30, and model horizon 3. Their purpose is to determine
whether online targets improved recovery-state value ordering despite failing
the behavioral gate; they cannot rescue the canary's disposition.

The probes completed and retain summaries plus per-state rows under
`experiments/2026-07-21_cartpole_online_target_on_policy_probe/`,
`experiments/2026-07-21_cartpole_online_target_fixed_history_best_source/`, and
`experiments/2026-07-21_cartpole_online_target_fixed_history_final_source/`.
The on-policy best checkpoint solves all 20 episodes at return `500.0`; the
final checkpoint averages `183.2` and solves none.

| Fixed history source | Target | Q/real balanced | Q/real corr. | Actor/Q | Actor/real balanced | Posterior x MSE | One-step prior x MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Best, return 500.0 | Best 3,100 | 0.515 | 0.132 | 0.951 | 0.533 | 0.139 | 0.104 |
| Best, return 500.0 | Final 3,500 | 0.557 | 0.061 | 0.935 | 0.582 | 0.155 | 0.175 |
| Final, return 183.2 | Best 3,100 | 0.332 | -0.049 | 0.962 | 0.334 | 1.702 | 0.689 |
| Final, return 183.2 | Final 3,500 | 0.433 | 0.065 | 0.865 | 0.482 | 0.128 | 0.063 |

The final checkpoint therefore improves rather than forgets action ordering on
the solved checkpoint's fixed histories. Its own histories are harder: mean
absolute cart position is `0.637` instead of `0.538`, and `28.8%` of states
have `|x| >= 1` instead of `20.0%`. The best checkpoint represents those
histories poorly and ranks their actions worse; the final checkpoint adapts its
representation but still ranks real action outcomes below chance. Its actor
continues to follow that ranking on `86.5%` of actionable states.

The boundary gate fails. Relative to the slow-target canary on final-policy
histories, online targets change Q/real correlation from `-0.072` to `0.065`
but reduce balanced accuracy from `0.521` to `0.433`; actor/Q agreement remains
above its `0.8` floor. The isolated stale-target causal hypothesis is rejected:
online targets improve speed and one aspect of ordering but do not supply a
robust recovery value landscape. No replication seeds or combined corrective
changes are authorized by this result. The evidence-selected next step is to
finish the remaining objective/optimizer source audit and preregister one
conformance intervention only if it exposes another concrete divergence that
can explain recovery-state value errors.

#### Preregistered value-head representability comparison

The remaining source audit confirms that official `main` is still reference
commit `e3f0224`. Its value head has three `d_hidden`-wide hidden layers, each
followed by RMSNorm and SiLU. The local actor and value networks still share an
original two-hidden-layer SiLU MLP with no normalization; this omission predates
and survived the later repository-wide RMSNorm correction. The mismatch is
directly adjacent to the measured latent-to-value ordering failure, but a
source difference alone does not justify another online run.

- **Question:** on identical frozen posterior latents and trusted value labels,
  does the reference-style value head generalize materially better than the
  deployed local head?
- **Frozen source:** the online-target final checkpoint from run
  `d9c14b1ba1d74173aaa85a7d8531f05c`. Collect 4,096 states with random behavior
  using seed 17, split whole episodes 80/20, and compute exact discounted
  remaining return under constant action 0 and constant action 1 separately.
- **Causal variable:** compare the current two-hidden-layer SiLU critic with a
  three-hidden-layer RMSNorm/SiLU critic. Use the same frozen encoder/RSSM,
  two-hot bins, dataset/split, zero output initialization, AdamW `1e-3`, batch
  128, 30 epochs, and random initialization seed for both heads.
- **Primary gate:** the reference head must improve the worse of the two
  posterior-latent held-out correlations by at least `0.10`, without increasing
  held-out MAE for either action by more than `5%`. True-state results are a
  control, not the selection metric.
- **Stop rule:** this is a read-only paired comparison, not a trainer change. A
  failed gate rejects value-head representability as the next intervention. A
  pass authorizes only an isolated, backward-compatible value-head conformance
  canary; it does not authorize actor, reward-head, optimizer, ramp, or replay-
  gradient changes.

The paired comparison completed from clean probe commit `43a6ed6`; summaries
are under `experiments/2026-07-22_cartpole_value_head_comparison/`. Both target
policies used the same 4,096 states from 184 episodes and the same 897-state
held-out episode split.

| Target | Head | Latent test corr. | Latent test MAE | True-state test corr. | True-state test MAE |
|---|---|---:|---:|---:|---:|
| Always action 0 | Local | 0.957 | 0.502 | 0.944 | 0.689 |
| Always action 0 | Reference | 0.947 | 0.481 | 0.978 | 0.327 |
| Always action 1 | Local | 0.908 | 0.567 | 0.947 | 0.629 |
| Always action 1 | Reference | 0.913 | 0.541 | 0.966 | 0.372 |

The primary gate fails: the worse latent correlation improves by only `0.004`,
not `0.10`, while action-0 correlation decreases. The reference head is much
better on true physical state and slightly lowers latent MAE, so it remains a
reasonable replication cleanup; it does not remove the deployed latent-value
representability bottleneck measured here. No value-head training canary is
authorized.

#### Preregistered truncation-bootstrap correction

Continuing the replay-target audit exposed a concrete semantic bug. Collector
rows correctly preserve `is_last = terminated or truncated` separately from
`is_terminal = terminated`. The replay lambda-return path then multiplies
these flags into one continuation before constructing targets. Consequently a
time-limit transition receives `reward` only, while the reference recurrence
uses `reward + discount * bootstrap`; both correctly omit the bootstrap for a
true physical terminal. This distinction matters specifically after CartPole
starts reaching its 500-step time limit, although only sampled windows that
contain the episode boundary receive the incorrect target.

- **Hypothesis:** terminalizing CartPole's successful time-limit transitions
  injects a solved-policy-specific low-value target into replay and contributes
  to training away from solved behavior. Preserving the reference truncation
  bootstrap will improve post-solve stability and final-policy action ordering.
- **Causal variable:** change only replay lambda-return construction to accept
  separate `is_last` and `is_terminal` tensors. Use `is_terminal` for the live
  discount and `is_last` for the lambda recursion, exactly matching reference
  commit `e3f0224`. Keep imagined returns, continuation training, pair masks,
  online critic targets, model architecture, replay scale, optimizer settings,
  and all benchmark hyperparameters unchanged.
- **Mechanical gate:** focused recurrence tests must show that truncation
  targets `reward + discount * bootstrap`, true termination targets `reward`,
  ordinary lambda recursion is unchanged, and padded/cross-episode rows remain
  masked. Full fast validation and a one-update process smoke must pass.
- **Frozen run:** repeat the online-target seed-0 3,500-update command and fixed
  20-episode evaluation cohort from the preceding canary.
- **Behavioral gate:** reach mean return `450`; never fall below `300`
  afterward; final at least `400`; best-to-final gap at most `100`.
- **Boundary gate:** rerun the final-policy fixed-history comparison. Final
  Q/real balanced accuracy must exceed the current `0.433`, Q/real correlation
  must remain positive, and actor/Q agreement must remain above `0.8`.
- **Stop rule:** one seed only. Failure rejects the truncation bootstrap as a
  sufficient cause and does not authorize stacking value-head, optimizer-ramp,
  or representation-gradient changes.

#### Truncation-bootstrap canary result

The frozen seed-0 canary completed normally at source commit `2207409` with
manifest run ID `6237f54a2abe4ca3b4cce092ef7eca28` and MLflow run ID
`a1196f2b69aa489197f6b29a508c1131`. It used 3,500 learner updates, 21,011
environment steps, and 1,118.8 seconds on ROCm. The collector loaded every
published revision observed during monitoring, the parent exited successfully,
and periodic, best, and final checkpoints were retained under
`experiments/2026-07-22_cartpole_truncation_bootstrap_seed0_3500/`.

The behavior gate fails. Evaluation stayed at the `9.4` floor through update
1,500, jumped to `493.05` at 1,600, and immediately fell below the required
post-solve floor: `218.00` at 1,700, `154.60` at 1,800, and `136.90` at 1,900.
It subsequently recovered to `453.45` at 2,300 and `483.55` at 2,700, then
fell to `163.95` at 2,800. It ended at `237.15`; the best-to-final gap is
`255.90`. These repeated large changes in a fixed 20-episode evaluation cohort
show that the run can rediscover near-solved behavior but cannot retain it.

The preregistered fixed-history evidence is retained under
`experiments/2026-07-22_cartpole_truncation_bootstrap_fixed_history_final_source/`.
The final policy averages return `226.85` on seeds 17--36 and supplies 2,451
actionable rows from 4,537 fixed states.

| Final-policy histories | Target | Q/real balanced | Q/real corr. | Actor/Q | Actor/real balanced | Posterior x MSE | One-step prior x MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Return 226.85 | Best 1,600 | 0.227 | -0.232 | 0.828 | 0.109 | 1.075 | 1.107 |
| Return 226.85 | Final 3,500 | 0.293 | -0.264 | 0.849 | 0.481 | 0.330 | 0.286 |

The boundary gate also fails: final Q/real balanced accuracy is `0.293`, below
the required `0.433`; correlation is negative rather than positive; only the
actor/Q agreement floor passes. The best checkpoint performs even worse on the
final policy's histories and represents their cart position poorly, while the
final checkpoint adapts its representation but still learns an inverted or
otherwise unreliable action ordering. The first actionable broken boundary
therefore remains recovery-state value/target quality, with the actor mostly
following the critic it is given.

This rejects missing truncation bootstrap as a sufficient cause of instability,
while retaining the correction as required replay semantics. Only windows that
contain a 500-step boundary receive the corrected target, so its sparse causal
reach is a plausible limitation. Seeds 1 and 2 and all combined interventions
are stopped by the preregistered rule. The next step is another bounded source
audit focused on value-target construction and gradient ownership; a further
training change requires a concrete divergence that can affect the much broader
recovery distribution.

#### Preregistered symmetric two-hot conformance canary

The continued value-path audit returns to an early observation that was never
corrected. CartPole overrides the global two-hot support from symmetric
symlog-space endpoints `[-20, 20]` to asymmetric endpoints `[-5, 6]`. After
`symexp`, those 255 bins span approximately `[-147.41, 402.43]` and their
uniform mean is `+23.505`. The value output is zero-initialized, so its first
zero-logit prediction is therefore `+23.505`, not the intended zero. This exact
value appeared in the first logged imagined critic distributions of the initial
three-seed investigation. The reward layer has a second cold-start defect: its
weight is zeroed but its PyTorch bias is left at random initialization, despite
the source comment claiming zero reward initialization. The result is a state-
independent but nonuniform initial reward distribution.

Reference commit `e3f0224` instead uses symmetric `[-20, 20]` symlog endpoints
and pairs negative and positive terms when computing the expectation. That
pairwise summation is deliberate: a naive float32 reduction over the very wide
physical bins suffers cancellation error even when probabilities are uniform.

- **Hypothesis:** the positively biased cold reward/value distributions give
  early imagination a fictitious large return and seed unstable critic/policy
  optimization. Restoring the symmetric, cancellation-safe two-hot contract
  will start learned reward and value at exactly zero and improve later action-
  ordering stability.
- **Causal variable:** centralize two-hot expectation decoding using the
  reference's symmetric pairwise reduction, remove only CartPole's `[-5, 6]`
  support override so new runs inherit `[-20, 20]`, and zero the reward output
  bias alongside its already-zero weight. These are the three parts of the
  reference's zero-at-initialization distribution contract. Keep network depth,
  optimizer rates and ownership, replay losses, online targets, truncation
  bootstrap, actor objective, continuation head, data pacing, and every other
  benchmark setting unchanged. Historical checkpoints remain loadable because
  their config snapshots retain their authored endpoints.
- **Mechanical gate:** initialized reward and value logits over the new support
  must decode to exactly zero; the decoder must preserve ordinary weighted
  expectations and gradients; all direct expectation sites must share it.
  Focused tests, the full fast suite, compile/type gates, and a one-update
  process smoke must pass.
- **Frozen run:** repeat the same seed-0, 3,500-update command and fixed
  20-episode evaluation cohort used by the preceding two canaries. Confirm the
  resolved config uses endpoints `[-20, 20]` and the first logged dreamed reward
  and value predictions no longer begin near `+23.505`.
- **Behavioral gate:** reach mean return `450`; never fall below `300`
  afterward; final at least `400`; best-to-final gap at most `100`.
- **Boundary gate:** rerun the final-policy fixed-history comparison. Final
  Q/real balanced accuracy must exceed the truncation canary's `0.293`, Q/real
  correlation must be positive, and actor/Q agreement must remain above `0.8`.
- **Stop rule:** one seed only. Failure rejects the cold two-hot bias as a
  sufficient cause; it does not authorize stacking optimizer warmup, joint
  ownership, value-head depth, or representation gradients.

The first launch attempt is retained as interrupted operational evidence under
`experiments/2026-07-22_cartpole_symmetric_twohot_seed0_3500/`, with manifest
run ID `206f4d5a6487409e847e119a150fd5d0` and MLflow run ID
`e56179f7872b46eb849e9a51dcc6fff2`. Runtime telemetry confirmed the intended
mechanism at update 0: imagined reward and critic-value means were both exactly
`0.0`, rather than the historical cold critic's `+23.505`. The run was manually
stopped near update 220 without a learning disposition because throughput was
only `0.98` updates/second, versus `3.57` for the matched predecessor at update
200. The learner was collection-gated: frame/update ratios were matched and the
host simultaneously had a separate VLA study issuing heavy CPU/filesystem work
against an NTFS volume. No Dreamer code or process fault occurred. A clean
retry of the same seed and frozen configuration remains the one authorized
research canary; the interrupted attempt is not a second statistical seed.

The first retry is likewise retained, not interpreted, under
`experiments/2026-07-22_cartpole_symmetric_twohot_seed0_3500_retry1/`, with
manifest run ID `0d0ad13580bc426b82e71fa7a0412456` and MLflow run ID
`4cc53cd9fbd943179ce05fae728bb1e2`. It reached update 500 at the expected
`9.35` pre-learning floor. Throughput was `3.61` updates/second while the host
was free, then fell to `1.00` after the external VLA supervisor launched its
next worker; Dreamer and VLA together saturated all 20 logical CPUs. The run
was stopped to avoid perturbing the VLA study. The final clean retry will cap
OpenMP, MKL, and OpenBLAS to one thread per Dreamer process; this is an
operational resource limit, not an authored model, data, optimizer, or RNG
change. It must first demonstrate normal bounded CPU use and adequate
throughput before consuming the full canary budget.

#### Symmetric two-hot canary result

The resource-limited clean retry completed normally from source commit
`7ba5282` with manifest run ID `3601530f568b48a286ea3be7f4672595`
and MLflow run ID `92a6db1fe8da4fa7957de4d4d81ee15c`. It used 3,500
learner updates, 21,543 environment steps, and 1,026.4 seconds on ROCm. OpenMP,
MKL, and OpenBLAS were limited to one thread per Dreamer process; observed
Dreamer CPU use stayed near one trainer core plus a small collector fraction
while the unrelated VLA study continued. The run completed at
`max_train_steps`, saved best/final/periodic checkpoints, and stopped the
collector normally under
`experiments/2026-07-22_cartpole_symmetric_twohot_seed0_3500_retry2/`.

The mechanical mechanism passes. The resolved support is `[-20, 20]`, and the
first logged imagined reward and critic-value means are both exactly `0.0`.
Reward reaches approximately `1.0` by update 50, so zero initialization does
not prevent the supervised reward head from learning its CartPole target.

The behavioral gate fails, but with substantially different dynamics. Return
stayed at `9.4` through update 1,600, reached `142.0` at 1,700 and `323.65` at
1,800, then oscillated between `147.2` and `337.75` through update 2,800. It
late-solved at `500.0` for updates 2,900 and 3,000 and remained high at `484.5`
at 3,100. It then collapsed to `235.2` at 3,200 and `217.1` at 3,300, partly
recovered to `437.3`, and ended at `324.6`. The post-solve floor is below
`300`, the final is below `400`, and the best-to-final gap is `175.4`.

The preregistered fixed-history evidence is retained under
`experiments/2026-07-22_cartpole_symmetric_twohot_fixed_history_final_source/`.
The final policy averages return `332.3` on seeds 17--36; its trajectories have
mean absolute cart position `0.949`, with `46.0%` of states at `|x| >= 1`, and
supply 4,431 actionable rows from 6,646 fixed states.

| Final-policy histories | Target | Q/real balanced | Q/real corr. | Actor/Q | Actor/real balanced | Posterior x MSE | One-step prior x MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Return 332.3 | Best 2,900 | 0.347 | -0.076 | 0.875 | 0.313 | 0.682 | 0.696 |
| Return 332.3 | Final 3,500 | 0.559 | 0.009 | 0.816 | 0.506 | 0.451 | 0.405 |

The boundary gate passes all literal thresholds: final Q/real balanced accuracy
exceeds the preceding `0.293`, correlation is positive, and actor/Q agreement
remains above `0.8`. The margin is not uniformly reassuring: correlation is
only `0.009`, and the final policy visits the broadest recovery distribution of
these canaries. Still, the final checkpoint improves both representation error
and action ordering over its solved best checkpoint on those identical
histories. It has not simply forgotten the solved controller's local value
landscape; it adapts after the policy moves into harder states, too late to
prevent the closed-loop failure.

This rejects cold two-hot bias as a sufficient cause of instability while
retaining the correction as required source semantics and a measurable boundary
improvement. It also suggests that the historical `+23.505` prior supplied
accidental optimism: removing it delays useful behavior by roughly 1,200
updates, but ultimately produces a cleaner late value ranking. Seeds 1 and 2
are stopped because the behavioral gate dominates. The next bounded source
audit should focus on imagined-loss weighting and return-normalization update
scope, which directly control policy-improvement step scale while leaving the
now-improved distributional support intact. No optimizer or architecture
change is selected from this result alone.

#### Preregistered return-normalizer conformance canary

The next source audit found that the frozen canary already disables the local
per-imagination advantage z-score, so its actor update is directly controlled
by the running return-percentile span. That span contracts from `33.72` at
update 2,500 to `27.63` at 3,000 and `16.41` at 3,500. The policy first reaches
`500` during this interval and subsequently collapses; holding the raw modeled
advantage fixed, the contraction makes its final policy-gradient coefficient
about `1.78` times the update-3,000 coefficient. This is correlation, not yet a
causal conclusion.

There are three concrete differences from reference commit `e3f0224` in the
same normalizer component. Reference `imag_loss` computes one return tensor for
all `B * K` replay starts, updates its 5th/95th-percentile EMA from that complete
tensor, and immediately uses the updated span in the current policy loss. Its
`debias: false` state begins at `(lo, hi) = (0, 0)`, so the first observation is
incorporated at the authored rate `0.01`. Local training instead updates after
the forward pass from only the final valid replay start, uses the stale scale
for the current actor loss, and special-cases the first observation by assigning
its percentiles in full. The local `normalize_advantages=false` override means
these differences are not canceled by a later z-score.

- **Hypothesis:** the partial, first-batch-debiased, one-update-stale return
  normalizer makes the actor's effective step scale track one replay position
  too aggressively and contributes to post-solve oscillation. Restoring the
  reference normalizer state and update scope will reduce best-to-final collapse.
- **Causal variable:** initialize return percentiles at zero, always apply the
  rate-`0.01` EMA without first-update debiasing, aggregate lambda returns from
  every valid post-burn-in imagination start, and use that updated span for the
  current REINFORCE actor loss. Keep `normalize_advantages=false`, actor/critic
  objectives and learning rates, imagination weights, online value targets,
  symmetric two-hot support, replay pacing and sampling, and every architecture
  setting unchanged.
- **Mechanical gate:** unit tests must prove zero-state first-update behavior,
  aggregation across multiple replay starts, and current-batch actor scaling.
  Checkpoint save/resume must retain the new scalar state; focused/full tests,
  compile/type checks, and a one-update process smoke must pass.
- **Frozen run:** repeat seed 0 for 3,500 learner updates with the exact symmetric
  two-hot retry-2 overrides and fixed 20-episode evaluation cohort. Retain the
  one-thread OpenMP/MKL/OpenBLAS cap while the unrelated VLA job shares the host.
- **Behavioral gate:** reach mean return `450`; never fall below `300`
  afterward; final at least `400`; best-to-final gap at most `100`.
- **Boundary gate:** rerun the final-policy fixed-history comparison. Final
  Q/real balanced accuracy must exceed `0.559`, Q/real correlation must exceed
  `0.009`, actor/real balanced accuracy must exceed `0.506`, and actor/Q
  agreement must remain above `0.8`.
- **Stop rule:** one seed only. Failure rejects return-normalizer mismatch as a
  sufficient cause and does not authorize combining the separate imagination-
  weighting discrepancy, optimizer changes, or architecture changes.

#### Return-normalizer canary result

The mechanical gate passes at source commit `26699b6`. Focused normalization,
actor-loss, checkpoint, and configuration tests passed (`27`), the full fast
suite passed (`136`), compile and the supported scoped type gate passed, and the
one-update CPU process smoke exited normally with its collector stopped. The
implementation retains checkpoint compatibility while new runs start the
non-debiased percentile state at zero. It aggregates every valid post-burn-in
return batch and applies the resulting current-batch scale to REINFORCE; the
authored default now also matches reference `advnorm: none`.

The frozen seed-0 run completed normally under
`experiments/2026-07-22_090738_CartPole-v1/`, with manifest run ID
`17406d7e69364583afec79bba34af908` and MLflow run ID
`b07fa34cf414412fb2fe60a283b673ef`. It used 3,500 learner updates, 21,517
environment steps, and 927.4 seconds on ROCm. Its manifest records clean source
`26699b6`, `max_train_steps`, final/best/periodic checkpoints, and successful
completion. The one-thread OpenMP/MKL/OpenBLAS cap preserved approximately
`3.6`--`4.2` updates/second while sharing the host.

The intended scale path is measurably different from the preceding final-start-
only implementation:

| Update | 500 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 3,500 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Corrected span | 4.74 | 7.89 | 14.22 | 14.79 | 19.08 | 26.22 | 16.56 |
| Pre-fix span | 4.22 | 3.49 | 6.72 | 19.25 | 33.72 | 27.63 | 16.41 |

The behavioral gate fails decisively. Evaluation leaves the floor at update
1,400 (`215.70`), reaches `450.30` at 2,300, and immediately violates the
post-solve floor with `195.25` at 2,400. It recovers to the run best `477.35`
at 2,700, then falls through `292.55`, `262.65`, and `143.30` at updates
2,900--3,100 before ending at `154.15`. The best-to-final gap is `323.20`.
Notably, the deep late collapse begins while the span grows from `19.08` at
2,500 to `26.22` at 3,000, which *reduces* rather than amplifies the normalized
actor coefficient. The earlier correlation between a contracting span and
collapse is therefore not causal by itself.

The fixed-history evidence is retained under
`experiments/2026-07-22_cartpole_return_normalizer_fixed_history_final_source/`.
The final actor averages return `167.4` on source seeds 17--36 and supplies
2,003 actionable rows from 3,348 states.

| Final-policy histories | Target | Q/real balanced | Q/real corr. | Actor/Q | Actor/real balanced | Posterior x MSE | One-step prior x MSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Return 167.4 | Best 2,700 | 0.616 | 0.183 | 0.916 | 0.575 | 0.082 | 0.102 |
| Return 167.4 | Final 3,500 | 0.592 | 0.088 | 0.959 | 0.537 | 0.161 | 0.030 |

The literal boundary gate passes, but best-to-final movement is adverse: Q/real
accuracy, Q/real correlation, actor/real accuracy, and current-posterior cart-
position reconstruction all worsen, while actor/Q agreement rises. The final
actor is more faithfully following a critic whose recovery-state ranking is
less accurate than at the best checkpoint. Aggregate ranking remains above the
previous canary's thresholds, so this is degradation rather than complete
critic inversion; it still cannot support stable closed-loop control.

This rejects return-normalizer mismatch as a sufficient cause while retaining
the correction as source-conforming infrastructure. Seeds 1 and 2 are stopped.
The next isolated source audit is imagination weighting: reference policy/value
weights begin with the learned continuation of the replay start, whereas local
weights begin at exactly one and only use continuations from imagined successor
states. That difference gives terminal replay starts full actor/critic weight
instead of suppressing them. It must be quantified and preregistered separately
before any new run; no optimizer or architecture change is selected here.

#### Preregistered continuation-coverage canary

The weighting audit found that correcting the consumer alone cannot currently
provide the intended terminal mask. On deterministic seeds 17--36, the update-
500 head assigns physical terminal states mean continuation probability `0.992`.
On histories fixed by the collapsed final actor, the best update-2,700 head
assigns terminals `0.441` mean probability with `0.65` recall below probability
`0.5`; the final head assigns those identical terminals `0.986`, has zero
terminal recall, and has balanced terminal/live accuracy `0.500`. Its live mean
is `0.958`. The final continuation head is therefore an always-continue
classifier, not a usable source of the missing start weight.

The producer-side cause is a replay edge bias. Reference commit `e3f0224`
inserts a sliding sequence at every environment step and permits sequences to
span episode resets, whose `is_first` flags reset recurrent state. Local replay
stores each episode separately and samples only fully contained length-`T`
windows (except one padded window for episodes shorter than `T`). For an episode
of length `L >= T`, an interior transition occurs in as many as `T` sampled
windows, but the physical terminal transition occurs only in the one window
whose end equals the episode end. With the frozen `T=16`, terminal continuation
labels can therefore receive about one-sixteenth the inclusion multiplicity of
interior live labels exactly as learned episodes grow longer. This compounds
the natural one-terminal-per-episode imbalance and matches the observed drift:
the head partially recognizes failures at the best checkpoint, then forgets
them while recent episode length increases.

- **Hypothesis:** per-episode full-window sampling starves the continuation head
  of failure labels, allowing it to become overoptimistically constant and to
  inflate imagined survival/value near the cart boundary. Correcting only the
  continuation loss for each row's window-inclusion probability will preserve
  terminal discrimination and reduce post-solve collapse.
- **Causal variable:** have replay return a continuation importance weight for
  every sampled row. For a full episode, compute each transition's exact number
  of valid length-`T` windows and weight it by the episode's mean inclusion
  multiplicity divided by its own multiplicity; padded short episodes retain
  unit weights. Apply these weights only to continuation BCE. This makes all
  transitions within an episode contribute equally in expectation while
  preserving mean continuation-loss scale. Do not change sequence selection,
  reward/reconstruction/KL losses, actor/critic losses, the currently missing
  start-continuation multiplier, or any run hyperparameter.
- **Mechanical gate:** enumerating every valid window of synthetic episodes
  must show equal aggregate continuation weight for every transition and mean
  sampled-row weight one; terminal weighting must affect only continuation BCE.
  Padded masks and exact-return fields must remain aligned. Focused/full tests,
  compile/type gates, and a one-update process smoke must pass.
- **Frozen run:** repeat the return-normalizer seed-0 run for 3,500 learner
  updates with its exact configuration, evaluation cohort, and one-thread host
  resource cap.
- **Behavioral gate:** reach mean return `450`; never fall below `300`
  afterward; final at least `400`; best-to-final gap at most `100`.
- **Continuation gate:** on deterministic seeds 17--36, the final posterior
  continuation probability on physical terminal transitions must be below
  `0.5`, terminal recall must exceed `0.5`, and terminal/live balanced accuracy
  must exceed `0.7`. The current matched-history baselines are `0.986`, `0.0`,
  and `0.500`, respectively.
- **Boundary gate:** if the behavior gate passes, rerun the final-policy fixed-
  history Q comparison; final Q/real accuracy and correlation must not regress
  below `0.592` and `0.088`, and actor/Q agreement must remain above `0.8`.
- **Stop rule:** validate and run seed 0 only. Failure rejects terminal-label
  coverage as sufficient and does not authorize stacking the deferred start-
  continuation multiplier, loss-class balancing, or architecture changes.

#### Continuation-coverage canary result

The mechanical gate passes at source commit `f29f7a8`. Exact enumeration tests
show equal aggregate continuation weight for every transition in a fully
sampled episode and mean sampled-row weight one; short padded episodes retain
unit weight on real rows and zero weight on padding. Loss-isolation tests show
that the importance tensor changes continuation BCE only. The focused suite
passed (`24`), the full fast suite passed (`138`), compile and the supported
scoped type gate passed, and the one-update CPU process smoke exited normally
with its collector stopped.

The frozen seed-0 run completed normally under
`experiments/2026-07-22_093847_CartPole-v1/`, with manifest run ID
`f7dcbf8531f34a949b690604bccc41d2` and MLflow run ID
`72f1939bbecb48edb97cb05298da7f8f`. It used 3,500 learner updates, 21,357
environment steps, and 947.6 seconds on ROCm. Its manifest records clean source
`f29f7a8`, `max_train_steps`, final/best/periodic checkpoints, and successful
collector shutdown. The one-thread OpenMP/MKL/OpenBLAS cap remained in force.

The behavioral gate fails strictly, although this is the strongest sustained
late behavior on the current corrected stack. Evaluation first reaches `463.5`
at update 1,900 and immediately violates the post-solve floor at `207.3`. It
later falls to `169.45` at update 2,700, then holds a six-evaluation near-solved
plateau from updates 2,800--3,300: `488.4`, `500.0`, `499.1`, `461.95`,
`470.35`, and `473.95`. It ends at `396.65` after `331.8` at update 3,400.
Consequently, final return misses `400` by `3.35` and the best-to-final gap is
`103.35`, also missing its limit by `3.35`. Compared with the preceding
return-normalizer canary, first solve is 400 updates earlier and final return is
242.5 points higher, so the intervention has useful behavioral evidence even
though it does not pass the frozen criterion.

The continuation gate fails decisively. The deterministic seed-17--36 probe is
retained under
`experiments/2026-07-22_cartpole_continuation_coverage_probe/`. The best
checkpoint solves all 20 episodes to the 500-step time limit, so it contains no
physical terminal transitions on which to measure failure recall. The final
checkpoint averages return `377.25` and supplies 20 physical failures across
7,545 transitions. Its posterior expected effective discount is `0.98574` on
terminal rows versus `0.98702` on live rows. Dividing by configured discount
`0.997`, this corresponds to mean terminal continuation probability about
`0.9887`, not below `0.5`. Every terminal remains classified as continue, so
terminal recall is `0.0`, balanced terminal/live accuracy is `0.500`, and
failure ROC AUC is only `0.554`. The one-step prior is similarly
non-discriminative (`0.98530` terminal effective discount; AUC `0.536`).

This isolates two conclusions. Correcting replay inclusion multiplicity is a
real and materially beneficial training correction, but it is not sufficient
to make the learned continuation head recognize CartPole failures. The late
near-solved plateau can coexist with an always-continue model because most
successful CartPole rollouts end by the 500-step time limit, while rare
physical failures remain a tiny binary-loss minority even after removing the
extra window-position bias. Seeds 1 and 2 are stopped, and the conditional
fixed-history Q gate is not run because the behavior gate did not pass. Per the
preregistered stop rule, this result does not authorize stacking the deferred
start-continuation multiplier or class balancing. The next intervention must
come from a fresh source/gradient audit rather than treating the near-pass as a
success.

#### Preregistered sequence-loss reduction canary

The next gradient-path audit found that local losses are reduced differently
for reporting and optimization. Each per-row world-model loss already averages
over the batch and is accumulated over all `T` replay rows. Each imagined
actor/critic loss averages over batch and imagination horizon and is accumulated
over the `K = T - burn_in` replay starts. `core.py` backpropagates those sums
directly. Only the later logger divides by `K`, so the dashboard displays an
apparent mean that the optimizer never receives. The replay-auxiliary helper
and its test explicitly state that the caller will divide the accumulated loss,
but no such caller exists.

Reference commit `e3f0224` takes the mean over batch, replay starts, and
imagination time before its single optimizer. Under the frozen `T=16`,
`burn_in=4` contract, local world-model gradients are therefore scaled by 16
and actor/critic gradients by 12 relative to the authored reduction. This does
not imply 16x or 12x parameter updates: LaProp's per-coordinate RMS transform
is invariant to a fixed positive gradient multiplier up to epsilon. The
remaining direct mechanism is AGC, which runs *before* LaProp. Multiplication
can move a parameter tensor across AGC's `0.3 * ||parameter||` threshold and
thereby change its clipped gradient history; the hidden reduction also makes
that behavior depend on sequence length.

- **Hypothesis:** backpropagating sequence sums causes avoidable, sequence-
  length-dependent AGC clipping and contributes to target/policy oscillation.
  Applying the authored means before clipping will reduce late CartPole
  collapse.
- **Causal variable:** divide the accumulated world-model objective by all `T`
  replay rows and the actor and critic objectives by the `K` post-burn-in
  imagination starts before backward. Preserve replay auxiliary coefficients
  at their authored values after this division, normalize the opt-in replay
  representation diagnostic consistently, and stop dividing the already-mean
  totals in scalar/progress logging. Do not change loss contents, replay
  sampling, continuation weights, learning rates, AGC factor, LaProp, model
  architecture, or benchmark configuration.
- **Mechanical gate:** focused tests must prove that the returned objectives
  and their gradients equal arithmetic sequence means, replay auxiliary scale
  remains `0.3`, and logging does not divide them a second time. Focused/full
  tests, compile/type gates, and a one-update process smoke must pass.
- **Frozen run:** repeat seed 0 for 3,500 updates with the exact continuation-
  coverage configuration, fixed evaluation cohort, and one-thread host
  resource cap.
- **Behavioral gate:** reach mean return `450`; never fall below `300`
  afterward; final at least `400`; best-to-final gap at most `100`.
- **Boundary gate:** only if behavior passes, rerun the final-policy fixed-
  history comparison. Final Q/real accuracy and correlation must not regress
  below `0.592` and `0.088`, and actor/Q agreement must remain above `0.8`.
- **Stop rule:** one seed only. Failure rejects sequence-sum reduction as a
  sufficient cause and does not authorize stacking AGC changes, learning-rate
  warmup, optimizer unification, or continuation class balancing.

#### Sequence-loss reduction canary result

The mechanical gate passes at source commit `216cf37`. The focused loss,
gradient, replay-auxiliary, return-normalizer, replay, and logging tests passed
(`19`); the full fast suite passed (`143`); compile and the supported scoped
type gate passed; and the one-update CPU process smoke exited normally with its
collector stopped. The forward result now contains actual mean objectives,
replay representation diagnostics use the same reduction, and scalar/progress
logging no longer applies a second division.

The frozen seed-0 run completed normally under
`experiments/2026-07-22_cartpole_sequence_loss_seed0_3500/`, with manifest run
ID `f2a2e4b3b9424047a2859db33c72b65c` and MLflow run ID
`4c8a47f4c0694d959bd4e2f5bb9a48d6`. It used 3,500 learner updates, 21,370
environment steps, and 920.3 seconds on ROCm. The manifest records clean source
`216cf37`, `max_train_steps`, final/best/periodic checkpoints, and successful
collector shutdown under the one-thread host resource cap.

The reduction materially changes optimization but fails the behavioral gate.
The policy remains near the 9.35 floor through update 2,000, begins moving at
2,200 (`130.4`), and oscillates through `75.2`, `340.15`, and `284.9` before
first solving at update 2,600 (`483.0`). Its complete post-solve sequence is
`483.0`, `490.75`, `369.25`, `438.55`, `457.9`, `437.95`, `301.0`, `408.45`,
`297.8`, and `310.85`. Update 3,400 therefore violates the strict floor by
2.2 points; final return is 89.15 below 400, and the best-to-final gap is
179.90 rather than at most 100.

Compared with the preceding continuation-coverage run, first solve is delayed
by 700 updates, final return falls from `396.65` to `310.85`, and the six-point
near-solved late plateau is not reproduced. The longer 2,600--3,300 interval
above the floor shows that mean reduction is not equivalent to the old summed
path under pre-LaProp AGC, but its eventual drop shows that sequence-dependent
clipping was not the sufficient instability mechanism. The source-conforming
reduction is retained so sequence length no longer silently changes the
authored objective or makes logs disagree with backward. Seeds 1 and 2 are
stopped, and the conditional fixed-history Q probe is not run because behavior
failed. Any later learning-rate or optimizer experiment requires a fresh
contract calibrated to the corrected mean-loss path; it cannot be smuggled into
this result as compensation.

#### Preregistered continuation-representability probe

The continuation-coverage and sequence-mean canaries leave a narrower question
before any class-balancing intervention: does the frozen observation-conditioned
latent expose physical failure well enough for the existing continuation-head
architecture to classify it? Low original-head AUC could reflect either missing
state information or online fitting dominated by live labels.

- **Question:** can a freshly fitted copy of the authored one-hidden-layer
  continuation MLP discriminate physical terminal rows from live rows when the
  encoder/RSSM features are frozen and the fitting loss is balanced?
- **Frozen source and data:** use the sequence-loss final checkpoint at update
  3,500. Run its deterministic actor for 100 episodes on reset seeds 17--116.
  Record the posterior-mode joined latent and raw physical state after every
  real action, with the resulting `terminated` label. Split whole episodes
  80/20, so neighboring transitions never cross train/test.
- **Paired fits:** train (a) the same continuation MLP on frozen posterior
  latents and (b) an equal-width MLP on true physical state as a control. Use
  one fixed initialization seed, AdamW `1e-3`, 20 epochs, batch 256, and
  `pos_weight = live_train / terminal_train`. Do not update the checkpoint or
  tune the cutoff. Also score the checkpoint's original head on the identical
  held-out rows.
- **Gate:** the true-state control must reach held-out failure AUC at least
  `0.95` and balanced accuracy at least `0.90`. If the fresh latent head reaches
  AUC at least `0.90` and balanced accuracy at least `0.80`, terminal state is
  representable and the next audit may focus on online imbalance/optimization.
  If the control passes but latent fails, representation is the nearer
  boundary. A failed control makes the probe inconclusive.
- **Stop rule:** this is one read-only checkpoint, one episode cohort, split,
  seed, and fit configuration. Do not train the end-to-end agent, alter loss
  weighting, or iterate the classifier settings after observing the result.

#### Continuation-representability result

The preregistered probe completed from clean diagnostic commit `5ea09c7` and
is retained under
`experiments/2026-07-22_cartpole_continuation_representability/`. Focused probe
tests passed (`5`), its direct type check and bytecode compilation passed, and
the frozen checkpoint was not modified. The 100 deployed deterministic
episodes average return `308.42`; none truncate, so they supply exactly 100
physical terminal rows. The episode-level split contains 24,724 training rows
and 80 terminals, a live-to-terminal ratio of `308.05`, plus 6,118 held-out
rows and 20 terminals.

| Held-out input/head | Failure AUC | Balanced accuracy | Terminal recall | Terminal failure probability | Live failure probability |
|---|---:|---:|---:|---:|---:|
| Original posterior-latent head | 0.944 | 0.500 | 0.000 | 0.172 | 0.053 |
| Fresh balanced posterior-latent head | 0.973 | 0.935 | 1.000 | 0.944 | 0.124 |
| Fresh balanced true-state control | 0.998 | 0.989 | 1.000 | 0.973 | 0.023 |

Both preregistered gates pass. The true-state control proves the cohort and fit
can express the physical failure rule. More importantly, the same authored
one-hidden-layer MLP fitted on the frozen joined posterior latent exceeds both
latent thresholds by wide margins. The checkpoint's original head already has
strong risk ordering, but all 20 terminal probabilities remain below the fixed
failure cutoff; it is miscalibrated toward the overwhelmingly common live
class rather than deprived of terminal information.

This classifies the next boundary as online continuation supervision/calibration,
not encoder/RSSM representability. A training intervention may now balance the
terminal and live contributions, but it must preserve the overall continuation-
loss scale and adapt to the changing failure prevalence rather than hard-code
the late-policy ratio `308.05` into early training. The fresh AdamW fit is a
capacity diagnostic, not evidence that the trainer should switch optimizers or
use its `1e-3` rate.

#### Preregistered balanced-continuation canary

The representability probe selects a class-calibration intervention, but its
late-policy weight `308.05` must not be hard-coded into startup. The replay
failure fraction changes by more than an order of magnitude as episodes grow,
and an unnormalized terminal multiplier would also change the total
world-model objective scale.

- **Hypothesis:** natural live/terminal imbalance drives the continuation head
  toward a well-ranked but always-continue solution. An online prevalence-
  balanced BCE will retain the latent's demonstrated failure discrimination,
  improve imagined survival calibration, and prevent the late policy/value
  collapse.
- **Causal variable:** maintain a non-debiased rate-`0.01` EMA of the sampled
  terminal fraction, computed with valid-row masks and the existing replay
  inclusion weights. Initialize prevalence at `0.5`. For continuation BCE only,
  multiply terminal rows by `0.5 / max(p, 0.001)` and live rows by
  `0.5 / max(1-p, 0.001)`, in addition to their inclusion weight. When the EMA
  matches prevalence, terminal and live classes each contribute half the loss
  while expected total weight remains one. Save/restore the EMA. Keep every
  other world-model term, sequence reduction, replay sample, actor/critic loss,
  optimizer, learning rate, and run hyperparameter unchanged.
- **Mechanical gate:** tests must establish the masked importance-weighted
  prevalence update, non-debiased first update, equal expected class mass,
  continuation-only loss effect, disabled-path identity, and checkpoint
  persistence. Focused/full tests, compile/type gates, and a one-update process
  smoke must pass.
- **Frozen run:** repeat seed 0 for 3,500 updates from the sequence-mean
  configuration, fixed evaluation cohort, and one-thread resource cap, changing
  only balanced continuation on.
- **Behavioral gate:** reach mean return `450`; never fall below `300`
  afterward; final at least `400`; best-to-final gap at most `100`.
- **Continuation gate:** on deterministic seeds 17--36, final posterior
  terminal continuation probability must be below `0.5`, terminal recall must
  exceed `0.5`, and terminal/live balanced accuracy must exceed `0.7`.
- **Boundary gate:** only if behavior passes, final fixed-history Q/real
  accuracy and correlation must not regress below `0.592` and `0.088`, and
  actor/Q agreement must remain above `0.8`.
- **Stop rule:** one seed only. Failure rejects adaptive class balancing as a
  sufficient cause and does not authorize learning-rate, optimizer, start-
  continuation, or architecture changes.

#### Balanced-continuation canary result

The mechanical gate passes at source commit `dcaf998`. The focused balance,
checkpoint, forward, and trainer tests passed (`37`); the full fast suite passed
(`150`); compile and the supported scoped type gate passed; and a balance-enabled
one-update CPU process smoke exited normally with its collector stopped. The
trainer now applies the adaptive weights only to continuation BCE, reports the
batch/EMA prevalence and class scales, and persists the EMA across checkpoints.
The option remains disabled by default.

The frozen seed-0 run completed normally under
`experiments/2026-07-22_cartpole_balanced_continuation_seed0_3500/`, with manifest
run ID `00d4d04df8d54214a672cfc3f948fb0f` and MLflow run ID
`a52e7c1334e3402ebdc39dde445d807d`. It used 3,500 learner updates, 21,392
environment steps, and 910.6 seconds on ROCm. The manifest records clean source
`dcaf998`, `max_train_steps`, final/best/periodic checkpoints, and successful
collector shutdown under the one-thread host resource cap.

The behavior gate fails decisively. Evaluation stays at the 9.35--9.40 floor
through update 1,400, rises to `120.65` at 1,600, and then mostly occupies a
subsolved 160--225 band. The best evaluation is `271.15` at update 3,400 and the
final evaluation is `159.45`, for a best-to-final gap of `111.70`. It never
reaches 450, so the conditional fixed-history Q probe is not run.

The preregistered deterministic continuation probe is retained under
`experiments/2026-07-22_cartpole_balanced_continuation_probe/`. At the final
checkpoint, seeds 17--36 average return `166.0` with no truncations. Posterior
terminal continuation probability is `0.221`, terminal recall at the 0.5 cutoff
is `0.95`, and balanced accuracy is `0.772`, so the narrow continuation gate
passes. However, only `59.4%` of 3,300 live rows are classified live. At the
best checkpoint, terminal probability is `0.118` and recall is `1.0`, but only
`30.5%` of 5,723 live rows are classified live and balanced accuracy is `0.653`.
The mean posterior continuation prediction is only `0.655` at final and `0.392`
at best, versus the live target `0.997`.

This rejects adaptive class-balanced BCE as a sufficient or semantically valid
continuation fix. It solves the rare-class cutoff problem by changing the
effective class prior: the raw sigmoid becomes a score calibrated to a
50/50-weighted training distribution, not the real probability of continuing.
Using that score directly as the imagination discount makes common live states
look likely to terminate and shortens imagined credit assignment. The result
therefore does not support enabling `balance_continuation` by default. Any
follow-up must preserve or explicitly recover the natural-prior probability
used for discounting; it cannot treat balanced classification accuracy alone as
calibration evidence. The behavior failure stops this seed and does not
authorize the Q probe or a broad sweep.

#### Preregistered reference-RSSM-core canary

The next source/history audit found a concrete architecture regression in the
recurrent dynamics path. Pinned reference commit `e3f0224` preprocesses the
deterministic state, flattened stochastic state, and action independently with
`Linear -> RMSNorm -> SiLU`; concatenates those features with each recurrent
block's deterministic state; applies one grouped hidden projection followed by
`RMSNorm -> SiLU`; and then uses one grouped projection for reset, candidate,
and update gates. The current local `step_dynamics` instead sends one linear
stochastic embedding plus the raw action through a conventional split-input
GRU. It omits deterministic/action preprocessing and the normalized grouped
hidden layer.

This discrepancy has explicit repository history. Commit `a559e55` attempted
to restore the three input preprocessors on 2026-02-28. It was removed from the
working file so a pre-change 15k Pong checkpoint could resume, as recorded in
`research_current.md`, and commit `9058140` then made that compatible file the
new source while nominally adding posterior unimix. No indexed exact-comparison
run evaluates the new recurrent architecture. The old patch was also only a
partial port: it retained separate conventional GRU input/hidden projections
and omitted the reference grouped hidden normalization. The intervention will
therefore implement the pinned equation rather than cherry-pick that patch.

- **Hypothesis:** the unnormalized legacy recurrent core makes the learned
  transition representation unnecessarily sensitive to changing on-policy
  state/action scales and contributes to the observed recovery-state model and
  value drift. The reference grouped core will improve late CartPole behavior
  stability.
- **Causal variable:** replace only the recurrent `_core` calculation for new
  authored runs with the pinned reference sequence: three independent
  `Linear -> RMSNorm -> SiLU` inputs, grouped hidden projection,
  `RMSNorm -> SiLU`, grouped three-gate projection, and the existing
  `sigmoid(reset)`, `tanh(reset * candidate)`, `sigmoid(update - 1)` update.
  Preserve stochastic size, deterministic size, block count, prior/posterior
  heads, encoder/decoder, losses, optimizers, rates, replay, and benchmark
  settings. Keep a legacy core mode for historical checkpoint construction;
  resume must restore the checkpoint's mode unless semantic migration is
  explicit.
- **Mechanical gate:** equation-level tests must compare the grouped PyTorch
  result with an explicit per-block calculation, establish output/gradient
  shapes and recurrent gradient flow across multiple observed rows, prove that
  authored Hydra selects the reference mode, and prove that historical
  snapshots/checkpoints retain the legacy parameter layout. Focused/full tests,
  compile/type gates, and a one-update process smoke must pass.
- **Frozen run:** repeat seed 0 for 3,500 updates under the exact unbalanced
  sequence-mean configuration and fixed 20-episode evaluation cohort, changing
  only the RSSM core. Retain the one-thread host resource cap.
- **Behavioral gate:** reach mean return `450`; never fall below `300`
  afterward; final at least `400`; best-to-final gap at most `100`.
- **Diagnostics:** regardless of behavior, run the deterministic final
  continuation audit on seeds 17--36. Only if behavior passes, run the
  final-policy fixed-history Q comparison, requiring Q/real accuracy at least
  `0.592`, correlation at least `0.088`, and actor/Q agreement above `0.8`.
- **Stop rule:** one seed only. Failure rejects reference-core conformance as a
  sufficient stability fix and does not authorize stacking posterior/encoder
  normalization, joint optimization, learning-rate warmup, or continuation
  weighting.

#### Reference-RSSM-core canary result

The mechanical gate passes at source commit `b52bf15`. The new equation-level
tests cover the legacy and reference parameter layouts, compare the grouped
reference calculation with an explicit block-by-block calculation, and verify
recurrent gradients. The focused suite passed (`32`, followed by `21` after the
last compatibility test), the full fast suite passed (`156`), compile and the
supported scoped type gate passed, and a canonical one-update CPU process smoke
completed with its collector stopped. New Hydra-authored runs select the
reference core, while historical snapshots and checkpoints retain or infer the
legacy core.

The frozen seed-0 run completed normally under
`experiments/2026-07-22_cartpole_reference_rssm_core_seed0_3500/`, with manifest
run ID `c3e29f0406aa46e68d2ef513f7d60721` and MLflow run ID
`78c5e66ed4214b11a3d7523a7e12b886`. It used 3,500 learner updates, 21,454
environment steps, and 1,030.75 seconds on ROCm. The manifest records clean
source `b52bf155075be0aba94ab87ff954362e6ac7a173`, `max_train_steps`, and normal
collector shutdown under the one-thread host resource cap.

The reference core materially improves capability and learning speed. Mean
evaluation first exceeds 450 at update 1,200, reaches 500 at updates 1,500,
1,800, 1,900, 2,100, and 2,500, and reaches at least 450 at 13 evaluations.
This is substantially earlier and more repeatable solved behavior than the
recent legacy-core controls. The complete 100-update evaluation sequence is:

```text
9.35, 9.35, 9.35, 9.35, 18.45, 37.50, 119.70, 152.90, 399.85,
437.00, 429.10, 471.45, 497.85, 476.40, 500.00, 464.35, 281.45,
500.00, 500.00, 446.80, 500.00, 385.35, 446.05, 496.70, 500.00,
222.35, 325.40, 499.15, 475.75, 195.15, 311.90, 231.15, 237.30,
498.15, 313.00
```

The behavioral stability gate nevertheless fails. After first solving, return
falls below 300 at updates 1,700, 2,600, 3,000, 3,200, and 3,300. Best return is
`500` at update 1,500 and final return is `313`, so final is below 400 and the
best-to-final gap is `187`. The implementation correction is retained because
it restores the pinned architecture and exposes substantially more capability;
it is rejected only as a sufficient explanation of the late collapse.

The required deterministic final continuation audit is retained under
`experiments/2026-07-22_cartpole_reference_rssm_core_continuation_probe/`.
Seeds 17--36 average return `303.1` with no truncations, yielding 6,062 rows and
20 physical terminals. The final posterior predicts mean continuation `0.804`
on terminal rows and `0.787` on live rows: terminal states are assigned *more*
survival than live states. Failure AUC is `0.291`, balanced accuracy is `0.500`,
and the Brier score is `0.0533`. Mode-state and prior-state audits agree
(failure AUC `0.317` and `0.302` respectively), while posterior-to-prior
transport RMS is only `0.00464`. Terminal and near-terminal rows remain around
`0.805` continuation versus `0.786` at distance ten or greater.

This localizes the next boundary to continuation-head supervision rather than
posterior-to-prior transport. The exact recurrent core makes solved behavior
accessible but does not teach the natural-prior continuation head the physical
failure boundary. The earlier balanced-loss intervention demonstrated that
greater terminal gradient exposure can produce useful ranking, but its raw
sigmoid represents a 50/50 reweighted class distribution and is therefore not
a valid imagination discount. The next intervention must increase rare
terminal learning while preserving—or explicitly recovering—the real
continuation probability. Because the behavior gate failed, the conditional Q
probe is not run and no broad seed sweep is authorized.

#### Preregistered cross-reset replay-stream canary

The next replay audit distinguishes unbiased expectation from usable gradient
variance. The existing episode-window inclusion correction is mathematically
unbiased, but a physical terminal can occur only in the single local window
whose end is the episode end. For batch size `8`, sequence length `16`, and
episode length `L`, the probability that an update contains any terminal is
`1 - (1 - 1 / (L - 15))^8`. Its rare terminal row is then multiplied by about
`16 * (L - 15) / L` to recover the correct expected class mass.

The completed reference-core run ended with rolling training episode length
`176.32`. At `L=176`, only `4.86%` of local updates are expected to contain a
terminal, the terminal weight is `14.64`, and the estimator's terminal-gradient
variance is `14.63` times that of a unit-weight stream estimator. The run's
logged snapshots agree: only 4 of 60 diagnostic batches at or after update
2,000 contained a terminal, and observed terminal weights reached `14.14`.
At `L=300`, the local terminal-bearing probability falls to `2.77%`.

Pinned reference commit `e3f0224` instead appends every environment step to a
per-worker stream and registers one replay item at every valid sequence start.
Sequences remain contiguous within a worker but may cross episode boundaries;
`is_first` resets the recurrent carry inside the sampled sequence. An interior
terminal therefore appears at each of the `T` possible sequence offsets. With
`B=8`, `T=16`, and `L=176`, the approximate probability of at least one
terminal becomes `1 - (1 - 1 / 176)^128 = 51.78%`, using unit row weights and
the natural class prior. This is variance reduction, not class rebalancing.

- **Hypothesis:** rare, high-magnitude terminal gradient spikes let the natural
  continuation head repeatedly forget CartPole's failure boundary. Sampling
  reference-style cross-reset streams will preserve the same expected natural
  BCE while exposing terminal rows in many more updates, retaining calibrated
  failure discrimination and reducing post-solve behavior collapse.
- **Causal variable:** add an authored `stream` replay-sequence mode. Retain
  complete episode transport, but identify each collector's ordered episode
  stream and sample every valid length-`T` start across consecutive episodes
  from that collector. Mark the first sampled row and each crossed episode
  boundary with `is_first`; zero that batch row's RSSM deterministic and
  stochastic carry before its observe step. Stream rows use unit continuation
  inclusion weights. Preserve the current per-episode sampler as `episode`
  mode for historical configuration/resume semantics. Do not change natural
  continuation targets, loss weights, batch size, sequence length, replay
  ratio, recent fraction, architecture, optimizer, rates, or actor/critic
  equations.
- **Mechanical gate:** exact synthetic enumeration must show one stream window
  per valid start and `T` appearances for an interior terminal; sequences may
  cross consecutive episodes from one collector but never collectors or a gap
  in episode IDs. State/action/reward/terminal/future-return fields must remain
  aligned. `is_first` must reset both recurrent carries and block gradients
  across the boundary while retaining within-episode BPTT. Episode mode and
  historical config loading must remain unchanged. Focused/full tests,
  compile/type gates, and a one-update multiprocess CPU smoke must pass.
- **Frozen run:** repeat seed 0 for 3,500 updates from the completed reference-
  RSSM-core configuration and fixed 20-episode evaluation cohort, changing
  only replay sequence mode from `episode` to `stream`. Retain the one-thread
  host resource cap.
- **Behavioral gate:** reach mean return `450`; never fall below `300`
  afterward; final at least `400`; best-to-final gap at most `100`.
- **Continuation gate:** on deterministic seeds 17--36, final posterior
  terminal continuation probability must be below `0.5`, terminal recall must
  exceed `0.5`, terminal/live balanced accuracy must exceed `0.7`, and live
  mean continuation must remain above `0.9` so the score is usable as a
  discount rather than merely a balanced classifier.
- **Boundary gate:** only if behavior passes, final fixed-history Q/real
  accuracy and correlation must not regress below `0.592` and `0.088`, and
  actor/Q agreement must remain above `0.8`.
- **Stop rule:** one seed only. Failure rejects cross-reset replay exposure as
  a sufficient stability fix and does not authorize combining it with balanced
  BCE, logit correction, optimizer changes, or larger batches.

#### Cross-reset replay-stream mechanical result

The implementation gate passes before launch. Canonical Hydra-authored runs now
select `stream`, while the dataclass and historical/missing snapshot fallback
remain `episode`; resume restores the checkpoint's authored replay semantics.
Collector payloads carry collector and episode IDs. Replay groups only
consecutive episodes from the same collector, samples uniformly over every
valid stream start (including starts whose sequence crosses a reset), emits
unit continuation weights, and marks the sampled first row plus each real reset
with `is_first`. The observe path zeros deterministic and stochastic carries
only for marked batch rows, retaining within-episode BPTT and blocking gradient
flow across resets. Existing `is_last` pair masks independently prevent replay
critic targets from crossing those boundaries.

Synthetic enumeration confirms six valid length-3 starts in an eight-row
two-episode stream and exactly three appearances of an interior terminal. Tests
also cover aligned observation/action/reward/terminal/future-return fields,
unit stream weights, recent/uniform selection compatibility, no cross-collector
or episode-ID-gap composition, episode-mode padding/importance behavior, and
row-selective recurrent gradient blocking. Focused runtime/config/resume tests
passed (`48`), the full fast suite passed (`162`), compile and the supported
scoped type gate passed, and a one-update canonical stream-mode CPU process
smoke completed with its collector stopped.

Final review also closed two replay liveness edges without changing the
scientific intervention. Stream readiness now clears when FIFO eviction leaves
no complete same-collector sequence, so a sampler blocks until a later episode
restores a gap-free stream. When replay-ratio pacing pauses the learner, the
trainer extends the absolute collection budget far enough to make the next
update affordable; episode-sized queue delivery can no longer deadlock below
that threshold. Regression tests cover both readiness loss/recovery and fresh
and resumed pacing budgets. On the final reviewed tree, the focused
config/replay/resume/RSSM suite passed (`50`), the full fast suite passed
(`166`), and source, test, and script compilation passed.

A bounded sampler microprofile used the canary capacity of 512 stored episodes,
episode length 176, `B=8`, `T=16`, and 1,000 batch draws. Episode mode required
`0.053 ms` per batch and produced 40 terminal-bearing batches; stream mode
required `0.142 ms` per batch and produced 521 terminal-bearing batches. The
additional `0.089 ms` of replay bookkeeping is negligible beside model
forward/backward, while the observed terminal exposure matches the predicted
roughly tenfold increase in terminal-bearing updates.

#### Cross-reset replay-stream canary result

The frozen seed-0 run completed normally under
`experiments/2026-07-22_cartpole_replay_stream_seed0_3500/`, with manifest run
ID `5a63760c4a3b413d97374d35b3f64cb4` and MLflow run ID
`c5615d0e62bb4f469d62de1c27cdda96`. It used 3,500 learner updates, 21,816
environment steps, and 1,024.2 seconds on ROCm. The manifest records clean
source `4cb5671`, `max_train_steps`, final/best/periodic checkpoints, and
successful collector shutdown under the one-thread host resource cap.

The stream run first solves at update 1,100, reaches 500 at updates 1,400,
1,500, and 1,600, and then remains variable. Its complete 100-update evaluation
sequence is:

```text
9.35, 9.35, 9.35, 9.40, 39.15, 110.40, 302.15, 360.25, 363.60,
349.30, 496.60, 321.50, 334.85, 500.00, 500.00, 500.00, 350.35,
227.60, 365.05, 376.00, 381.85, 415.50, 373.00, 488.85, 446.30,
406.70, 373.25, 462.00, 470.40, 196.05, 478.85, 203.65, 397.15,
310.25, 205.00
```

The behavioral gate fails. Return falls below 300 after first solve at updates
1,800, 3,000, 3,200, and 3,500. Best return is `500` at update 1,400 and final
return is `205`, so final is below 400 and the best-to-final gap is `295`.
Compared with the episode-contained reference-core canary, stream replay solves
100 updates earlier but has a lower final return (`205` versus `313`) and a
larger best-to-final gap (`295` versus `187`). The conditional Q probe is not
run because the behavior gate failed.

The deterministic continuation audit is retained under
`experiments/2026-07-22_cartpole_replay_stream_continuation_probe/`. Seeds
17--36 average return `195.75` with no truncations, yielding 3,915 rows and 20
physical terminals. Cross-reset replay fixes risk *ordering*: posterior failure
AUC rises from the episode canary's `0.291` to `0.998`, terminal continuation
falls from `0.804` to `0.772`, live continuation rises from `0.787` to `0.972`,
and the distance profile changes monotonically from `0.772` at failure to
`0.980` at distance ten or greater. Prior failure AUC is `0.995`, and
posterior-to-prior transport RMS remains small at `0.00535`.

The preregistered continuation gate nevertheless fails. Terminal continuation
remains above `0.5`; terminal recall at half-discount is zero and balanced
accuracy is `0.500`. The intervention therefore did what its sampling analysis
predicted---every logged diagnostic batch contained terminal supervision and
the final head learned an almost perfect danger ranking---but variance reduction
alone did not create enough logit margin for a calibrated physical termination
decision. Cross-reset stream replay is retained as a reference-conformance and
variance fix, not as a sufficient CartPole stability fix. No broad seed sweep
is authorized by this result.

#### Preregistered stream best/final boundary probe

The stream canary now supplies a solved checkpoint and a collapsed checkpoint
from one controlled trajectory. Before changing training again, compare those
checkpoints on an identical environment-state history to identify which learned
boundary moved.

- **Question:** between the best checkpoint at update 1,400 and final checkpoint
  at update 3,500, does real action-value knowledge degrade, or does the actor
  stop expressing knowledge that the model/value path retains?
- **Frozen data and measurement:** use 512 states from the same seed-17 random-
  action CartPole history for both checkpoints. Compare each action with a
  trusted 30-step real-simulator rollout. Measure the authored three-step
  model/critic action value, actor preference, one-step state prediction, and
  the hybrid learned-one-step/real-remainder control. No model is updated.
- **Primary readout:** report Q/real accuracy and delta correlation, actor/real
  accuracy, and actor/Q agreement on non-tied Q rows. Report one-step state MSE
  and hybrid-state/real accuracy as model controls.
- **Interpretation rule:** if final Q/real and hybrid controls hold or improve
  while actor/Q agreement falls below `0.8` or by at least `0.1`, localize the
  changed boundary to actor transfer. If final Q/real accuracy or correlation
  falls by at least `0.1` together with a degraded model control, localize it to
  model/value knowledge. If both move materially, classify the collapse as
  coupled rather than selecting an actor-only or model-only fix.
- **Stop rule:** two read-only checkpoints only. Record the result before any
  optimizer, loss, architecture, or training-run proposal.

#### Stream best/final boundary-probe result

The read-only probe is retained under
`experiments/2026-07-22_cartpole_replay_stream_boundary_probe/`. Both
checkpoints saw the identical 512-state seed-17 random-action history, including
104 states where the two trusted 30-step real rollouts had a strict preference.

| Checkpoint | Q/real accuracy | Q/real correlation | Actor/real accuracy | Actor/Q actionable | One-step MSE | Hybrid/real margined |
|---|---:|---:|---:|---:|---:|---:|
| Best, update 1,400 | 0.769 | 0.499 | 0.875 | 0.894 | 0.0412 | 0.746 |
| Final, update 3,500 | 0.856 | 0.694 | 0.875 | 0.981 | 0.0186 | 0.932 |

The collapsed final checkpoint is better on every model/value control and does
not lose actor transfer on these states. Its three-step survival-only ordering
also improves from `0.625` accuracy and `0.490` correlation to `0.846` and
`0.793`. This rejects both preregistered simple mechanisms: fixed-history
model/value knowledge did not degrade, and the actor did not stop following its
learned Q preference there.

The remaining discrepancy is distributional. The fixed history represents
ordinary random-policy states, while each deployed policy induces a different
long-horizon state distribution. A checkpoint can improve on the former yet
visit a narrower or more dangerous subset under its own actor, where small
systematic errors compound. The next diagnostic must therefore compare
best/final on-policy critical and near-terminal decisions rather than propose a
global optimizer or loss change from the fixed-state result.

#### Preregistered stream on-policy distribution probe

- **Question:** which on-policy state/decision statistic changes when the same
  training trajectory moves from solved best behavior to collapsed final
  behavior despite improving on a fixed random-state history?
- **Frozen cohort:** run deterministic actors from the best and final
  checkpoints on reset seeds 17--36. At every visited state, compare both first
  actions with a trusted 30-step real-simulator rollout and compute the same
  three-step learned Q and one-step model controls. No weights are updated.
- **Primary readout:** mean/min/max return; actor/real accuracy overall, at
  trusted margin at least 15, and within ten steps of episode end; avoidable
  immediate failures; actionable errors per episode; actor/Q agreement; and
  one-step state MSE. Compare action/state distributions from the retained row
  files if aggregate accuracy does not explain the return gap.
- **Interpretation rule:** lower final critical or terminal-window accuracy
  selects an on-policy policy-error boundary. Stable accuracies with more final
  visits to high-regret or near-failure states selects state-distribution drift.
  Simultaneous degradation in on-policy learned-Q/real accuracy selects a
  coupled model/value distribution-shift boundary. Do not infer causality from
  raw return alone.
- **Stop rule:** these two read-only checkpoints only; document the result before
  proposing another training intervention.

#### Stream on-policy distribution-probe result

The read-only result is retained under
`experiments/2026-07-22_cartpole_replay_stream_on_policy_probe/`. The best
checkpoint solves all 20 seeds at the 500-step time limit (10,000 states); the
final checkpoint physically fails on all 20 at mean return `195.75` (3,915
states, range 166--243).

| Metric | Best, update 1,400 | Final, update 3,500 |
|---|---:|---:|
| Actor/real accuracy | 0.517 | 0.476 |
| Actor/real balanced accuracy | 0.572 | 0.619 |
| Critical-margin accuracy | 0.311 | 0.684 |
| Terminal-window accuracy | 0.508 | 0.424 |
| Episodes with terminal-window error | 6 / 20 | 20 / 20 |
| Three-step Q/real accuracy | 0.430 | 0.632 |
| Three-step Q/real balanced accuracy | 0.623 | 0.576 |
| Actor/three-step-Q agreement | 0.892 | 0.778 |
| One-step state MSE | 0.0834 | 0.0337 |

The aggregate accuracies do not support a global model collapse. Final
one-step prediction is substantially better, raw Q/real accuracy improves, and
critical-margin actor accuracy improves. Instead, the state rows show a sharp
deployment-distribution change. Every final episode ends at the negative cart
position limit: the last pre-action positions are approximately `-2.38` to
`-2.40`, with pole angle near `-0.11`. Final states have median absolute cart
velocity `0.529` versus `0.158` at best, median absolute pole angle `0.0445`
versus `0.0032`, and 90th-percentile absolute position `1.81` versus `0.98`.
The final trusted preference is action 1 on 2,055 of 2,168 actionable states,
while the actor executes slightly more action 0 than action 1 overall
(`2,051` versus `1,864`). Its actionable error rate rises from `0.407` more
than 100 steps before failure to roughly `0.53--0.58` over the final 100 steps.

Thus the final policy enters a coherent left-drift regime and fails to produce
the sustained rightward correction needed to leave it. This is not an
unavoidable one-step trap: there are no states where the chosen action fails
immediately and the alternative survives, because errors accumulate before the
last state. The best actor stays near upright and reaches time-limit truncation;
its apparent near-end errors are not physical terminal errors.

The changed boundary is coupled state-distribution/actor transfer. Learned
one-step dynamics and action-value ranking improve globally, but actor/Q
agreement drops by `0.114` on the induced final distribution and below the
preregistered `0.8` threshold. This does not yet prove that the actor disagrees
with its *authored* policy-conditioned lambda-return target: the three-step Q
diagnostic maximizes future branches. Before a training proposal, the next
audit must inspect the actor's exact REINFORCE target and gradient on these
left-drift states, or use the existing policy-conditioned Q estimator on a
bounded retained cohort.

#### Preregistered stream policy-target tracking probe

- **Question:** does the best-to-final loss of actor/planner agreement also
  occur against the exact sampled target used by the authored REINFORCE loss?
- **Frozen cohort:** run the best and final deterministic actors on only reset
  seeds 17--19. At every visited state, estimate both forced first actions with
  64 Monte Carlo rollouts using the checkpoint's own 15-step horizon, sampled
  categorical priors, subsequent 1%-unimixed actor actions, learned reward and
  continuation, online critic bootstrap, and lambda `0.95`. No weights change.
- **Primary readout:** among states whose policy-Q difference exceeds 1.96
  combined standard errors, report actor agreement, sample count, and policy-Q/
  trusted-real balanced accuracy on actionable states. Retain the corresponding
  best/final return and state counts.
- **Interpretation rule:** best agreement at least `0.8` followed by final below
  `0.8` selects moving-target actor lag. Agreement at least `0.8` for both but
  worse final policy-Q/real accuracy selects a target-quality failure. Both
  below `0.8` selects a systemic policy-optimization mismatch. Fewer than 100
  separated states in either checkpoint is inconclusive.
- **Stop rule:** three episodes per checkpoint and 64 samples only. Do not tune
  the estimator or launch training before recording the result.

#### Stream policy-target tracking result

The read-only result is retained under
`experiments/2026-07-22_cartpole_replay_stream_policy_q_probe/`. The best
checkpoint solves all three episodes (1,500 states); final averages `192.33`
over 577 states. The 64-sample, 15-step policy-conditioned estimator gives:

| Checkpoint | Confident states | Actor/policy-Q confident | All actor/policy-Q | Policy-Q/real balanced | Mean abs policy-Q delta |
|---|---:|---:|---:|---:|---:|
| Best, update 1,400 | 392 | 0.954 | 0.812 | 0.563 | 0.107 |
| Final, update 3,500 | 69 | 0.884 | 0.673 | 0.586 | 1.529 |

The best checkpoint passes the tracking gate with ample support. The final
checkpoint has only 69 differences larger than 1.96 combined standard errors,
so the preregistered minimum of 100 makes its categorical result inconclusive.
It does not provide evidence for a simple actor sign/index bug: on the final
states where its own sampled target clearly separates, the actor still agrees
`88.4%` of the time. However, agreement over all actionable rows drops, only 40
confident final rows have a trusted real preference, and their balanced
policy-Q/real accuracy is `0.400`.

The target geometry changes sharply. Final mean absolute policy-Q difference is
about fourteen times the best value, yet far fewer rows exceed their Monte
Carlo uncertainty. This is consistent with a higher-variance, less coherent
imagined target on the induced left-drift distribution. It does not establish
whether the variance comes primarily from stochastic latent transition,
continuation, reward, or critic bootstrap, and the stop rule forbids increasing
samples after seeing the result.

This evidence makes the already-audited optimizer/time-scale mismatch relevant
again. Pinned reference `e3f0224` uses one optimizer over world model, policy,
and value at learning rate `4e-5`, with a 1,000-update linear learning-rate
ramp. The frozen canary instead uses separate rates `3e-4`, `3e-5`, and `8e-5`
without a ramp: the representation/transition target can move ten times faster
than the actor parameters. The next step is a source-level audit and a coherent
reference optimizer contract, not an after-the-fact increase to diagnostic
samples.

#### Preregistered reference optimizer-contract canary

The source audit against pinned reference `e3f0224` confirms four coupled
optimizer semantics: one parameter-local LaProp transform, one `4e-5` learning
rate, a linear 0-to-`4e-5` ramp over the first 1,000 updates, and replay-value
gradients enabled into the observed encoder/RSSM representation. Local LaProp
and per-tensor AGC keep no optimizer-global statistics, so three disjoint
optimizer containers are numerically equivalent to one container when their
rates/schedules match and shared representation losses are backpropagated
together. Retaining the containers avoids a checkpoint-format migration; it is
an implementation detail, not a semantic relaxation.

- **Hypothesis:** the local `3e-4` world-model rate moves imagined latent and
  continuation targets too quickly for the `3e-5` actor, while detached replay
  value loss cannot shape those features. The coherent reference contract will
  reduce late target variance and prevent the learned policy from entering the
  systematic left-drift regime.
- **Causal bundle:** add an authored `reference` optimizer contract that uses
  `4e-5` for world model, actor, and critic; linearly ramps all three from zero
  over updates 0--1,000; and applies the scale-0.3 replay-value loss to both its
  critic head and observed encoder/RSSM features in one combined backward pass.
  All modules still train from update zero---this is not the removed actor-
  freeze warmup. Preserve a `legacy` contract for historical snapshots and
  resume. Do not change architecture, replay, losses, actor objective, entropy,
  batch/sequence sizes, train ratio, or benchmark budget.
- **Mechanical gate:** prove reference rates at updates 0, 500, 1,000, and later;
  legacy constant/cosine behavior unchanged; replay value reaches encoder/RSSM
  only in reference mode; combined backward has the same per-parameter update as
  one joint LaProp optimizer; and snapshots/resume preserve their contract.
  Focused/full tests, compile/type gates, and a multiprocess CPU smoke must pass.
- **Frozen run:** repeat seed 0 for 3,500 updates from the completed reference-
  core, stream-replay configuration and fixed 20-episode evaluation cohort,
  changing only optimizer contract. Use reference rate `4e-5`, ramp 1,000, and
  the one-thread host resource cap.
- **Behavioral gate:** reach mean return `450`; never fall below `300`
  afterward; final at least `400`; best-to-final gap at most `100`.
- **Diagnostics:** regardless of behavior, run the deterministic final
  continuation audit. Only if behavior passes, run the fixed-history Q boundary
  probe with the existing floors (`0.592` accuracy, `0.088` correlation, and
  actor/Q agreement above `0.8`). If late degradation persists, repeat the
  bounded left-drift and exact policy-target readouts before selecting a cause.
- **Stop rule:** one seed only. Failure rejects this coherent optimizer contract
  as sufficient and does not authorize stacking actor-head depth, balanced
  continuation, larger replay batches, or diagnostic resampling.

#### Reference optimizer-contract canary result

The frozen seed-0 run completed normally under
`experiments/2026-07-22_cartpole_reference_optimizer_seed0_3500/`, with manifest
run ID `32c41510ade04d36891b6085102ec750` and MLflow run ID
`27ffda015826420fb00047e2c59f3f82`. It used 3,500 learner updates, 21,435
environment steps, and 918.4 seconds on ROCm. The manifest records clean source
commit `90946ccc0d228b0845f66d007838145b836b9fc3`, the collector stopped normally,
and periodic, best, and final checkpoints were retained separately.

The 20-episode evaluation means at each 100-update boundary were:

```text
100:9.40 200:16.35 300:9.35 400:9.35 500:9.35 600:9.35
700:9.35 800:9.35 900:9.35 1000:9.35 1100:9.35 1200:22.45
1300:9.40 1400:11.45 1500:9.35 1600:9.35 1700:10.95
1800:75.70 1900:39.00 2000:34.75 2100:42.25 2200:60.10
2300:62.25 2400:170.95 2500:129.10 2600:141.35 2700:189.65
2800:195.20 2900:162.00 3000:176.10 3100:133.85 3200:45.80
3300:34.95 3400:107.25 3500:170.15
```

The behavioral gate failed: the run never reached `450`, its best was only
`195.20` at update 2,800, and its final return was `170.15`. The small
best-to-final gap of `25.05` does not rescue the gate because the policy never
entered the solved regime. Relative to the stream-replay canary, the coherent
reference time scale delayed useful behavior by roughly 700 updates and then
plateaued below 200 before a transient fall to `34.95` at update 3,300.

The required deterministic final continuation audit is under
`experiments/2026-07-22_cartpole_reference_optimizer_continuation_probe/`. On
seeds 17--36 it produced mean return `167.6` across 3,352 transitions and 20
physical terminals. Posterior effective discount was `0.99107` on terminal
transitions and `0.99118` on live transitions, with failure ROC AUC `0.523` and
balanced accuracy `0.5`; the prior result was similarly uninformative (terminal
`0.99093`, live `0.99063`, AUC `0.410`). Posterior/prior transport RMS was only
`0.00085`, so this is a learned continuation-head failure rather than latent
transport corruption.

This rejects the optimizer bundle as sufficient and, more specifically, shows
that it regresses the stream canary's strong terminal-risk ordering (posterior
AUC `0.998`) to chance. The fixed-history Q probe is skipped because the
behavioral gate failed. Before choosing another intervention, use the already
bounded on-policy left-drift and exact policy-target readouts to distinguish a
globally weak policy/value system from the previous late distributional drift;
do not stack another training change onto this failed canary.

#### Preregistered reference optimizer boundary readouts

- **Question:** did the best-to-final degradation reproduce the stream canary's
  left-drift distribution shift, and does the actor track its exact sampled
  policy-conditioned target despite the failed continuation head?
- **On-policy cohort:** compare best update 2,800 and final update 3,500 on the
  same deterministic seeds 17--36, real horizon 30, learned-model horizon 3,
  decomposition horizons 1 and 3, critical real-action margin 15, and terminal
  window 10 used for the stream readout.
- **Exact target cohort:** compare the same two checkpoints on only seeds
  17--19, using 64 sampled 15-step policy-conditioned returns per forced first
  action. Retain the existing 1.96-combined-standard-error confidence rule and
  require at least 100 separated states before a categorical tracking claim.
- **Interpretation:** similar left-drift statistics with actor/target agreement
  below `0.8` would reproduce the prior transfer failure; globally weak action
  ranking or ample actor/target agreement would instead show that the coherent
  bundle changed the failure boundary. These readouts describe the failed run
  and cannot rescue it.
- **Stop rule:** these two checkpoint comparisons only; do not resample, tune,
  or launch another training intervention before recording both results.

#### Reference optimizer boundary-readout result

The 20-seed on-policy comparison is retained under
`experiments/2026-07-22_cartpole_reference_optimizer_on_policy_probe/`:

| Metric | Best, update 2,800 | Final, update 3,500 |
|---|---:|---:|
| Mean return / solved episodes | 159.05 / 0 of 20 | 167.60 / 0 of 20 |
| Actor/real accuracy | 0.488 | 0.493 |
| Actor/real balanced accuracy | 0.489 | 0.592 |
| Critical-margin accuracy | 0.464 | 0.462 |
| Terminal-window accuracy | 0.695 | 0.508 |
| Three-step Q/real balanced accuracy | 0.444 | 0.562 |
| Three-step Q/real correlation | -0.001 | 0.020 |
| Actor/three-step-Q agreement | 0.490 | 0.723 |
| Mean one-step state MSE | 0.191 | 0.455 |
| Mean one-step prior x MSE | 0.306 | 1.225 |

The failure distribution changes across training. The best checkpoint fails by
pole angle in both directions, with terminal x near plus or minus `1.4--1.6`
and theta near plus or minus `0.205`; it is globally weak rather than a solved
policy observed on rare recovery states. Every final episode instead crosses
the negative cart boundary at x approximately `-2.38--2.40` while theta remains
near upright. On the final distribution the trusted real rollout prefers action
1 on 2,214 of 2,314 actionable states, but the actor executes action 0 on 1,733
of all 3,352 states. This reproduces the systematic left-drift endpoint, while
the roughly balanced action histogram shows it is not a constant-action actor.
World-model state fidelity also degrades materially from best to final.

The exact 64-sample, 15-step readout is retained under
`experiments/2026-07-22_cartpole_reference_optimizer_policy_q_probe/`:

| Metric | Best, update 2,800 | Final, update 3,500 |
|---|---:|---:|
| Three-seed mean return | 187.0 | 168.0 |
| Statistically separated states | 176 | 309 |
| Actor/policy-Q agreement on separated states | 0.955 | 0.663 |
| All-state actor/policy-Q agreement | 0.830 | 0.533 |
| Separated actionable states | 75 | 221 |
| Policy-Q/real balanced accuracy there | 0.470 | 0.500 |
| Policy-Q/real raw accuracy | 0.415 | 0.060 |
| Mean absolute policy-Q action margin | 0.153 | 0.378 |

Both checkpoints exceed the 100-state confidence floor, so the tracking result
is categorical. At the best checkpoint the actor accurately follows a sampled
policy target that is already worse than chance against real rollouts. At the
final checkpoint the target becomes more confident and selects action 0 on 483
of 504 states even though the trusted rollout prefers action 1 on 336 of 351
actionable states; actor/target agreement then also falls below `0.8`.

The coherent optimizer bundle therefore changes, but does not repair, the
failure chain. Its first demonstrated broken boundary is learned policy-target
quality, not actor tracking: bad target advice is present while tracking still
passes. Later training adds actor/target mismatch, worse one-step state fidelity,
chance continuation ranking, and the familiar left-drift state distribution.
Continuation saturation is a plausible contributor to the target inversion,
but these readouts do not isolate it causally. The next intervention must be
selected against the earlier stream checkpoint, which preserved terminal-risk
ranking and reached solved behavior; the full reference optimizer bundle should
not become the authored default on the strength of this result.

#### Preregistered sampled-policy deployment audit

Pinned reference DreamerV3 `e3f0224` samples categorical policy actions in its
policy method, including evaluation calls; the local benchmark deliberately
uses deterministic `argmax` evaluation so a reported solution is reproducible.
The difference may expose a real deployment boundary: a stochastic policy can
cancel small directional biases over time, while `argmax` repeatedly applies
the same side of a nearly tied distribution and can accumulate cart drift.

- **Question:** are the stream canary's solved-best/collapsed-final distinction
  and the reference-optimizer canary's weak behavior specific to deterministic
  policy extraction, or are their categorical policies also weak when sampled?
- **Frozen cohort:** evaluate stream best (update 1,400), stream final (3,500),
  reference-optimizer best (2,800), and reference-optimizer final (3,500) on
  environment reset seeds 17--116. For every checkpoint, compare `argmax` with
  categorical sampling using a separate action generator seeded from the same
  fixed base. Use posterior latent modes in both conditions, matching the
  existing deterministic evaluation except for action selection.
- **Primary readout:** 100-episode mean, median, range, and solved fraction for
  each mode/checkpoint. The earlier 20-seed deterministic values remain context,
  but this larger frozen cohort is the paired comparison of record.
- **Interpretation:** sampled mean at least 450 with solved fraction at least
  0.8 while `argmax` fails localizes useful stochastic policy knowledge that is
  lost at deterministic extraction. If both fail, sampling does not rescue the
  learned behavior. If sampling materially harms a solved `argmax` checkpoint,
  entropy/probability calibration is itself a deployment problem.
- **Stop rule:** these eight read-only evaluations only. Do not tune action
  temperature, sampling seeds, or training from this result. Select the next
  intervention only after recording all eight cells.

#### Sampled-policy deployment-audit result

The paired evaluation is retained under
`experiments/2026-07-22_cartpole_sampled_policy_audit/`. All eight cells used
the same reset seeds 17--116; sampled cells used the independent action seed
1,000,017.

| Checkpoint | Argmax mean / median / solved | Sample mean / median / solved |
|---|---:|---:|
| Stream best, update 1,400 | 500.00 / 500.0 / 1.00 | 500.00 / 500.0 / 1.00 |
| Stream final, update 3,500 | 203.02 / 199.5 / 0.00 | 191.40 / 185.0 / 0.00 |
| Reference optimizer best, update 2,800 | 173.28 / 155.5 / 0.02 | 153.74 / 141.0 / 0.00 |
| Reference optimizer final, update 3,500 | 169.33 / 168.5 / 0.00 | 166.84 / 166.0 / 0.00 |

The stream-best policy solves every episode under both extraction modes, while
sampling does not solve a single episode from any of the three failed
checkpoints. Sampling slightly reduces all three failed means rather than
recovering hidden stochastic policy knowledge. The deterministic benchmark is
therefore stricter than pinned reference evaluation in implementation, but it
does not cause the observed instability. The solved-to-collapsed change is in
the learned policy distribution itself, not merely its `argmax` projection.

This rejects action sampling or temperature as the next intervention. Return
to the earlier stream contract, whose best checkpoint demonstrates adequate
model and optimizer capacity, and audit the training signal that later moves
that same policy away from solved behavior.

#### Preregistered episode-coherent collector canary

The collection audit found a concrete recurrent-state contract violation.
`collect_experiences()` polls the weight mailbox before every environment step.
When an update arrives, it replaces the encoder, RSSM, and actor parameters but
retains `h`, `action_onehot`, and `z_prev_embed` from the previous parameter
snapshot. The next policy decision therefore combines a hidden state expressed
by one RSSM with transition and policy functions from another. Raw replay rows
remain valid environment transitions, but the behavior policy that generated
them is neither the old policy nor the new policy.

This is not hypothetical update traffic. The failed reference-optimizer run
published 700 versions and loaded 470 of them across 740 completed episodes;
the source permits any of those loads to occur mid-episode. The effect becomes
especially relevant as CartPole episodes lengthen and will be more severe for
Pong, where the recurrent state carries information absent from one frame.

- **Hypothesis:** preserving one coherent `(parameters, recurrent carry)` pair
  for each episode prevents artificial policy-state discontinuities and reduces
  the solved-to-collapsed feedback loop in the stream canary.
- **Causal change:** apply collector weight updates only immediately before an
  episode reset, when recurrent state is rebuilt. Preserve one independent,
  one-item mailbox per collector, but replace an unseen stale snapshot with the
  newest published snapshot so a long episode does not force the next episode
  to start from an obsolete version. Do not change sync frequency, stochastic
  action/latent sampling, replay contents, training losses, architecture, or
  optimizer settings.
- **Mechanical gate:** unit-test latest-snapshot replacement and fan-out; prove
  by process-smoke logs that loaded versions identify an episode boundary; run
  focused/full tests, compile/type checks, and the multiprocess CPU smoke.
- **Frozen run:** repeat seed 0 for 3,500 updates with the exact prior stream
  configuration: reference RSSM/core and continuation head, `d_hidden=128`,
  batch 8, sequence 16, burn-in 4, replay ratio 16, legacy optimizer contract,
  rates `3e-4/3e-5/8e-5`, entropy `1e-3`, 20 deterministic evaluation episodes
  every 100 updates, and the one-thread host cap.
- **Behavioral gate:** reach mean return 450; never fall below 300 afterward;
  final at least 400; best-to-final gap at most 100.
- **Diagnostics:** always retain the full evaluation curve and collector version
  logs. If behavior fails, compare the final continuation and on-policy
  left-drift readouts before choosing another training change.
- **Stop rule:** one seed only. Failure rejects recurrent snapshot coherence as
  sufficient, while retaining it as a correctness fix; do not combine it with
  the missing start-continuation weight or an architecture change.

The mechanical implementation now swaps model snapshots only before reset and
uses newest-wins replacement in each independent mailbox. A three-update,
two-episode-minimum CPU process smoke published versions 0, 1, and 2 while the
collector was active; versions 0 and 1 were replaced without blocking, and the
collector loaded version 2 at the explicit boundary `episode=8`. The full fast
suite passes (`188 passed`), as do compile and the project-scoped type checks.
This establishes the collection contract but is not behavioral evidence; the
frozen canary remains required.

#### Episode-coherent collector canary result

The frozen seed-0 run completed normally under
`experiments/2026-07-22_cartpole_episode_coherent_seed0_3500/`, with manifest
run ID `e3620074487546ffaf3f796dfef605bf` and MLflow run ID
`ab14d731464447f8926977a2f45e6d33`. It used 3,500 updates, 21,474 environment
steps, and 1,007.7 seconds from clean source `5221e5c`. The collector applied
163 snapshots at episode boundaries, while newest-wins mailboxes replaced 536
stale pending snapshots across 700 publications. Final shutdown was normal.

The full 20-episode evaluation curve was:

```text
100:9.35 200:9.35 300:9.35 400:9.35 500:115.05 600:139.55
700:374.05 800:303.45 900:223.70 1000:405.20 1100:387.80
1200:201.40 1300:388.95 1400:262.15 1500:434.85 1600:428.25
1700:500.00 1800:486.80 1900:500.00 2000:493.45 2100:464.15
2200:375.35 2300:434.75 2400:384.25 2500:333.50 2600:434.45
2700:457.90 2800:416.15 2900:305.55 3000:261.90 3100:364.65
3200:246.00 3300:210.10 3400:217.65 3500:200.40
```

The behavioral gate fails. The run first reaches 500 at update 1,700 and then
holds above 300 through update 2,900, but falls to 261.90 at update 3,000 and
finishes at 200.40. Its best-to-final gap is 299.60. Episode coherence therefore
does not solve CartPole, but it materially changes the failure trajectory: the
prior stream run first solves at 1,100 and has several sub-300 collapses before
3,000, whereas this run sustains a contiguous 1,200-update post-solve interval
above the floor. Retain the runtime change as a recurrent-policy correctness
fix, not as a sufficient learning fix.

The required final continuation audit is retained under
`experiments/2026-07-22_cartpole_episode_coherent_continuation_probe/`. On seeds
17--36 it averages 194.3 return over 3,886 transitions. Posterior terminal
effective discount is 0.930 versus 0.987 on live rows, failure AUC is 0.992,
and the distance profile is monotonic from 0.930 at failure to 0.990 at distance
ten or greater. Prior AUC is 0.989 and posterior/prior transport RMS is 0.00164.
The continuation head remains poorly calibrated at the 0.5 cutoff, but it has
not forgotten terminal-risk ordering; that is not the first remaining boundary.

The best/final on-policy comparison is retained under
`experiments/2026-07-22_cartpole_episode_coherent_on_policy_probe/`:

| Metric | Best, update 1,700 | Final, update 3,500 |
|---|---:|---:|
| Mean return / solved fraction | 500.0 / 1.0 | 194.3 / 0.0 |
| Actor/real balanced accuracy | 0.607 | 0.604 |
| Critical-margin accuracy | 0.553 | 0.508 |
| Terminal-window accuracy | 0.514 | 0.430 |
| Three-step Q/real balanced accuracy | 0.648 | 0.628 |
| Three-step Q/real correlation | 0.119 | 0.004 |
| Actor/three-step-Q agreement | 0.867 | 0.589 |
| Mean absolute Q action margin | 0.201 | 3.804 |
| Mean one-step state MSE | 0.140 | 0.043 |

Final world-model one-step fidelity improves by about threefold, and balanced
Q/real ranking changes little, but Q correlation disappears, Q margins inflate
nineteenfold, and actor/Q agreement falls by 0.278. Every final episode again
enters the negative-x drift regime and crosses the left cart boundary; the last
episode, for example, reaches x `-2.381` with negative velocity while alternating
actions. Trusted real rollouts prefer action 1 on 2,162 of 2,297 actionable
final states, while the actor takes action 0 slightly more often overall.

Episode coherence therefore removes artificial recurrent discontinuities but
does not remove the learned late target/actor drift. The remaining failure is
not global one-step model accuracy or continuation ordering. Before selecting
the next training change, repeat the exact policy-conditioned target readout on
this cleaner solved/final pair; the planner-like three-step Q result cannot tell
whether the actor has stopped following its authored REINFORCE target.

#### Preregistered episode-coherent policy-target readout

- **Question:** after removing recurrent snapshot mixing, does late actor/Q
  disagreement reflect failure to track the actual sampled lambda-return target,
  or does that authored target itself become wrong on the left-drift states?
- **Frozen cohort:** compare best update 1,700 and final update 3,500 on reset
  seeds 17--19, using 64 sampled, 15-step policy-conditioned returns per forced
  first action and the existing 1.96-combined-standard-error separation rule.
- **Interpretation:** at least 100 separated states per checkpoint makes actor/
  target agreement categorical. Best at least 0.8 followed by final below 0.8
  selects transfer lag; agreement at least 0.8 for both with worse target/real
  ranking selects target quality; both moving selects a coupled failure.
- **Stop rule:** these two read-only checkpoints only. Do not increase samples
  or begin another training intervention before recording the result.

#### Episode-coherent policy-target-readout result

The result is retained under
`experiments/2026-07-22_cartpole_episode_coherent_policy_q_probe/`:

| Metric | Best, update 1,700 | Final, update 3,500 |
|---|---:|---:|
| Three-seed mean return | 500.0 | 191.67 |
| Statistically separated states | 610 | 264 |
| Actor/target agreement there | 0.870 | 0.803 |
| Separated actionable states | 189 | 146 |
| Target/real balanced accuracy there | 0.548 | 0.507 |
| All-state target/real balanced accuracy | 0.540 | 0.577 |
| Mean absolute target action margin | 0.151 | 3.022 |

Both checkpoints exceed the 100-state support floor and both actor/target
agreements meet the categorical `0.8` threshold. The remaining failure is
therefore target quality rather than simple actor fitting: final targets become
about twenty times more separated in magnitude, yet their confident action
ordering is at chance against trusted real rollouts. The actor continues to
express that target accurately enough to fail with it.

The one-step decomposition localizes most of this action margin to critic
bootstrap, not reward. At final, the one-step critic-bootstrap preference
selects action 0 on 507 of 575 states, has mean absolute difference 5.31, and
only 0.191 raw accuracy against the heavily action-1 recovery distribution.
The learned one-step reward difference remains approximately zero, as expected
for CartPole's constant reward. Continuation has strong failure ordering but
only modest action-specific discrimination. The actionable boundary is thus
the imagined critic/lambda-return target on recovery states.

This reproduces the earlier target-quality diagnosis after removing collector
state mixing. The next source-conformance candidate is the still-missing
imagination start-continuation weight: local policy/value losses give every
replay start weight one, including terminal states, whereas pinned reference
weights the first action by predicted continuation of the observed start and
carries that factor through the imagined trajectory. Its causal reach must be
measured before authorizing another canary because terminal and near-terminal
starts are rare.

#### Preregistered imagination-start-weight canary

The source audit confirms one exact consumer mismatch. The pinned reference
constructs its imagination weight sequence from the learned continuation at
the observed replay start followed by the learned continuations of imagined
successors. Local `compute_actor_critic_losses()` instead prefixes successor
continuations with one. Thus local step-zero policy and value losses are never
discounted by the start continuation, and every later imagined loss is missing
that same multiplicative factor. Return construction itself is already based
on successor continuations and will remain unchanged.

The causal prior is deliberately weak. Training telemetry from the episode-
coherent seed-0 run reports mean start continuation weight about `0.971`, so
the correction removes only about `2.9%` of aggregate actor/value loss mass.
Its only plausible high-leverage route is selective suppression of rare
failure-adjacent starts. The final continuation probe also found terminal-state
probability `0.930`, making even that suppression modest late in training.
This evidence does not justify a sweep.

- **Hypothesis:** multiplying every imagined actor/value loss by the detached
  continuation probability of its observed replay start prevents rare
  terminal or near-terminal starts from authoring disproportionately harmful
  critic targets, improving late recovery-state target ranking and behavioral
  retention.
- **Single causal variable:** add the missing start factor to the existing
  imagined-loss weights. Keep lambda-return construction, successor discount
  recurrence, replay, collector episode coherence, architecture, optimizers,
  rates, seeds, and all other settings fixed. New authored runs enable the
  behavior; historical checkpoint snapshots that lack the field retain the
  old unit-prefix semantics unless semantic migration is explicitly allowed.
- **Frozen run:** one CartPole seed (`0`), clean source after validation,
  `3,500` learner updates, one collector, `d_hidden=128`, reference RSSM,
  one-hidden-layer continuation head, stream replay, replay ratio `16`, buffer
  `512`, minimum `16` episodes, batch `8`, sequence length `16`, burn-in `4`,
  world-model/actor/critic rates `3e-4/3e-5/8e-5`, legacy optimizer contract,
  no warmup, entropy `1e-3`, online critic target, evaluation every `100`
  updates over `20` deterministic episodes, checkpoints every `500`, and no
  early stop. This is otherwise identical to the episode-coherent canary.
- **Behavior gate:** reach mean evaluation at least `475`; after the first such
  evaluation, never fall below `300`; finish at least `400`; and keep the
  best-to-final gap at most `100`.
- **Boundary readout:** regardless of the behavioral result, repeat the frozen
  reset-seed `17`--`19`, 64-sample, 15-step policy-target probe on best and
  final. A credible target-quality improvement requires final confident
  target/real balanced accuracy above the prior `0.507` and at least `0.55`,
  with at least 100 separated states. Actor/target agreement must remain at
  least `0.8`; otherwise the selected boundary has moved to actor fitting.
- **Stop rule:** run this seed once. Do not add seeds or combine another
  learning intervention if either gate fails. Retain the source-conforming
  correction as semantics infrastructure, record the negative result, and
  return to source/data-flow localization of the critic target.

#### Imagination-start-weight canary result

The mechanical correction and resume contract are retained at source commit
`3c4079e`. Authored Hydra runs now multiply the whole actor/value imagination
trajectory by the detached predicted continuation probability of its observed
replay start; historical snapshots missing the field retain the old unit
prefix. Focused config, resume, and loss tests passed (`39`), including exact
half-scaling of every actor and critic term for a start logit of zero and proof
that the factor is not a world-model gradient path. The full fast suite passed
(`190`), compile and the supported scoped type check passed, and a one-update
multiprocess CPU smoke completed normally. Ruff was not available in the
project environment and was not installed.

The frozen seed-0 run completed normally under
`experiments/2026-07-22_cartpole_start_weight_seed0_3500/`, with manifest run
ID `629f81be4b6f45a5ba61a3c0d4c633fe` and MLflow run ID
`38300d351e584accaf38af4e659e44d4`. It used 3,500 learner updates, 21,539
environment steps, and 1,034.9 seconds on ROCm. Its manifest records clean
source `3c4079e`, `weight_imagination_starts=true`, successful completion, and
separate best/final checkpoints. Observed mean terminal fraction was `0.0208`,
mean live start continuation was `0.9783`, and mean terminal start continuation
was `0.7326`; their prevalence-weighted mean factor is approximately `0.973`,
confirming the preregistered prediction that aggregate loss mass changes by
only about `2.7%`.

The policy first reached `493.15` at update 1,500 and reached the run best
`500.0` at updates 1,700 and 1,800. It then retained a long high-return region,
but not the frozen stability contract:

| Update | 1,500 | 1,700 | 2,000 | 2,300 | 2,400 | 2,500 | 3,000 | 3,100 | 3,500 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean return | 493.15 | 500.0 | 455.8 | 432.85 | **285.65** | 400.15 | 449.5 | **203.25** | 357.75 |

The first sub-300 evaluation at update 2,400 irreversibly fails the retention
gate. Final return also misses `400`, and the best-to-final gap is `142.25`.
Relative to the episode-coherent predecessor, the final is better (`357.75`
versus `200.4`) but the first sub-300 excursion occurs earlier (2,400 versus
3,000). This is not a uniform stability improvement and does not authorize
replication seeds.

The preregistered policy-target probe is retained under
`experiments/2026-07-22_cartpole_start_weight_policy_q_probe/`:

| Metric | Best, update 1,700 | Final, update 3,500 | Prior coherent final |
|---|---:|---:|---:|
| Three-seed mean return | 500.0 | 474.0 | 191.67 |
| Statistically separated states | 832 | 569 | 264 |
| Actor/target agreement there | **0.688** | 0.938 | 0.803 |
| Separated actionable states | 113 | **23** | 146 |
| Confident target/real balanced accuracy | 0.667 | 0.810 | 0.507 |
| All-actionable target/real balanced accuracy | 0.601 | 0.658 | 0.577 |
| Mean absolute target action margin | 0.415 | 0.770 | 3.022 |
| One-step state MSE | 0.088 | 0.039 | 0.045 |

The final checkpoint is directionally healthier than the predecessor: its
target margin no longer explodes, its all-actionable ordering improves, and it
solves two of the three diagnostic seeds. But the target-quality gate lacks
support because only 23 final states are both statistically separated and
simulator-actionable, not the required 100. The solved best checkpoint also
fails the `0.8` actor/target agreement condition. The fixed three-seed probe
cohort is substantially easier for the final policy than the 20-seed training
evaluation (`474.0` versus `357.75`), so it cannot override the behavioral
failure.

This rejects the missing start factor as a sufficient cause while retaining it
as a source-conforming correction. The result is consistent with its small
causal reach: it can temper target margins and alter the trajectory, but it
cannot prevent recurrent recovery failures. No seeds 1 or 2 are launched. The
next step returns to the critic-target source/data-flow audit; a further
training change requires a concrete divergence with broader reach than a
roughly 2.7% loss reweighting.

#### Preregistered slow-value-regularizer semantic audit

The next source audit finds such a divergence. In pinned reference
`agent.py`, both imagined and replay value objectives call
`value.loss(stop_gradient(slowvalue.pred()))`. The reference two-hot output's
`pred()` decodes the slow distribution to one scalar expectation, and its
`loss()` then re-encodes that scalar as a one- or two-bin target. Local code
instead uses cross-entropy against `softmax(critic_ema_logits)`, copying the
slow critic's complete probability distribution. These targets have the same
decoded mean but generally different entropy and gradients. Local comments
that describe a “distributional regularizer” obscured the mismatch.

This path has broad causal reach. Its coefficient is `1.0` beside every
imagined critic target and it is applied again inside the scale-`0.3` replay
value objective. A broad slow distribution can therefore teach the online
critic to preserve stale probability mass across many value bins, while the
reference teaches only the slow model's scalar value using the same sharp
two-hot observation model as the primary return target. This is directly
upstream of the overconfident, incorrectly ordered critic bootstrap found in
recovery states.

- **Question:** on trained CartPole latents, do the local full-distribution and
  reference decoded-mean/two-hot slow targets produce materially different
  regularizer gradients, or are they numerically interchangeable in practice?
- **Frozen cohort:** collect 4,096 post-transition states from one environment
  action stream generated independently of either checkpoint, with environment
  seed `17` and NumPy action seed `17`. Observe that identical true-state/action
  history through each checkpoint's posterior-mode RSSM. Measure the episode-
  coherent canary and start-weight canary at both best and final checkpoints.
- **Metrics:** slow-target total variation and entropy; cosine similarity and
  norm ratio between logit gradients `p_online - p_slow` (local) and
  `p_online - twohot(E_slow)` (reference); fraction with negative cosine; and
  each target's cross-entropy against the online critic. Preserve per-
  checkpoint results rather than pooling away best/final movement.
- **Decision gate:** authorize an isolated semantic correction only if mean
  target total variation is at least `0.25` and mean gradient cosine is below
  `0.8` in at least one final checkpoint. Otherwise reject this mismatch as
  numerically inert and continue the audit without a training run.
- **Stop rule:** this is read-only checkpoint analysis. Do not alter training
  or launch a canary until the measurements and source interpretation are
  recorded.

#### Slow-value-regularizer audit result and canary preregistration

The read-only probe is retained under
`experiments/2026-07-22_cartpole_slow_regularizer_probe/`. It used one fixed
4,096-state random-action history for all four checkpoints; every result has
history SHA-256 `765a551e4fd6a5a9398ff0a96de378e0d4d26b08c1d5e1d36801f00fe70449ac`,
confirming identical true states and actions. The results decisively pass the
predeclared causal-reach gate:

| Run/checkpoint | Target TV | Slow/ref entropy | Local/ref grad norm | Grad cosine | Negative cosine |
|---|---:|---:|---:|---:|---:|
| Coherent best 1,700 | 0.612 | 1.938 / 0.524 | 0.091 / 0.691 | 0.562 | 24.9% |
| Coherent final 3,500 | 0.512 | 1.834 / 0.565 | 0.058 / 0.532 | 0.577 | 17.7% |
| Start-weight best 1,700 | 0.618 | 2.117 / 0.531 | 0.101 / 0.676 | 0.538 | 8.1% |
| Start-weight final 3,500 | 0.494 | 1.807 / 0.567 | 0.075 / 0.538 | 0.664 | 3.7% |

Both finals exceed target total variation `0.25` and fall below gradient cosine
`0.8`. The reference gradient is also about seven to nine times larger in norm
than the local gradient. The difference is therefore not only target entropy:
local full-distribution matching often supplies a much weaker update and, on
the collapsed coherent final, points in the opposite half-space on 17.7% of
states. This authorizes one isolated semantic correction.

#### Preregistered decoded-mean slow-regularizer canary

- **Hypothesis:** copying the slow critic's broad distribution weakens or
  misdirects the value regularizer, allowing the online critic's expected
  action margins to drift and inflate. Matching the reference's decoded slow
  scalar through a fresh two-hot target will provide a stronger, semantically
  aligned anchor and improve recovery-state target stability.
- **Single causal variable:** for both imagined and replay value regularizers,
  replace only `softmax(slow_logits)` as the cross-entropy target with
  `twohot(twohot_expectation(slow_logits))`. Keep the slow model update rate,
  coefficient, primary lambda-return target, online target selection, start
  continuation weights, architecture, replay, optimizers, and all run settings
  fixed. Add an explicit authored target-mode field; historical snapshots that
  lack it retain full-distribution matching unless semantic migration is
  requested.
- **Mechanical gate:** tests must prove the decoded-mean target equals an
  ordinary two-hot scalar target, differs from a broad same-mean distribution,
  is detached, and is used by both imagination and replay paths. Focused/full
  tests, compile, supported type checks, and a multiprocess CPU smoke must pass.
- **Frozen run:** repeat the start-weight seed-0 canary exactly for 3,500
  updates and the same fixed evaluation cohort, changing only the slow-target
  mode. Preserve clean source, the one-thread host cap, and all checkpoint/
  manifest semantics.
- **Behavior gate:** reach mean evaluation at least `475`; after the first such
  evaluation, never fall below `300`; finish at least `400`; and keep the
  best-to-final gap at most `100`.
- **Boundary gate:** repeat the reset-seed `17`--`19`, 64-sample, 15-step
  policy-target probe on best and final. Final actor/target agreement must be at
  least `0.8`, all-actionable target/real balanced accuracy at least `0.60`,
  and at least 100 states must be both statistically separated and simulator-
  actionable. Fewer than 100 makes the boundary result underpowered rather than
  a pass.
- **Stop rule:** one seed only. Replicate no additional seed unless both gates
  pass. Failure retains the source correction but rejects this mismatch as a
  sufficient stability cause; do not combine value-head depth or a new replay
  intervention in the same run.

#### Decoded-mean slow-regularizer canary result

The correction is retained at source commit `81dc6ed`. It adds resume-safe
`critic_ema_target` semantics, uses one shared detached target builder for
imagined and replay value regularization, and preserves historical
`distribution` behavior while authored Hydra runs select `mean_twohot`.
Focused tests passed (`40`), the full fast suite passed (`195`), compile and
the supported scoped type check passed, and the one-update multiprocess CPU
smoke completed normally with the new mode resolved in its config.

The frozen seed-0 run completed normally under
`experiments/2026-07-22_cartpole_slow_mean_seed0_3500/`, with manifest run ID
`882d31a600d34af2839a6a5fcfc750fb` and MLflow run ID
`6cd5da56de444ee7aced1a6f1c814d50`. It used 3,500 learner updates, 21,450
environment steps, and 1,012.1 seconds on ROCm. The manifest records clean
source `81dc6ed`, `critic_ema_target=mean_twohot`, normal collector shutdown,
and distinct best/final checkpoints.

Behavior fails before the stability question: the policy never reaches the
`475` solve threshold. It peaks at only `409.4` at update 1,200 and ends at
`179.7`:

| Update | 900 | 1,000 | 1,200 | 1,400 | 1,900 | 2,400 | 2,900 | 3,100 | 3,500 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean return | 318.35 | 399.55 | **409.4** | 286.45 | 255.0 | 219.4 | 167.65 | 347.25 | **179.7** |

The policy-target probe under
`experiments/2026-07-22_cartpole_slow_mean_policy_q_probe/` also fails:

| Metric | Best, update 1,200 | Final, update 3,500 |
|---|---:|---:|
| Three-seed mean return | 391.67 | 183.67 |
| Statistically separated states | 466 | 106 |
| Actor/target agreement there | 0.845 | **0.443** |
| Separated actionable states | 164 | **84** |
| Confident target/real balanced accuracy | **0.455** | **0.407** |
| All-actionable target/real balanced accuracy | 0.513 | **0.443** |
| Mean absolute target action margin | 0.177 | 0.985 |
| One-step state MSE | 0.119 | 0.158 |

Best and final target ordering are below chance or near chance; final also
falls below the actor/target transfer threshold and the 100-state support
floor. The failure is therefore coupled target-quality and actor-transfer
degradation, not a stable critic whose policy head alone fails to fit.

A post-run fixed-history mechanism check under
`experiments/2026-07-22_cartpole_slow_mean_regularizer_probe/` confirms the
intended code path operated but did not make the slow distribution itself
point-like. Slow-target entropy is `2.123` at best and `1.736` at final, while
the re-encoded reference targets have entropy `0.466` and `0.486`. At final,
the hypothetical full-distribution and decoded-mean targets still have total
variation `0.476` and gradient cosine `0.466`. Thus the intervention continued
to deliver the materially different reference gradient throughout training;
the negative result is not explained by the two contracts becoming equivalent.

This rejects decoded-mean slow regularization as a sufficient stability fix
and stops seeds 1 and 2. The source correction remains necessary for a faithful
replication, but this isolated run deliberately preserved the historical split
learning rates and legacy optimizer ownership. It therefore does not establish
that the complete reference optimizer-plus-regularizer contract fails; nor
does it authorize stacking that contract without a new, explicit integration
hypothesis. The next audit should examine critic-target change rate versus
actor update rate and joint optimizer ownership, because the stronger critic
anchor exposes final actor/target agreement `0.443` under the current
actor/critic rate ratio `3e-5/8e-5`.

## Reliability follow-up

### Preregistered fixed-history actor/target time-scale audit

The decoded-mean canary failed with both incorrect learned target ordering and
late actor/target disagreement, but those endpoint readouts do not say which
boundary broke first. A new training run would therefore be premature.

- **Question:** on one checkpoint-independent state history, does the learned
  policy-conditioned target become wrong before the actor stops following it,
  or does actor lag appear while the target is still useful?
- **Frozen history:** drive CartPole with the decoded-mean canary's best
  checkpoint on reset seeds 17--18. Evaluate periodic updates 500, 1,000,
  1,500, 2,000, 2,500, 3,000, and 3,500 plus best update 1,200 on exactly those
  states and previous actions. Use deterministic posterior modes, the
  checkpoint's 15-step policy-conditioned lambda-return target, 64 Monte Carlo
  samples per forced first action, and the existing 30-step trusted-real action
  score. No weights change.
- **Primary readouts:** at every checkpoint, report the number of target action
  differences exceeding 1.96 combined standard errors, actor/target agreement
  there, target/real balanced accuracy on confident actionable states, action
  preference counts, target margin, and one-step model error. Across adjacent
  checkpoints, report target-preference and actor-action flip rates on the
  aligned `(episode, timestep)` rows.
- **Interpretation:** target/real accuracy falling below `0.592` while
  actor/target agreement remains at least `0.8` selects target-quality failure
  as the first broken boundary. Actor/target agreement below `0.8` while target
  accuracy remains at least `0.592` selects actor lag. Both failing at the first
  adequately supported checkpoint selects a coupled optimization failure.
  Fewer than 100 confident actionable rows at a checkpoint is descriptive, not
  categorical.
- **Stop rule:** these eight checkpoint views and two reset seeds only. Do not
  launch training or change rates after seeing the result; use it to decide
  whether an integrated current-stack reference optimizer canary is justified.

The diagnostic extends the existing fixed-history probe to expose the exact
sampled policy-conditioned target alongside its deterministic branch-max Q
readout. The generated rows retain the standard errors and confidence rule so
uncertain Monte Carlo preferences cannot masquerade as categorical labels.

### Fixed-history actor/target time-scale result

The read-only audit completed under
`experiments/2026-07-22_cartpole_slow_mean_fixed_history_target_timeline/` from
clean source `812cec5`. Both reset seeds follow the decoded-mean canary's best
actor for a shared 830-state history (mean return `415.0`); every target
checkpoint therefore sees identical observations and previous actions. The
common Monte Carlo seed also reduces sampling noise between checkpoints.

| Target update | Confident | Confident actionable | Actor/target | Target/real balanced | Mean abs target delta | Mean delta SE |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 539 | **116** | **0.314** | **0.501** | 1.483 | 0.530 |
| 1,000 | 224 | 66 | **0.638** | **0.169** | 0.437 | 0.216 |
| 1,200 best | 312 | 82 | 0.833 | **0.420** | 0.156 | 0.066 |
| 1,500 | 410 | 80 | **0.788** | **0.553** | 0.721 | 0.292 |
| 2,000 | 335 | 81 | 0.904 | 0.641 | 1.448 | 0.571 |
| 2,500 | 407 | **104** | 0.980 | 0.620 | 3.908 | 1.567 |
| 3,000 | 302 | 71 | 0.954 | 0.692 | 3.271 | 1.617 |
| 3,500 | 211 | 24 | 0.972 | 0.783 | 2.067 | 1.493 |

Only updates 500 and 2,500 meet the preregistered 100-confident-actionable
support floor. The first selects a coupled early optimization failure: neither
target quality nor actor tracking passes. By update 2,500 both pass. The
underpowered intervening rows show the order rather than support a categorical
gate claim: actor tracking crosses `0.8` at the best checkpoint while target
ordering is still poor, then both improve on this fixed good-policy history.

Adjacent actor-action flip rates are `0.633`, `0.129`, `0.190`, `0.067`,
`0.265`, `0.393`, and `0.075`. Raw sampled target preferences flip `0.511`,
`0.286`, `0.237`, `0.178`, `0.218`, `0.248`, and `0.208`. However, among rows
that are statistically separated at both adjacent checkpoints, target flips
fall from `0.425` at 500-to-1,000 to `0.043` at 1,200-to-1,500 and exactly zero
for every later comparison. The large later raw flip rates therefore occur in
uncertain rows, not in strong target reversals. One-step prior x MSE also falls
from `0.371` at update 500 to `0.011--0.074` after update 1,500 on this cohort.

This rejects critic-target change rate versus actor update rate as the main
late-collapse mechanism on replay-supported good trajectories. It also makes
the final checkpoint's earlier on-policy result more informative: on its own
final-policy histories, actor/target agreement was `0.443` and target/real
balanced accuracy `0.407`, whereas on the fixed best-policy histories the same
weights give `0.972` agreement and a descriptive `0.783` target/real score.
Useful behavior has not been overwritten globally. The learned system retains
it on the old trajectory but the deployed policy induces a different state and
latent-history distribution where both boundaries fail.

The result does not justify an integrated optimizer rerun: equal rates could
change early learning, but the suspected late actor lag is absent on the fixed
cohort. The next source audit should instead inspect how imagined actor updates,
replay sampling, and posterior/prior state construction constrain the policy to
data-supported histories. Any intervention must distinguish a genuine local
implementation mismatch from ordinary model exploitation before training.

### Preregistered decoded-mean continuation-boundary audit

CartPole emits reward one on every live transition, so imagined action
preference is dominated by predicted continuation and bootstrapped value. The
fixed-history result shows that final weights remain useful on the best-policy
trajectory but fail on their own induced trajectory. Before attributing that
failure to generic model exploitation, isolate whether survival prediction is
the first broken model boundary.

- **Frozen cohort:** compare decoded-mean best update 1,200 and final update
  3,500 on deterministic actor episodes for reset seeds 17--36. At every state,
  record the physical terminal label, distance to terminal, posterior-mode
  continuation, and 64 posterior/prior categorical samples. Do not alter policy
  extraction, weights, horizon, or thresholds.
- **Primary readouts:** mean return; posterior and prior effective discount on
  terminal versus live rows; physical-failure ROC AUC and balanced accuracy;
  distance-to-terminal strata; posterior/prior transport RMS; and latent KL or
  posterior-mode support where available.
- **Interpretation:** posterior failure AUC at least `0.8` with prior below
  `0.6` selects posterior-to-prior transport. Both below `0.6` selects a learned
  continuation/supervision failure. Both at least `0.8` rejects continuation as
  the first broken target component and shifts attention to reward/value. Values
  between the thresholds are weak and do not select an intervention.
- **Stop rule:** these two checkpoint views, 20 episodes, and 64 samples only.
  Do not train or rebalance continuation after seeing the result; compare it
  with the already-recorded actor/target and one-step-state boundaries first.

### Decoded-mean continuation-boundary result

The frozen audit completed under
`experiments/2026-07-22_cartpole_slow_mean_continuation_probe/` from clean
source `df40377`. Best update 1,200 averages `379.15` over seeds 17--36 (17
physical failures and three time-limit truncations); final update 3,500 averages
`185.65` with 20 physical failures.

| Metric | Best 1,200 | Final 3,500 |
|---|---:|---:|
| Posterior failure ROC AUC | 0.910 | **0.987** |
| Prior failure ROC AUC | 0.887 | **0.985** |
| Posterior terminal / live effective discount | 0.9948 / 0.9970 | **0.9315 / 0.9835** |
| Prior terminal / live effective discount | 0.9945 / 0.9970 | **0.9266 / 0.9824** |
| Posterior-to-prior transport RMS, all rows | 0.00030 | 0.00199 |
| Posterior-to-prior transport RMS, terminal rows | 0.00204 | 0.00539 |
| Balanced accuracy at half task discount | 0.500 | 0.500 |

Both channels exceed the preregistered `0.8` ordering gate at both checkpoints,
so posterior-to-prior transport and failure-risk ordering are rejected as the
first broken target component. The final head is especially informative: it
becomes *better* at ranking imminent failure while behavior falls by roughly
194 return. Transport error is also tiny relative to terminal label error.

This does not mean continuation is calibrated correctly. No terminal row
crosses the half-discount decision boundary; the final posterior still assigns
actual failures effective discount `0.9315`. Conversely, it underpredicts the
live target `0.997` at `0.9835`; that roughly 1.35-percentage-point per-step gap
compounds across imagination. The distance profile is monotonic and useful
(`0.9315` at failure versus `0.9858` at distance ten or greater), but it acts as
a smooth hazard score rather than a physical termination probability. This
matches the earlier continuation-capacity, stream-exposure, and class-balance
experiments: the signal is present, while naïve balancing destroys the natural
probability needed as a discount and does not stabilize behavior.

The current failure therefore remains value-target quality on recovery states,
not missing reward or latent transport. The existing policy-Q decomposition
supports that ordering: on final-policy actionable states, one-step critic-
bootstrap balanced accuracy is `0.407`, while the learned three-step model-
return action delta is only `0.00295` in mean absolute magnitude and has
balanced accuracy `0.356`. The 15-step policy target's much larger `0.985` mean
action margin is consequently dominated by bootstrapped value geometry rather
than immediate learned reward differences. The next audit must focus on why
the critic is self-consistent but wrong on recovery histories; repeating
continuation-head capacity or class-weight experiments is not authorized.

### Preregistered online-versus-slow critic boundary audit

The online critic supplies the authored policy-conditioned target; its Polyak
copy is updated by only two percent of each online parameter change per learner
step. Comparing them on identical actor trajectories distinguishes transient
online value noise from a stable wrong value geometry without changing
training.

- **Frozen cohort:** for decoded-mean best update 1,200 and final update 3,500,
  drive the checkpoint's deterministic actor on reset seeds 17--21. Evaluate
  every state twice with common random numbers: once using `critic` and once
  using `critic_ema` for the 15-step, 64-sample policy-conditioned lambda-return
  target. Keep the actor, encoder/RSSM, reward/continuation heads, trusted
  30-step real rollout, and all confidence rules identical.
- **Primary readouts:** confident target/real balanced accuracy, confident
  actor/target agreement, confident actionable support, target preference
  histogram, action margin, and one-step critic-bootstrap ordering. The online
  cells are paired replications of the earlier three-seed endpoint probe, not a
  replacement for its recorded result.
- **Interpretation:** slow target/real balanced accuracy at least `0.592` and at
  least `0.10` above online, with 100 confident actionable rows, selects noisy
  online value drift. Both critics below `0.592`, or a gap below `0.10`, selects
  a stable target-grounding failure shared across the smoothing window. A slow
  critic that improves ordering but loses actor agreement is descriptive
  because the actor was not trained against that target.
- **Stop rule:** four paired cells, five episodes each, and 64 samples only. Do
  not switch training to the slow target or launch a new run from this result;
  first compare it with the historical slow-target canaries and the pinned
  reference's authored online-target semantics.

### Online-versus-slow critic boundary result

The four paired cells completed under
`experiments/2026-07-22_cartpole_slow_mean_online_slow_critic_probe/` from clean
source `ba64af5`. Each online/slow pair follows the same actor for the same five
reset seeds and uses common random numbers for 64 sampled returns per forced
first action.

| Checkpoint | Critic | Confident actionable | Actor/target | Target/real balanced | One-step bootstrap/real balanced | Mean abs target delta |
|---|---|---:|---:|---:|---:|---:|
| Best 1,200 | Online | 313 | 0.850 | **0.417** | 0.569 | 0.167 |
| Best 1,200 | Slow EMA | 315 | 0.862 | **0.423** | 0.533 | 0.152 |
| Final 3,500 | Online | 140 | **0.436** | **0.359** | **0.418** | 0.998 |
| Final 3,500 | Slow EMA | 143 | **0.469** | **0.390** | **0.437** | 0.992 |

All cells clear the 100-confident-actionable support floor. The slow critic
improves final target/real balanced accuracy by only `0.031`, remains `0.202`
below the `0.592` gate, and does not restore actor tracking. At the best
checkpoint its target accuracy differs from online by only `0.006`. Target
margins are also nearly identical within each checkpoint. The preregistered
result is therefore a stable target-grounding failure shared across the Polyak
smoothing window, not transient online-critic jitter.

This agrees with the pinned reference's online-target semantics and rejects
switching lambda returns back to the slow critic. It also changes the optimizer
decision relative to the earlier fixed-history audit. That audit rejected late
actor lag on good histories; this paired audit now shows that both temporal
views of the critic are stably wrong on recovery histories under the legacy
split-rate contract. The remaining source-conforming optimizer question is not
whether the actor should chase a moving critic, but whether replay-value
representation grounding can form a correct critic under coherent shared
optimization after the newer signal corrections.

### Preregistered corrected-stack optimizer integration canary

The earlier reference-optimizer canary at source `90946cc` cannot answer this
integration question: it predates episode-coherent collector updates,
replay-start continuation weighting, and decoded-mean slow-value
regularization. Those corrections change the behavior distribution, imagined
loss weights, and every critic regularizer target. Repeating the optimizer
contract now is an integration acceptance test, not a causal retry or a claim
that the earlier negative result was invalid.

- **Hypothesis:** with the corrected data and value signals, equal optimizer
  time scales plus replay-value representation gradients prevent the online
  and slow critics from converging together to the wrong recovery-state
  geometry.
- **Causal bundle:** from the decoded-mean seed-0 canary, change only the
  authored optimizer contract: set world-model, actor, and critic rates to
  `4e-5`; ramp them linearly from zero over 1,000 updates; and enable the
  reference replay-value gradient into observed encoder/RSSM features through
  the already-validated combined backward path. Preserve reference RSSM,
  stream replay, episode-coherent collection, start weighting, decoded-mean
  regularization, batch 8, sequence 16, burn-in 4, replay ratio 16, entropy
  `1e-3`, and every environment/evaluation setting.
- **Behavioral gate:** reach mean return `450`; never fall below `300`
  afterward; final at least `400`; best-to-final gap at most `100`.
- **Boundary readouts:** regardless of behavior, retain the full curve and run
  the 20-seed continuation audit plus the five-seed, 64-sample online policy-
  target probe on best and final. Final confident target/real balanced accuracy
  must reach `0.592`, actor/target agreement `0.8`, and at least 100 confident
  actionable rows to pass the value boundary. Continuation is diagnostic and
  cannot rescue a failed behavior/value gate.
- **Interpretation:** a pass shows that the corrected components work as an
  integrated reference optimizer system but does not identify which optimizer
  dimension caused it. A failure rejects this corrected-stack bundle as
  sufficient and ends optimizer-contract retries; the next step must be a
  model/target architecture decision rather than another learning-rate tweak.
- **Stop rule:** one seed for 3,500 learner updates. No rate tuning or seeds 1
  and 2 unless both behavioral and value-boundary gates pass.

Interrupted manifests correctly record `status: interrupted` and evaluation
history, but incorrectly retain `progress.train_step: 0` and `env_steps: 0`.
Fix this bookkeeping issue separately from the learning experiment so it does
not alter the scientific intervention.

### Corrected-stack optimizer integration result

The preregistered seed-0 run completed normally under
`experiments/2026-07-22_cartpole_corrected_reference_optimizer_seed0_3500/`
from clean source `ff9de4e7e4ac438355b6f046400fc56dc676a36b`.
Its manifest run ID is `34df3ce8fa5045bfac79a3a92420a390` and its MLflow
run ID is `ea5d8acdfe38448f900ca96d9009a320`. It consumed 3,500 learner
updates and 21,490 environment transitions in 920.8 seconds, then recorded a
`completed` lifecycle with the `max_train_steps` stop reason.

The run used the exact preregistered bundle: reference RSSM and heads, stream
replay, episode-coherent collection, replay-start continuation weighting,
decoded-mean slow-value regularization, equal `4e-5` world-model/actor/critic
rates, 1,000-update linear warmup, and replay-value representation gradients.
Batch size remained 8, sequence length 16, burn-in 4, replay ratio 16, actor
entropy `1e-3`, one collector, and 20 deterministic evaluation episodes every
100 updates.

The complete evaluation-return curve was:

```text
step:    100  200  300  400  500  600  700  800  900 1000 1100 1200
return:  9.4 15.9 9.35 9.35 9.35 9.35 9.35 9.35 9.35 9.35 9.35 13.95

step:   1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400
return: 9.35 9.35 9.35 9.35 10.1  9.4 83.05 79.0 82.7 75.75 64.6 32.65

step:   2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500
return: 24.95 20.1 17.6 20.7 25.15 16.3 40.8 30.5 36.35 154.4 272.5
```

The run never reached the required return of 450, ended at 272.5, and showed
an unstable rise-collapse-recovery trajectory. The behavior gate therefore
fails before considering any diagnostic. Best and final both point to update
3,500; direct tensor comparison verified that encoder, world model, actor,
online critic, and slow critic weights in `checkpoint_best.pt` and
`checkpoint_final.pt` are identical. The required endpoint probes were
therefore run once rather than duplicating identical weights.

The 20-seed continuation audit under
`experiments/2026-07-22_cartpole_corrected_reference_optimizer_continuation_probe/`
observed mean return 258.9 across 5,178 transitions, with 16 physical failures
and four time-limit truncations.

| Continuation readout | Final 3,500 |
|---|---:|
| Posterior expected failure ROC AUC | **0.675** |
| Posterior mode failure ROC AUC | **0.680** |
| Prior expected failure ROC AUC | **0.603** |
| Posterior terminal / live effective discount | **0.9895 / 0.9909** |
| Prior terminal / live effective discount | **0.9898 / 0.9899** |
| Posterior-to-prior RMS, all / terminal rows | 0.00117 / 0.00139 |
| Balanced accuracy at half task discount | 0.500 |

Unlike the legacy split-rate decoded-mean final checkpoint, whose posterior
and prior failure AUCs were about 0.987 and 0.985, this bundle's continuation
predictions are nearly constant and only weakly order failure risk. Transport
remains small, but actual terminal transitions retain approximately 0.99
effective discount instead of zero. Equal rates and replay-value gradients did
not merely fail to solve the old critic geometry; in this seed they also
coincide with worse learned survival discrimination.

The five-seed, 64-sample online-critic policy-target audit under
`experiments/2026-07-22_cartpole_corrected_reference_optimizer_policy_q_probe/`
followed 1,777 actor states and found 223 actionable states. Returns were
355.4 mean, 435 median, 119 minimum, and 500 maximum, with one of five episodes
solved.

| Policy/value readout | Final 3,500 |
|---|---:|
| Actor / trusted-real balanced accuracy | 0.502 |
| One-step critic-bootstrap / trusted-real balanced accuracy | **0.500** |
| Three-step full-Q / trusted-real balanced accuracy | **0.498** |
| Policy target / trusted-real balanced accuracy, all actionable | 0.533 |
| Actor / policy-target agreement, all states | 0.843 |
| Mean absolute policy-target action delta | 0.105 |
| Confident states / confident actionable states | 426 / **53** |
| Confident actor / policy-target agreement | **0.979** |
| Confident policy-target / trusted-real balanced accuracy | **0.567** |
| One-step state MSE | 0.0616 |

The one-step critic bootstrap prefers action 1 on all 1,777 states. The actor
tracks statistically separated policy targets extremely well, but those
targets remain below the `0.592` trusted-real gate and provide only 53
confident actionable rows, below the required support floor of 100. Thus the
value-boundary gate also fails: strong actor/target agreement cannot rescue a
near-chance and under-supported target.

This is a decisive negative canary, not a reason to add seeds. The corrected
reference-optimizer bundle improves the endpoint over the older reference
optimizer canary (roughly 170 final) but remains materially worse than the
legacy split-rate decoded-mean peak of 409, never solves CartPole, and does not
stabilize recovery. Per the preregistered stop rule, seeds 1 and 2 are cancelled
and optimizer-contract retries end here. The next bounded decision must audit
the model or target architecture that creates the replay/recovery latent and
value geometry; another learning-rate, warmup, or target-smoothing change is
not evidence-selected.

### State-reconstruction aggregation source audit

The next audit found a direct world-model target-scale mismatch against pinned
reference `e3f0224`. The reference vector decoder produces a per-coordinate
symlog MSE and wraps it in `Agg(..., jnp.sum)` over the observation event
dimensions. Its state reconstruction term is therefore
`sum_i (prediction_i - symlog(target_i))^2` for each transition. Local
`compute_wm_loss()` instead computes
`0.5 * mean_i (prediction_i - symlog(target_i))^2`.

For CartPole's four-dimensional state, the local observation-grounding loss and
its gradient are exactly `1 / (2 * 4) = 1/8` of the pinned equation for
identical predictions and targets. This is not a logging-only reduction: the
scaled term is added beside reward, continuation, dynamics KL, and
representation KL before the shared world-model backward pass, so it weakens
the gradient reaching the state encoder, recurrent model, posterior head, and
decoder. Pixel reconstruction already uses a sum and is not affected.

The completed corrected-optimizer canary makes the scale difference concrete.
Across its 140 logged batches, the local state term averages `0.0271`, compared
with reward `1.285`, continuation `0.181`, dynamics KL `1.016`, and total world-
model loss `2.694`. The pinned aggregation would make the same recorded state
errors average `0.217`. At update 1,000 the local state term is `0.0715`; the
reference-equivalent value is `0.572`. Exact gradient interaction cannot be
inferred from scalar loss alone, but the deterministic eightfold component
gradient error is large enough to change shared representation optimization.

This mismatch is more proximal than the remaining depth/normalization
differences. Frozen supervision has repeatedly shown that the posterior latent
contains useful but asymmetric value information, while online recovery
targets fail and the corrected optimizer canary's continuation predictor loses
failure ordering. Underweighting the only direct physical-state reconstruction
target can allow KL and self-generated value objectives to dominate precisely
the small recovery-state distinctions that CartPole control needs. It is still
a hypothesis about learning outcome, not proof that stronger reconstruction
will stabilize the actor.

### Preregistered reference state-aggregation canary

- **Hypothesis:** restoring the authored sum-of-squared symlog state objective
  strengthens observation grounding enough for the corrected reference stack
  to preserve failure-risk and recovery-value geometry instead of converging to
  a near-constant continuation estimate and one-action critic bootstrap.
- **Causal variable:** add an explicit state-loss semantic mode. New authored
  runs use `reference_sum`; historical checkpoints/config snapshots without
  the field infer `legacy_half_mean`. Change only the state-vector reduction
  from half-mean to sum. Do not alter pixel, reward, continuation, KL, replay,
  optimizer, actor/value architecture, batch, sequence, ratio, entropy, or
  evaluation semantics.
- **Mechanical gate:** focused tests must prove that a four-coordinate state
  has exactly eight times the legacy loss and state-prediction gradient under
  `reference_sum`; every non-state loss is bitwise or numerically unchanged;
  authored Hydra selects the reference mode; old snapshots infer legacy mode;
  resume rejects an unapproved semantic migration; and checkpoint snapshots
  preserve the selected mode. Then run the full fast suite, compile, scoped
  type gate, and one-update multiprocess CPU smoke.
- **Frozen run:** repeat the clean seed-0, 3,500-update corrected reference-
  optimizer command and fixed 20-episode evaluation schedule from run
  `34df3ce8fa5045bfac79a3a92420a390`, changing only state-loss aggregation.
  Retain batch 8, sequence 16, burn-in 4, replay ratio 16, one collector, and
  the one-thread host resource cap.
- **Behavioral gate:** reach mean return `450`; never fall below `300`
  afterward; final at least `400`; best-to-final gap at most `100`.
- **Boundary gate:** regardless of behavior, run the final 20-seed continuation
  audit and five-seed, 64-sample online policy-target probe. Final posterior
  and prior failure ROC AUC must each reach `0.8`. The policy target must have
  at least 100 confident actionable rows, target/trusted-real balanced accuracy
  at least `0.592`, and actor/target agreement at least `0.8`.
- **Interpretation:** passing both gates selects state reconstruction scale as
  the first sufficient repair in the corrected stack, subject to replication.
  Improved continuation but failed value/behavior localizes the next boundary
  downstream of observation grounding. Failure ends aggregation retries and
  returns to the coupled encoder/posterior architecture audit.
- **Stop rule:** one seed only. Do not add seeds or combine normalized MLPs,
  posterior depth, reward-head depth, deterministic-state width, larger
  batches, or longer sequences unless the preregistered gates pass.

### Reference state-aggregation mechanical gate result

The implementation gate is complete at source `c228d2a`. Authored Hydra runs
select `reference_sum`; current snapshots preserve their selected mode, and
historical snapshots without the field infer `legacy_half_mean` unless semantic
migration is explicit. The focused four-coordinate regression proves exactly
eight times the legacy state loss and state-prediction gradient under
`reference_sum`, while pixel, reward, continuation, dynamics, and representation
losses remain unchanged.

CPU validation passed with 205 fast tests, compile checks over `src`, `tests`,
and `scripts`, the scoped Pyright gate with zero errors, and the supported
one-update multiprocess dry run. The smoke run composed `reference_sum`, loaded
two episodes, published model version 0 to its collector, and exited normally.
No 3,500-update canary was started during this mechanical gate; the frozen
seed-0 state-aggregation run above remains the next authorized experiment.

### Reference state-aggregation canary result

The preregistered seed-0 canary completed normally under
`experiments/2026-07-22_cartpole_reference_state_sum_seed0_3500/` from clean
source `f26f7c530461841a704dc17b15348caefcdc4216`. Its manifest run ID is
`bfdfe890249a42448758a51135715e14` and its MLflow run ID is
`82646ca7f51c448cb54e9b56da4079b7`. It consumed 3,500 learner updates and
21,829 environment transitions in 957.7 seconds. The command changed only
`train.state_loss_mode=reference_sum` relative to the corrected-reference-
optimizer canary and retained the fixed seed, evaluation schedule, resource
cap, and all other authored parameters.

The complete deterministic 20-episode evaluation curve was:

```text
step:    100  200  300  400  500  600  700  800  900 1000 1100 1200
return:  9.4 10.85 9.35 9.35 9.35 9.35 9.35 9.35  9.4 36.2 78.9 75.55

step:   1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400
return: 85.1 83.4 70.7 81.8 89.85 156.85 84.05 50.4 40.1 31.35 25.85 28.9

step:   2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500
return: 79.1 118.7 149.4 174.45 198.25 248.05 475.65 498.55 500.0 399.95 289.85
```

Restoring reference state aggregation materially changes learning. The matched
half-mean run remained near the floor through update 1,800 and never reached
450; the sum-loss run left the floor by update 1,000 and reached 500 at update
3,300. This is causal evidence that the eightfold state-gradient mismatch was
a real implementation defect and that the corrected stack can learn a solved
CartPole policy.

It is not yet a reliable solution. After first crossing 450 at update 3,100,
the run fell to 399.95 and then 289.85. It therefore violates the frozen
no-post-solve-return-below-300 rule, the final-return-at-least-400 rule, and the
best-to-final-gap-at-most-100 rule: the best/final gap is 210.15. This is an
oscillatory rise-collapse-recovery-collapse trajectory, not a monotonic failure
to acquire the task.

The required 20-seed final-checkpoint continuation audit is retained under
`experiments/2026-07-22_cartpole_reference_state_sum_continuation_probe/`.
Seeds 17--36 produced mean return 294.65 across 5,893 transitions, with 20
physical failures and no time-limit truncations.

| Continuation readout | Final 3,500 |
|---|---:|
| Posterior expected failure ROC AUC | **0.617** |
| Posterior mode failure ROC AUC | **0.600** |
| Prior expected failure ROC AUC | **0.577** |
| Posterior terminal / live effective discount | **0.9954 / 0.9956** |
| Prior terminal / live effective discount | **0.9942 / 0.9946** |
| Posterior-to-prior RMS, all / terminal rows | 0.00187 / 0.00178 |
| Balanced accuracy at half task discount | 0.500 |

The continuation gate fails decisively. Even after the actor has demonstrated
solved behavior, the learned continuation channel is nearly constant and does
not represent physical termination. State-loss scale alone does not repair
failure-risk semantics.

The five-seed, 64-sample online-critic policy-target audit is retained under
`experiments/2026-07-22_cartpole_reference_state_sum_policy_q_probe/`. It
followed 1,495 actor states and 746 trusted-actionable states. Returns were
299.0 mean, 308 median, 259 minimum, and 333 maximum; none of the five episodes
solved.

| Policy/value readout | Final 3,500 |
|---|---:|
| Actor / trusted-real balanced accuracy | 0.613 |
| One-step critic-bootstrap / trusted-real balanced accuracy | 0.591 |
| Three-step full-Q / trusted-real balanced accuracy | 0.529 |
| Policy target / trusted-real balanced accuracy, all actionable | 0.551 |
| Actor / policy-target agreement, all actionable | 0.618 |
| Mean absolute policy-target action delta | 0.509 |
| Confident states / confident actionable states | 1,200 / **600** |
| Confident actor / policy-target agreement | **0.717** |
| Confident policy-target / trusted-real balanced accuracy | **0.566** |
| One-step state MSE | 0.3345 |

The state correction expands confident actionable target support from 53 rows
in the matched half-mean run to 600, so the imagined action target is no longer
merely under-separated. However, its confident trusted-real accuracy remains
below the required 0.592 and actor/target agreement remains below 0.8. The
target prefers action 0 on 1,302 of 1,495 states while the actor is roughly
balanced, and the target/real action-margin correlation is effectively zero
(`0.002`). Stronger observation reconstruction creates usable policy diversity
without making the learned target reliably correspond to real counterfactual
outcomes.

This is a partial causal success and a frozen-gate failure. The legacy state
aggregation is rejected for authored runs, but seeds 1 and 2 are not launched:
one seed is enough to show that the isolated repair is not sufficient for
stability. Per the preregistration, aggregation retries end here. The next
bounded intervention is the coupled reference vector encoder/posterior
architecture audit; optimizer, replay, warmup, target-smoothing, and further
state-loss tuning remain frozen.

### Coupled reference observation-posterior source audit

Pinned source `e3f0224` applies three complete normalized hidden transforms to
vector observations. Each transform is `Linear(units) -> RMSNorm(eps=1e-4) ->
SiLU`, including the third layer's normalization and activation. Local
`StateOnlyEncoder` instead uses three linear projections with SiLU only after
the first two. It has no normalization and exposes the third layer's raw
affine output as the observation token. The authored `encoder_mlp_n_layers=3`
field does not change this hard-coded topology.

The mismatch continues immediately at the posterior boundary. Pinned RSSM
`obslayers=1` concatenates the deterministic state and encoder tokens, then
applies `Linear(hidden) -> RMSNorm(eps=1e-4) -> SiLU -> Linear(stoch*classes)`.
Local `compute_posterior()` sends the concatenation directly through one linear
logit projection. Thus the current posterior has no learned normalized mixing
layer between a 512-dimensional recurrent state and the 128-dimensional token
in the state-sum canary.

These are one coupled representational boundary, not two independent training
knobs: the posterior consumes the encoder's output directly, and neither
intermediate topology matches a pinned model. The state-sum result makes this
the earliest remaining shared failure point. The actor can reach return 500,
but final posterior continuation AUC is only 0.617 and the imagined policy-
target margin has effectively zero correlation with real counterfactual
margin. A richer normalized observation-to-latent map could preserve the small
terminal and recovery distinctions that the stronger reconstruction gradient
now supplies. This remains a hypothesis; the state-sum canary does not prove
that encoder/posterior capacity is the cause of late collapse.

### Preregistered coupled reference observation-posterior canary

- **Hypothesis:** matching the pinned vector encoder and one-hidden-layer
  posterior topology will turn the state-sum canary's newly useful observation
  grounding into stable latent failure-risk and action-value geometry.
- **Causal variable:** add explicit, checkpointed architecture semantics.
  `vector_encoder_mode=reference` uses three
  `Linear -> RMSNorm(eps=1e-4) -> SiLU` transforms;
  `posterior_head_layers=1` inserts one
  `Linear -> RMSNorm(eps=1e-4) -> SiLU` transform before posterior logits.
  Historical snapshots and field-less checkpoints retain/infer the old raw
  three-linear encoder and direct posterior projection. Change both modes
  together because the pinned observation-posterior boundary contains both.
- **Frozen variables:** retain source-state aggregation, seed 0, 3,500 learner
  updates, 20 deterministic evaluation episodes every 100 updates, `d_hidden`
  128, four recurrent blocks, batch 8, sequence 16, burn-in 4, replay ratio 16,
  one collector, equal `4e-5` rates, 1,000-update optimizer warmup, and every
  optimizer/replay/continuation/actor/value setting from run
  `bfdfe890249a42448758a51135715e14`. Do not also change model width, reward,
  actor, critic, decoder, dynamics, or continuation-head topology.
- **Mechanical gate:** prove the exact reference module sequence and epsilon,
  preserved legacy parameter layouts, encoder and posterior gradient flow,
  authored-Hydra selection, historical snapshot fallback, no-snapshot
  architecture inference, and resume rejection of an unapproved semantic
  migration. Then run focused tests, the full fast suite, compile, scoped
  Pyright, and a one-update multiprocess CPU smoke.
- **Behavioral gate:** reach mean return 450; never fall below 300 afterward;
  final at least 400; best-to-final gap at most 100.
- **Boundary gate:** regardless of behavior, run the same final 20-seed
  continuation audit and five-seed, 64-sample online policy-target probe.
  Posterior and prior failure ROC AUC must each reach 0.8. The policy target
  must have at least 100 confident actionable rows, target/trusted-real
  balanced accuracy at least 0.592, and actor/target agreement at least 0.8.
- **Interpretation:** passing both gates selects the complete observation-
  posterior boundary for replication. Improved latent gates but unstable
  behavior localizes the next defect downstream. Failure ends normalized
  observation-posterior retries and selects a fresh source-equation audit
  rather than width or optimizer tuning.
- **Stop rule:** one seed only. Do not add seeds or proceed to Pong unless both
  frozen gates pass.

### Coupled reference observation-posterior mechanical gate result

The implementation gate is complete at source `e90a070`. Authored Hydra runs
select `vector_encoder_mode=reference` and `posterior_head_layers=1`; both
values are saved in config snapshots and restored on resume. Historical
snapshots without either field retain the old vector encoder and direct
posterior projection. Snapshot-less checkpoints infer the modes from module
keys, while pixel-only historical encoders correctly infer a legacy inactive
vector branch.

Focused tests prove three complete `Linear -> RMSNorm(eps=1e-4) -> SiLU`
vector transforms, one complete normalized posterior hidden transform, the
unchanged legacy parameter keys, and nonzero gradients from posterior logits
through every new posterior and encoder parameter to both the observation and
deterministic state. At the frozen `d_hidden=128` width, this is not an
uncontrolled parameter-capacity increase: encoder parameters rise from 33,664
to 34,048, posterior parameters fall from 164,096 to 115,200 because of the
128-unit bottleneck, and the combined encoder/world-model parameter count falls
from 1,284,100 to 1,235,588.

Validation passed with 213 fast tests, compile checks over `src`, `tests`, and
`scripts`, zero errors in the repository's scoped Pyright gate, and a supported
one-update multiprocess CPU smoke. The smoke composed both reference modes,
collected two episodes, published model version zero, stopped its collector,
and exited normally. The frozen seed-0 canary is the next experiment; no long
run has yet tested the learning hypothesis.

### Coupled reference observation-posterior canary result

The preregistered seed-0 canary completed normally under
`experiments/2026-07-22_cartpole_reference_observation_posterior_seed0_3500/`
from clean source `73bf922f322d1f4d04f19ae42784d407b512c086`.
Its manifest run ID is `81458cd5f703460d823206da6b12bc92` and its MLflow
run ID is `7f08978bc00f4ccd9423cc60c95b9f42`. It consumed 3,500 learner
updates and 21,865 environment transitions in 1,039.9 seconds. Relative to the
state-sum canary, only the coupled vector-encoder and posterior modes changed.

The complete deterministic 20-episode evaluation curve was:

```text
step:    100  200  300  400  500  600  700  800  900 1000 1100 1200
return:  9.4  9.4 9.35 9.35 9.35 9.35 9.35 9.35 9.35  9.4 79.2 85.6

step:   1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300 2400
return: 94.15 148.9 115.4 103.05 134.95 161.0 400.2 159.4 500.0 500.0 496.25 500.0

step:   2500 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500
return: 500.0 500.0 488.95 497.45 500.0 500.0 500.0 500.0 500.0 500.0 500.0
```

The run passes the frozen behavior-retention gate. It first crossed 450 at
update 2,100, never fell below 488.95 afterward, finished at 500, and has a
best-to-final gap of zero. This is materially different from the state-sum-
only canary, which first crossed 450 at update 3,100 and fell to 289.85 by
update 3,500. The coupled architecture is therefore associated with both much
earlier acquisition and retention through another 1,400 learner updates in
this seed.

The acquisition trajectory remains highly unstable. Return changed from 400.2
at update 1,900 to 159.4 at update 2,000 and 500 at update 2,100. None of the
coarse logged scalar losses explains this swing directly: actor entropy stayed
between approximately 0.39 and 0.47, world-model loss between 2.15 and 2.75,
and critic loss between 3.62 and 5.56 over updates 1,900--2,100. Solved endpoint
retention does not make this within-run oscillation unimportant.

The frozen endpoint probes completed but were right-censored by perfect
behavior. The 20-seed continuation probe under
`experiments/2026-07-22_cartpole_reference_observation_posterior_continuation_probe/`
scored 500 on all seeds: 10,000 transitions, 20 time-limit truncations, and no
physical terminal transitions, so failure ROC AUC is undefined. Live posterior
and prior effective discounts were 0.9960 and 0.9960 against the 0.997 target.
The five-seed policy-target probe under
`experiments/2026-07-22_cartpole_reference_observation_posterior_policy_q_probe/`
also scored 500 on every episode. It found no trusted-actionable actor states,
so target/real accuracy is undefined. On all 2,500 safe-corridor states the
actor used each action exactly 1,250 times, agreed with 98.1% of 911 confident
policy targets, and had one-step state MSE 0.00833, approximately 40 times lower
than the state-sum-only final checkpoint's 0.3345.

These censored probes neither pass nor fail the preregistered latent boundary
thresholds. Per user direction, multi-seed replication and a synthetic or
perturbed near-failure coverage benchmark are explicit TODOs rather than the
active investigation. No further training sweep is selected here.

### Preregistered fixed-history instability diagnostic

- **Question:** what changed between the update-2,000 actor that scored 159.4
  and the update-2,500 actor that scored 500, despite both belonging to one
  uninterrupted run?
- **Evidence and fixed distribution:** collect five deterministic trajectories
  from `checkpoint_step_2000.pt` on seeds 17--21, then replay exactly those
  histories through target checkpoints 2,000, 2,500, and final 3,500. Use the
  existing fixed-history probe with real counterfactual rollout horizon 30,
  model horizon 3, and 64 sampled policy-conditioned returns per forced first
  action. This is read-only and starts no trainer or collector.
- **Primary readouts:** on the identical physical states, compare actor/trusted-
  real balanced accuracy, policy-target/trusted-real accuracy, actor/target
  agreement, target action-margin magnitude, and current-posterior/one-step-
  prior/one-step-posterior state reconstruction errors. Preserve absolute-cart-
  position stratification so easier solved-policy state coverage cannot explain
  a difference.
- **Interpretation:** improved state errors select representation/dynamics
  drift; stable state errors with improved target/real accuracy select critic
  or imagined-return drift; stable targets with changed actor agreement select
  actor drift. Several boundaries may move together because shared world-model
  updates change both the actor input and imagined training distribution.
- **Stop rule:** one fixed history set and the three existing checkpoints. Do
  not launch seeds, retraining, optimizer changes, or a new benchmark from this
  diagnostic alone. Use the result to preregister one instability-specific
  causal test.

### Fixed-history instability diagnostic result

The read-only diagnostic is retained under
`experiments/2026-07-22_cartpole_reference_observation_posterior_fixed_history_drift/`.
The update-2,000 source actor produced five seed-17--21 trajectories with mean
return 152.0, 760 physical states, and 484 trusted-actionable states. Every
target checkpoint was evaluated on those exact states and source actions.

| Target checkpoint | Actor actions 0 / 1 | Actor/real balanced | Q actions 0 / 1 | Q/real balanced | Actor/Q agreement | Confident policy-Q/real |
|---|---:|---:|---:|---:|---:|---:|
| 2,000 (return 159.4) | 403 / 357 | 0.547 | 473 / 287 | 0.556 | 0.882 | 0.502 |
| 2,500 (return 500) | 718 / 42 | 0.500 | 746 / 14 | 0.500 | 1.000 | 0.500 |
| 3,500 (return 500) | 742 / 18 | 0.500 | 696 / 64 | 0.488 | 0.977 | 0.500 |

The physical counterfactual labels are fixed: 22 states prefer action 0, 462
prefer action 1, and 276 are tied over the trusted 30-step horizon. The solved
actors therefore do not provide a generally improved controller on the failed
checkpoint's recovery distribution. They become almost constant-action actors,
and their critics and sampled policy targets become self-consistently aligned
with the same action while remaining at chance against real outcomes.

The representation also changes coordinate quality rather than improving
uniformly:

| Target checkpoint | Current posterior x MSE | One-step prior x MSE | Current posterior x-dot MSE | Current posterior theta-dot MSE |
|---|---:|---:|---:|---:|
| 2,000 | 1.006 | 1.057 | 0.753 | 0.0295 |
| 2,500 | 0.165 | 0.212 | 0.422 | 0.00894 |
| 3,500 | 2.178 | 0.734 | 0.0380 | 0.00159 |

At final, current-posterior cart-position MSE exceeds 5 in every fixed-history
bin above absolute position 1.0, despite excellent final on-policy one-step
state MSE of 0.00833. The model has become accurate inside its own solved
trajectory and poor on older recovery histories that are physically reachable.

This selects a brittle closed-loop-attractor interpretation. The solved actor,
critic, and model are mutually consistent inside a narrow trajectory corridor;
they are not robust to a history generated by the earlier policy. A small
parameter update can change which corridor the discrete argmax policy enters,
turning modest network drift into a large, discontinuous return change. The
post-2,100 retention result is real within this run, but it does not resolve
the larger instability problem.

### Preregistered recurrent-carry horizon diagnostic

There is a concrete train/deploy mismatch that can create this history
dependence. Stream replay samples 16-step sequences, explicitly zeroes the RSSM
carry at every sampled start (including mid-episode), and discards the first
four steps as burn-in. Actor/value training therefore sees recurrent states
generated by at most 16 consecutive real transitions. Evaluation preserves the
carry continuously for up to 500 transitions.

- **Hypothesis:** deployed recurrent states drift away from the truncated-carry
  manifold used for model and actor training. The same physical observation
  then maps to materially different features/actions under full episode carry
  versus a locally reconstructed carry, producing history-sensitive attractors
  and sharp performance swings.
- **Fixed evidence:** use the existing update-2,000 seed-17--21 histories and
  checkpoints 2,000, 2,500, and final. At each eligible physical state, replay
  the same observations/actions with continuous episode carry and with carry
  reset 4, 8, 16, and 32 steps earlier. Do not alter weights or collect data.
- **Primary readouts:** actor-action disagreement, actor-logit delta, latent
  feature RMS distance, decoded-state error, and critic-value difference by
  context length, episode time, and absolute cart-position bin.
- **Interpretation:** errors and action disagreement that grow with continuous
  carry length while short reconstructed contexts remain accurate select the
  training/deployment carry mismatch. Similar results across context lengths
  reject that mechanism and leave replay recovery coverage/value grounding as
  the proximal instability boundary.
- **Stop rule:** one read-only diagnostic. Do not change sequence length,
  burn-in, replay sampling, recurrent state transport, or training code until
  the measurement selects the carry mismatch.

### Recurrent-carry horizon diagnostic result

The read-only diagnostic and its tested implementation are retained under
`experiments/2026-07-22_cartpole_reference_observation_posterior_carry_horizon/`.
It produced 9,120 aligned rows over the three checkpoints and four reconstructed
context lengths.

| Checkpoint | Local context | Actor disagreement | Feature RMS | Absolute critic-value difference | Full / local x MSE |
|---|---:|---:|---:|---:|---:|
| 2,000 | 4 | 0.167 | 0.153 | 1.284 | 1.006 / 1.006 |
| 2,000 | **16** | **0.029** | 0.043 | 0.385 | 1.006 / 1.007 |
| 2,500 | 4 | 0.014 | 0.138 | 3.886 | 0.165 / 0.190 |
| 2,500 | **16** | **0.004** | 0.029 | 0.699 | 0.165 / 0.166 |
| 3,500 | 4 | 0.063 | 0.126 | 9.214 | 2.178 / 0.887 |
| 3,500 | **16** | **0.000** | 0.024 | 0.925 | 2.178 / 2.195 |

Thirty-two rows reduce action disagreement to 2.0%, 0%, and 0%. At the actual
16-row training horizon, full-episode carry and locally reconstructed carry
produce essentially identical state errors and almost identical actions. The
final checkpoint's large recovery-history cart-position error remains large
under local context, including after episode time 128 (`5.885` full versus
`5.882` local). Continuous carry accumulation is therefore not the proximal
cause of the brittle recovery geometry.

Four-row contexts can materially change actor logits and critic values, but
the trainer dreams from post-burn-in rows across the remainder of each 16-row
sequence; the measured divergence largely disappears by eight to 16 rows. This
result rejects a sequence-length or recurrent-state-transport intervention.
The active boundary remains policy/value behavior under induced distribution
shift.

### Preregistered stochastic-versus-mode policy diagnostic

- **Question:** do the large deterministic return swings reflect a change in
  the stochastic policy objective, or a discontinuous argmax controller moving
  between narrow closed-loop attractors?
- **Frozen checkpoints:** evaluate periodic updates 500, 1,000, 1,500, 2,000,
  2,500, 3,000, and final 3,500. For each, run both deterministic argmax and
  reproducibly sampled categorical actions on reset seeds 17--116 (100 episodes
  per cell). Keep posterior extraction deterministic and change only action
  selection. No model state or weights change.
- **Primary readouts:** mean, median, range, solved fraction, and per-seed paired
  returns. Compare temporal smoothness and best-to-final movement under the two
  extraction rules.
- **Interpretation:** a stable or smoothly improving sampled-policy curve with
  sharp argmax jumps selects mode-boundary sensitivity. A sampled policy that
  remains poor while argmax reaches 500 selects a narrow deterministic orbit
  that is not representative of the optimized stochastic policy. Similar
  swings under both rules select genuine expected-policy instability rather
  than evaluation extraction.
- **Stop rule:** 14 read-only cells over existing checkpoints. Do not change
  entropy, evaluation policy, actor objective, or training from this result
  alone; use it to identify the instability being measured.

### Action-extraction partial result and latent-semantics extension

The first 14 cells are retained under
`experiments/2026-07-22_cartpole_reference_observation_posterior_policy_extraction/`.

| Update | Argmax mean / solved fraction | Sampled-action mean / solved fraction | Paired return MAE |
|---:|---:|---:|---:|
| 500 | 9.39 / 0.00 | 23.95 / 0.00 | 14.62 |
| 1,000 | 9.25 / 0.00 | 21.58 / 0.00 | 12.33 |
| 1,500 | 127.43 / 0.01 | 102.81 / 0.00 | 35.10 |
| 2,000 | 159.17 / 0.00 | 156.53 / 0.00 | 11.10 |
| 2,500 | 500.00 / 1.00 | 500.00 / 1.00 | 0.00 |
| 3,000 | 500.00 / 1.00 | 500.00 / 1.00 | 0.00 |
| 3,500 | 500.00 / 1.00 | 500.00 / 1.00 | 0.00 |

Sampling only the actor action does not explain the acquired 500-return orbit;
after update 2,500 it produces exactly the same perfect returns as argmax over
100 seeds. Earlier stochastic actions improve the near-constant bad policies
and modestly reduce the update-1,500 policy, but the update-2,000-to-2,500 phase
change remains.

This partial result does not yet test the trained policy semantics. The
collector and pinned DreamerV3 both sample the categorical posterior latent on
every observed step. Local deterministic evaluation instead takes posterior
argmax before selecting the actor action. The existing action-sampling cells
therefore hold fixed a mode latent that the collection policy does not use.

The preregistered extraction diagnostic is extended once, before inspecting
the missing cells:

- add posterior-sampled/action-argmax and posterior-sampled/action-sampled
  evaluations for the same seven checkpoints, 100 reset seeds, and independent
  fixed latent/action RNG streams;
- apply the authored one-percent unimix before either categorical sample, as
  collection does; argmax ordering remains unchanged;
- compare the four extraction combinations. If posterior sampling alone
  destroys the solved orbit, the reported deterministic success is a mode-
  latent artifact and the stochastic learned/collected policy remains unstable.
  If both posterior-sampled cells remain solved, extraction is rejected more
  generally and the closed-loop phase change belongs to the trained policy.

The original no-training stop rule remains in force.
