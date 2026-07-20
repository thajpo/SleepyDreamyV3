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

## Reliability follow-up

Interrupted manifests correctly record `status: interrupted` and evaluation
history, but incorrectly retain `progress.train_step: 0` and `env_steps: 0`.
Fix this bookkeeping issue separately from the learning experiment so it does
not alter the scientific intervention.
