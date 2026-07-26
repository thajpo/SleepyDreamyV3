# DreamerV3 conformance and reliability plan

Status: active research plan, 2026-07-25

Local baseline: `35d7454c22d34bb9ba72d35c2f9f7254bdfac0cd`

Reference paper: DreamerV3 v2, <https://arxiv.org/abs/2301.04104>

Reference source: `danijar/dreamerv3@e3f02248693a79dc8b0ebd62c93683888ddaccfe`

Current decision: first requalify the corrected scale-only RMSNorm architecture
under the frozen seed-0 contract. The previous canary acquired but did not
retain a controller, but it used the superseded shift-bearing v1 model. Bundling
that correction with exact cached replay carry would confound the next result.
If the corrected v2 canary fails, Phase 3 selects exact cached replay carry as
the next reference-conformance change. Later results supersede earlier causal
readings while preserving them as chronology below.

## Phase 1 audited conclusion

The repository has repaired real runtime and replay defects and can acquire a
solved CartPole controller. It has not demonstrated stable learning. In the
failed replay-evidence run, the final coupled world-model, actor, and value
system loses useful recovery-action ordering on histories that remain eligible
for replay, while the actor continues to agree with its current imagined
preference.

The evidence does not isolate the failure to the critic or prove that the actor
is downstream. The existing checkpoint cross changes the encoder, RSSM,
reward/continuation heads, actor, critic, and the actor used to construct real
continuation labels. The supported conclusion is therefore **coupled
policy-conditioned parameter/target drift**.

This correction supersedes the stronger causal wording at the end of
`reports/cartpole_current_instability.md`; it does not erase the original
chronology.

## What is established

- Collection originally outran learning: the critical realized replay ratio was
  approximately `0.16` rather than the configured `16`.
- Continuous replay chunks corrected long-episode publication latency, but the
  frozen three-seed stability gate still failed.
- The final failed checkpoint is poor on both final replay evidence and the
  update-3,000 evidence that its earlier checkpoint handled well.
- Replay capacity had not evicted the earlier histories.
- Frozen final latents can support a freshly fitted recovery-value head; static
  information absence is not sufficient.
- Actor/dream agreement near 99% rejects action extraction or snapshot-local
  actor fitting as the first observed boundary.
- The central lambda-return, continuation-weight, slow-value, return-scaling,
  and LaProp equations have no presently identified indexing or sign defect.

## What remained unresolved at the Phase 1 boundary

- Whether value predictions lose a stable target or the policy-conditioned
  target itself changes first.
- Which of representation, reward/continuation model, critic, or actor is the
  first moving component.
- Whether replay eligibility translated into sufficient effective gradient
  exposure for the anchor histories.
- Whether the same CartPole behavior occurs in the pinned reference
  implementation.

## Confirmed conformance gaps at the Phase 1 boundary

- Production actor, value, and vector decoder heads lack the reference's three
  normalized hidden layers.
- Actor output scale is applied as an absolute normal standard deviation rather
  than a scale on fan-in truncated-normal initialization.
- The reward model is a bare linear head.
- Replay context/reset and action-masking semantics differ from the reference.
- World-model burn-in rows are trained but excluded from replay-ratio pacing;
  the frozen CartPole contract therefore gives the world model an effective
  ratio of `21.33` while actor/critic receive `16`.
- Pixel encoder, decoder, initialization, AGC, model scale, and collector
  mid-episode synchronization do not yet match current reference DreamerV3.

## Frozen contracts

Machine-readable contracts live in `reports/contracts/`:

- `paper_v2_atari100k.yaml`: paper benchmark, ratio 128 and 100K decisions.
- `official_e3f0224_atari100k.yaml`: current-source conformance, ratio 256 and
  110K decisions.
- `cartpole_drift_v1.yaml`: local late-collapse diagnostic at the audited source
  commit. It is not a paper benchmark.

The paper and current source contracts must never be merged into an unnamed
hybrid. `scripts/verify_dreamerv3_contracts.py` validates raw-frame and update
accounting.

## Phase 1 preregistration: fixed-policy component and target cross

### Question

On the retained update-3,000 and final replay histories, does useful ordering
first disappear through representation coordinates, reward/continuation model
targets, critic parameters, actor-conditioned rollouts, or an unresolved
interaction?

### Frozen artifacts

- Run `1adf567637bc437c9053cdc62ad98dce`.
- Update-3,000 and final checkpoints and their exact replay-evidence files.
- CPU execution, seed 17, posterior mode, real horizon 30, 64 model samples.
- No replay resampling and no training.

### Matrix

For each evidence cohort, cross update-3,000 versus final components:

1. encoder plus RSSM representation;
2. reward plus continuation heads;
3. online critic;
4. actor used for imagined continuation.

Run a fixed-reference-policy panel in which real continuation labels are
generated once by the complete update-3,000 controller and reused across every
cell. Report actor, posterior critic, one-step model value, and full imagined
value ordering. Any panel that changes the real continuation policy must be
reported separately and may not be used for component causality.

For the fixed anchor cohort, also report the exact replay target, imagined
lambda target, critic prediction, and gradient norm/cosine for imagined value,
replay value, and slow-value regularization against a trusted supervised
recovery-ordering loss.

### Decision and stop rule

- A component swap that transfers the failure while fixed labels and all other
  components remain solved selects that component as the next causal boundary.
- Failure only when multiple final components are combined selects an
  interaction and must not be rewritten as single-component causality.
- If targets remain useful but critic predictions fail, investigate fitting and
  retention. If the targets fail first, investigate the actor/model target
  generator.
- Do not start a training intervention until the offline matrix and independent
  numerical fixture are documented.

### Fixed-policy component-cross result

The complete 16-cell matrix ran from clean source `ab04ed0` on both retained
evidence cohorts. Outputs are under
`experiments/2026-07-25_cartpole_fixed_policy_component_cross/`. Each cell used
the exact real branch labels produced earlier by the complete update-3,000
controller. Both runs used CPU, one Torch thread, seed 17, 64 model samples,
and every actionable row. Final-evidence execution took 266.69 seconds with
peak RSS 806 MiB; update-3,000 evidence took 333.96 seconds with peak RSS
813 MiB.

| Fixed solved-policy labels | Solved representation/heads/critic/actor | Final representation/heads/critic/actor |
|---|---:|---:|
| Update-3,000 evidence: actor balanced accuracy | 0.907 | 0.912 |
| Update-3,000 evidence: posterior critic balanced accuracy | 0.820 | 0.691 |
| Update-3,000 evidence: full-dream balanced accuracy | 0.895 | 0.895 |
| Final evidence: actor balanced accuracy | 0.887 | 0.882 |
| Final evidence: posterior critic balanced accuracy | 0.853 | 0.760 |
| Final evidence: full-dream balanced accuracy | 0.867 | 0.888 |

The matched final system therefore retains useful ordering for the solved
controller's fixed real target. This rejects the previous interpretation that
final parameters simply destroy the already learned stable-policy value
function. The earlier complete-checkpoint cross changed the continuation
policy and therefore changed the real target being measured.

Crossed parameter groups show strong coordinate co-adaptation rather than one
portable failed head. On final evidence:

- final representation plus solved reward/continuation heads yields full-dream
  balanced accuracy `0.511` with the final critic/actor;
- restoring the matching final heads raises it to `0.888`;
- final critic on solved representation gives posterior accuracy `0.494`;
- solved critic on final representation gives `0.376`;
- matching final representation and critic gives `0.760`.

Those crossed failures cannot be called independent head or representation
defects: a head trained in one moving latent coordinate system need not remain
valid in the other.

The fixed and checkpoint-specific real labels reveal the changed boundary. On
final evidence, the solved controller has 202 actionable histories while the
final controller has 597. Among the 196 histories actionable for both,
preferences agree on 98.47% and action-margin correlation is `0.9988`. The
additional 401 histories were ties under solved continuation—mean score
`29.79/30` for both first actions—but under final continuation their mean
scores fall to `23.80` and `25.77`.

The final actor remains good on the shared actionable cohort (`0.905` balanced
accuracy), but scores only `0.420` on the 401 newly failure-sensitive histories.
The final full-dream target scores `0.360` there. Across all replay rows, solved
and final deployed actions agree only 71.03%; the final actor changes from a
roughly balanced action histogram (`1522/1550`) to an action-0-heavy histogram
(`2410/662`). On the shared actionable cohort the two actors agree exactly; the
drift occurs mainly in states where the solved controller made both forced
actions safe.

The first supported boundary is now:

> Policy drift in formerly safe/indifferent states makes the real closed loop
> failure-sensitive. The final imagined target does not correctly rank the new
> failure corridor, and the actor follows that target. Stable old-policy
> ordering is retained by matched final components.

This does not yet distinguish actor optimization noise, actor/representation
coordinate drift, or model exploitation under multi-step final-policy
rollouts.

## Phase 1 next preregistration: deployed-policy matched rollout

### Question

When the final policy creates the new failure corridor, does its world model
accurately predict the consequences of the exact actions it executes in the
real environment?

### Frozen read-only probe

- Evaluate update-3,000 and final checkpoints with
  `scripts/probe_cartpole_rollout_fidelity.py`.
- Use CPU, evaluation seeds 17--36, 20 episodes, horizons 1, 3, 5, 10, and 15,
  64 prior samples, batch size 64.
- Replay each deployed controller's exact real action sequence through its own
  prior. Do not substitute planner actions or compare different action
  sequences inside one error decomposition.
- Report real return, posterior critic return error, decoded-state MSE,
  predicted continuation, model/target correlation, and the existing reward,
  continuation, final-discount, and critic-transport error decomposition.

### Interpretation and stop rule

- Accurate posterior/oracle values with worsening prior rollout error selects
  model exploitation or rollout-distribution error.
- Accurate matched rollout with worsening posterior critic error selects value
  fitting on the final-policy distribution.
- Movement in both retains a coupled boundary and selects one final-only
  trajectory trace that crosses actor weights versus representation coordinates
  at the first real action divergence.
- This is one read-only comparison. Do not change training from it alone, and do
  not begin the replay-accounting canary until the result is documented.

### Deployed-policy matched-rollout result

The comparison ran from clean source `865eb93` and is retained under
`experiments/2026-07-25_cartpole_deployed_policy_rollout_fidelity/`. It took
629.84 seconds on one CPU thread with peak RSS 674 MiB.

The solved checkpoint completed all 20 episodes at the 500-step Gym time limit;
the final checkpoint averaged `45.2` over 904 transitions and 20 physical
terminations. Therefore their absolute critic/return errors are not directly
comparable: the probe's finite return-to-go target does not bootstrap at a
time-limit truncation, whereas Dreamer correctly treats truncation as
bootstrappable. The solved checkpoint's large negative oracle error is a
diagnostic-target artifact, not evidence that its working critic is worse.

Three same-semantics readouts remain valid:

| Matched deployed actions | Update 3,000 | Final 3,500 |
|---|---:|---:|
| Mean real return | 500.0 | 45.2 |
| One-step decoded-state mean MSE | 0.0061 | 0.0301 |
| Fifteen-step decoded-state mean MSE | 0.0550 | 0.1240 |
| Prior continuation on nonterminal transitions | 0.9971 | 0.9957 |
| Prior continuation on physical terminal transitions | not observed | 0.9919 |

Along the exact failed-policy action sequence, final one-step state error is
about five times the solved-path error and fifteen-step error is about 2.25
times larger. More importantly, the imagined prior assigns actual terminal
transitions continuation probability `0.9919`; it effectively does not foresee
the failures induced by its deployed policy. This selects rollout-distribution
error/model exploitation as part of the boundary, while retaining coupling
with the actor/representation drift that creates the bad action sequence.

## Phase 1 final preregistration: final-only trajectory trace

- Select 32 of the 401 final-only actionable replay occurrences uniformly
  without replacement using seed 23.
- Reconstruct update-3,000 and final posterior histories from the same retained
  final evidence.
- From each physical state, force each first action and then follow the complete
  final controller in real CartPole for at most 30 steps.
- On every realized history, evaluate the 2x2 actor/representation action cross:
  solved actor/solved representation, final actor/solved representation, solved
  actor/final representation, and final actor/final representation.
- For the actual final-controller action, sample 64 final priors and record
  decoded next-state error, reward, and continuation before observing the real
  successor. The replayed real branch score must match the previously retained
  final-policy label exactly.
- Report the first solved/final action divergence, action histograms and
  pairwise agreement, terminal/nonterminal predicted continuation, and state
  error by depth.

If the solved actor on final coordinates transfers the final action bias while
the final actor on solved coordinates does not, representation coordinates are
the leading policy-drift boundary. The reverse selects actor parameters. Both
moving retains co-adaptation. Failure prediction that remains optimistic on
the exact real terminal transitions selects the model target seen by
imagination. Stop after this read-only trace and complete the reference oracle;
do not modify training from this local implementation alone.

### Final-only trajectory-trace result

The frozen trace ran from clean source `d0015b1` and is retained under
`experiments/2026-07-25_cartpole_final_only_policy_trace/`. It selected 32
histories, executed both forced branches from each, and reproduced all retained
final-controller branch scores exactly (`0.0` maximum error). The 64 branches
contained 1,589 transitions and 58 physical terminations.

Every branch contained a solved/final deployed-action divergence. The first
divergence occurred at mean depth `2.97`; 40 of 64 branches diverged by depth
three. Across the 1,531 decision rows, solved/solved and final/final actions
agreed on `63.42%`, leaving 560 changed rows. On those changed rows:

- replacing solved actor weights with final actor weights while retaining the
  solved representation transferred the final action on `0%` of rows;
- replacing solved representation coordinates with final representation
  coordinates while retaining solved actor weights transferred the final action
  on `100%` of rows.

The action histograms make the same interaction visible. The solved actor emits
action 1 on 1,407 of 1,531 solved-coordinate rows but action 0 on 1,527 of 1,531
final-coordinate rows. The final actor partly compensates on final coordinates,
emitting action 1 on 847 rows rather than four. This is not evidence that one
semantic feature vanished: latent coordinates are free to move, and checkpoint-
crossed heads are not invariant to that movement. It is direct evidence that
the representation/policy interface moved faster or farther than the coupled
policy could harmlessly track on the realized failure histories.

The model-side failure remains independently visible. Final priors predict
continuation `0.99597` on physical terminal transitions and `0.99580` on
nonterminal transitions; they rank terminal transitions as very slightly *more*
likely to continue. Mean next-state MSE is `0.0459` overall and `0.0980` at
termination. Thus the changed latent-policy closed loop enters failure within a
few actions while imagination supplies almost no terminal warning.

The selected boundary is therefore **representation/policy tracking under a
model-exploited failure corridor**. The evidence rejects actor parameters alone
as the initiator on these histories, but it does not distinguish harmless latent
reparameterization plus lag from loss of dynamically useful state. Phase 1 now
stops as preregistered. The next action is independent numerical and architecture
conformance against pinned source before any training intervention.

## Execution ladder

1. Complete the offline component/target cross and independent JAX fixtures.
2. Separate replay context from trained rows and prove collector/replay latent
   parity at genuine resets.
3. Implement a versioned state-only `reference_v3` architecture with exact
   heads, initialization, dimensions, and normalization.
4. Qualify CartPole with one canary, then three seeds, then five seeds. Use a
   fixed decision budget, no early stop, held-out final evaluation, and retained
   replay/RNG state.
5. If reference CartPole still collapses, fork frozen-world-model,
   frozen-actor, frozen-critic, and fixed-replay continuations from an identical
   solved state before choosing one stabilizer.
6. Port and validate the reference pixel path, Atari protocol, model sizes,
   reward events, replay context, and collector version lag before Pong.
7. Progress through 1K, 10K, and 100K Pong decisions with explicit stop gates;
   run paper and current-source contracts as separate experiments.

## Phase 1B result: independent oracle and state architecture

The numerical oracle is generated by JAX 0.4.33 without importing the local
package. It transcribes the pinned source equations and records source commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe` in
`tests/fixtures/dreamerv3_e3f0224_oracle.json`. Six cross-framework tests cover
symlog/symexp, symmetric two-hot bins, two-hot targets and predictions,
imagination and replay lambda returns, and percentile normalization. The
fixture initially exposed an apparent imagination-index difference; tracing the
actual `H+1` value layout proved the production equation correct and the first
fixture translation wrong. The corrected independent comparisons all pass.

The source audit initially reported an additional architecture mismatch
relevant to the observed representation/policy movement: it claimed official
RMSNorm learned both a scale and a shift. A later pin-level audit corrected
that claim: the pinned RMS path learns scale only. An end-to-end follow-up also
corrected the initial audit note about vector preprocessing: the shared local
trainer/collector pipeline already symlogs state before the encoder. The new
encoder preserves that single transform instead of applying it a second time.
The previous use of the word “reference” therefore described topology, not
complete numerical conformance.

A new checkpointed `reference_v3_state` contract now composes the pinned
state-only size-1M architecture as one unit:

- deterministic state 512, hidden width 64, eight recurrent blocks, and
  stochastic state 32 by 4;
- symlog vector input and three normalized encoder layers;
- the then-selected learned-scale-and-shift RMSNorm with epsilon `1e-4`
  throughout;
- two prior layers, one posterior layer, and the grouped recurrent core;
- three hidden layers for actor, value, and vector decoder, and one hidden
  layer for reward and continuation;
- fan-in truncated-normal initialization with the pinned `1.1368` correction,
  zero biases, actor outscale `0.01`, and zero reward/value output weights.

The historical v1 qualification contract contains 639,173 trainable parameters
across encoder, world model, actor, and value model. The corrected scale-only
implementation contains 637,381 and is not qualified by the recorded canary;
it requires a separately frozen run. The architecture is deliberately rejected
for pixels until the pixel encoder/decoder is ported; mixed legacy/reference
component selections are also rejected. Historical snapshots retain
`architecture_contract=historical`, so existing evidence remains loadable. The
mechanical gate passed 61 focused tests, the complete 283-test fast suite,
compile and scoped type checks, and a one-update multiprocess CPU smoke using
the then-current architecture. The one resume integration fixture that
intentionally authors legacy modules requests the historical contract
explicitly.

No behavioral claim follows from mechanical conformance. Before a CartPole
canary, Phase 2 must correct and prove replay context/reset accounting so the
new architecture is not evaluated under a known data-semantics mismatch.

## Phase 2 result: replay context, reset, and accounting

The end-to-end row audit found two concrete defects in the authored CartPole
path.

First, `replay_burn_in=4` excluded four rows from actor/value starts and from
the replay-ratio denominator, but the trainer still summed world-model loss on
all 16 sampled rows and divided that loss by 16. At nominal replay ratio 16,
the world model therefore received 16/12 times as many supervised rows per
environment decision: effective ratio `21.33`. This was not merely a reporting
error; context observations supplied real encoder, dynamics, reconstruction,
reward, continuation, and KL gradients.

Second, the collector stored only observations reached *after* actions. Replay
then labeled the first stored successor as `is_first`, reset recurrent and
stochastic state, but supplied the action that caused the successor. Pinned
Dreamer instead records the reset observation itself with reward zero and zero
previous action, forms its posterior, and only then transitions under the first
environment action. Uniform stream samples also marked arbitrary mid-episode
starts as `is_first`, conflating a sampled truncation with a genuine reset.

The versioned reference path now:

- inserts the true reset observation with zero reward and previous action;
- preserves `is_first` only at real episode boundaries, including boundaries
  crossed inside a stream sample;
- masks recurrent state, stochastic state, and previous action at those genuine
  reset rows;
- runs leading burn-in rows only to reconstruct an approximate carry, detaches
  that carry at the train boundary, and applies no loss or diagnostic weighting
  to the context rows;
- divides world-model, actor, and value losses by the same trained-row count and
  uses that count for fresh and resumed replay pacing.

`replay_row_alignment=post_action` preserves every historical checkpoint and
probe. `reference_v3_state` requires `replay_row_alignment=reference`, so a new
run cannot silently combine the corrected model with old row semantics.

Focused evidence covers collector chunk contents and environment-step counts,
mid-episode versus crossed-reset stream masks, reset action masking, and zero
gradient on context tokens with nonzero gradient on the first trained row. The
complete 288-test fast suite, compile and scoped type checks, and a one-update
multiprocess CPU smoke all pass under the corrected reference row contract. The
remaining difference from pinned source is explicit: local replay reconstructs
carry from a bounded burn-in, whereas upstream can reuse replay-cached model
entries. This is truncated recurrent replay, not exact cached-carry parity; it
must be measured as a limitation rather than described as full replay identity.

The next gate is a read-only carry-parity measurement over the exact proposed
CartPole burn-in. If the first trained latent materially differs from a full-
episode rollout, increase or redesign context before training. If parity is
adequate, preregister one reference-state CartPole canary.

### Carry-parity preregistration

- **Question:** does eight-row truncated recurrent replay reconstruct the first
  trained latent closely enough to stand in for a full episode prefix under the
  new reference-state model?
- **Frozen setup:** clean source after the Phase 2 repair; seed-0 initialized
  `reference_v3_state` size-1M model; 20 random-policy CartPole episodes with
  environment seeds 17--36; deterministic posterior probabilities rather than
  categorical samples; burn-ins 1, 2, 4, 8, and 16.
- **Comparison:** for every episode depth supported by each burn-in, compare the
  latent obtained by replaying only the preceding context from zero with the
  latent obtained from the complete reset-state prefix. Report joined-feature
  relative L2 and cosine similarity, stochastic-probability error, actor
  probability L1, and modal-action agreement.
- **Primary gate at burn-in 8:** at least 99% modal-action agreement, median
  feature cosine at least 0.99, and 95th-percentile relative feature L2 at most
  0.10. These thresholds concern policy-interface parity, not exact floating-
  point identity.
- **Stop rule:** one initialized-model measurement. Failure blocks training and
  selects longer context or cached carry. Passing permits one preregistered
  CartPole canary but must be repeated on its trained checkpoints because
  recurrent memory can change during learning.

### Carry-parity result and bounded extension

The clean-source probe at `8e1e7f0` completed in 3.28 seconds with peak RSS
235 MiB. Its artifact is retained at
`experiments/2026-07-25_cartpole_reference_carry_parity/summary.json`.

| Burn-in | Comparisons | Feature cosine median | Feature relative L2 p95 | Actor action agreement |
|---:|---:|---:|---:|---:|
| 1 | 490 | 0.7398 | 0.7268 | 0.4163 |
| 2 | 470 | 0.8312 | 0.6280 | 0.8340 |
| 4 | 430 | 0.9240 | 0.4659 | 1.0000 |
| 8 | 350 | 0.9801 | 0.2752 | 1.0000 |
| 16 | 198 | 0.9971 | 0.1199 | 1.0000 |

Burn-in eight fails both latent thresholds despite preserving the initialized
actor's modal action. Sixteen passes cosine and action agreement but narrowly
misses the `0.10` relative-L2 threshold. The curve is monotonic and strongly
supports longer context rather than a categorical failure of truncated replay.

One bounded extension is authorized before considering cached carry: repeat the
identical model seed, environment seeds, action streams, deterministic posterior,
and metrics for burn-ins 20 and 24. Select the shortest context that clears the
same three thresholds. If neither passes, stop and implement replay-cached carry.
Because sequence length is 32, a passing 20-row context would leave 12 trained
rows per sample and a 24-row context would leave eight; replay pacing must use
that exact trained-row count.

### Carry extension result

The identical clean-source extension retained under
`experiments/2026-07-25_cartpole_reference_carry_parity/extension_20_24.json`
completed in 1.76 seconds with peak RSS 235 MiB. Burn-in 20 produced 139
comparisons, median cosine `0.99868`, p95 relative L2 `0.08486`, and actor
agreement `1.0`; burn-in 24 produced 99 comparisons, `0.99931`, `0.06032`, and
`1.0`. Both pass, so the stop rule selects the shorter 20-row context.

This result changes the proposed CartPole sample from length 16/context 4 to
length 32/context 20. With batch size 8, both contracts train exactly 96 rows
per update. At replay ratio 16, both therefore authorize one update per six
agent decisions and preserve the 21,000-decision/3,500-update comparison.

## Phase 2 qualification canary preregistration

The complete machine-readable contract is
`reports/contracts/cartpole_reference_v3_state_v1.yaml`.

- **Hypothesis:** after removing runtime, replay-throughput, row-alignment,
  architecture, and context-accounting confounds, the pinned state-only model
  will acquire and retain a CartPole controller through update 3,500.
- **Causal bundle:** `reference_v3_state`, reference reset rows, loss-free
  20-row context, and exact trained-row replay accounting. This is a replication
  conformance bundle, not a single-component causal canary.
- **Frozen run:** ROCm device, training seed 0, one collector, 3,500 updates,
  expected 21,000 decisions, batch 8, sequence 32, context 20, replay ratio 16,
  buffer 512, startup 16 completed episodes, uniform selector, 20 deterministic
  evaluation episodes every 100 updates, checkpoints/evidence every 500, no
  early stop.
- **Behavior gate:** reach mean return 475; never fall below 400 afterward;
  finish at least 475; best-to-final gap at most 25.
- **Evidence gate:** retain the manifest, all evaluation points, periodic/best/
  final checkpoints, and 256 replay evidence sequences per checkpoint. Repeat
  carry parity on trained checkpoints before interpreting a pass or failure.
- **Stop rule:** first profile an exact-config short run. If memory and projected
  runtime are safe, run seed 0 once. Failure selects diagnostics, not tuning.
  Passing selects unchanged seeds 1 and 2.

### Exact-config preflight result

The clean-source ROCm preflight at commit `c2925ae` completed 50 updates in
41.68 seconds wall time, including roughly 22 seconds of collector startup. It
peaked at 3,883,804 KiB resident memory and retained 17 MiB of run artifacts.
The run finished normally at 50 updates and 728 observed environment steps;
its manifest run ID is `23b3c52982d84503bebe30ddd5387a7b` and its MLflow
run ID is `d1fff3ef8ab242ef9a7166e696a1abb1`.

This is safe on the 24 GiB ROCm host and projects the frozen 3,500-update run at
roughly 30--40 minutes plus deterministic evaluation and checkpoint overhead.
The contract's 21,000 decisions are the replay-pacing authorization
(`3,500 * 96 / 16`), not a hard final counter: the manifest also includes the
completed-episode startup debt and collection overshoot while the trainer
stops. The canary must report its actual final environment-step count rather
than silently treating those two quantities as identical.

The preflight therefore passes its stop rule. The next and only authorized
behavioral run is the frozen seed-0 canary. No hyperparameter change is selected
from this profile.

### Seed-0 qualification result

The clean-source canary at commit `473e23f` completed all 3,500 updates normally
in 26:45.73 wall time, with 3,904,188 KiB peak RSS and no replay descriptor
drops. Its manifest run ID is `605ba73c3b1241e1a09a88b25093a027`, its MLflow
run ID is `f4d79c244ae54bf6a7b54799c6fee322`, and it ended at 21,408 actual
environment steps. The 408-step difference from the pacing authorization is
the expected startup/collector accounting debt identified by the preflight.

The run **fails** the frozen behavior gate. It acquired a near-solved controller,
reaching 403.55 at update 1,900 and the run best of 480.55 at update 2,400. It
then fell through the retention floor at update 2,600 (389.65), declined to
114.75 at update 3,200, and finished at 141.9. The best-to-final gap is 338.65.
Seeds 1 and 2 are therefore not authorized.

This is a narrower and more useful failure than the earlier controlled runs.
The corrected reference architecture and replay contract can acquire a
controller, while continued joint training still destroys deployed behavior.
The process, fan-out, replay delivery, artifact, and evaluation paths remained
healthy, so the old collection-outruns-learning defect does not explain this
collapse. Low contemporaneous pole-angle reconstruction error also makes plain
state reconstruction failure insufficient as an explanation.

Per the frozen stop rule, no tuning follows this result. The next experiment is
diagnostic: run trained carry parity on `checkpoint_best.pt` and
`checkpoint_final.pt`, then apply the existing fixed-policy component cross and
rollout-fidelity probes to the best/final pair. Those measurements decide
whether the first post-acquisition break is replay carry, policy-conditioned
representation drift, or imagined dynamics/value error.

### Trained carry result and next diagnostic contract

The trained carry probe ran from clean commit `ad905bb` on the same 20 random
CartPole episodes and the selected 20-row context. The best checkpoint fails:
median full-prefix feature cosine `0.98505`, p95 relative L2 `0.41128`, and
actor agreement `0.93525`. The final checkpoint also fails the feature gate,
with `0.99565`, `0.31928`, and `0.99281`, respectively. At 24 rows the best
checkpoint still fails (`0.98708`, `0.32714`, `0.93939`), while final remains
better but still misses the L2 gate (`0.99776`, `0.23407`, `1.0`).

Training therefore makes the initialized model's 20-row carry qualification
non-stationary. This is a real replay/deployment mismatch and disqualifies an
unqualified causal reading of truncated replay features. It does **not** by
itself explain best-to-final collapse: the failed final controller has better,
not worse, carry parity than the best controller on this frozen off-policy
panel. Cached replay state or a trained-policy carry contract remains a likely
future conformance requirement, but changing replay now would violate the
diagnostic stop rule.

The next bounded diagnostic uses no training and changes no model parameters:

- matched rollout fidelity on best and final, CPU, seeds 17--36, 20 episodes,
  horizons 1/3/5/10/15, 64 prior samples, batch 64;
- fixed real labels generated once from `checkpoint_best.pt` over
  `replay_evidence_best.npz`, real horizon 30, seed 17, 64 model samples, cap
  760;
- the complete 16-cell representation/heads/critic/actor cross on those same
  labels and evidence, CPU, one Torch thread, seed 17, 64 model samples.

The fixed label policy and evidence prevent the target from changing between
cells. Stop after the matrix and rollout summaries. If matched final components
retain the best policy's real action ordering while the deployed final actor
does not, select policy/representation tracking. If model action values lose
ordering first, select imagined dynamics/value targets. Cross-coordinate cell
failures alone are not evidence that one module is broken.

### Best-evidence diagnostic result

The frozen offline diagnostics completed from clean commit `a870eb2`. Matched
rollout fidelity took 6:23.37 and peaked at 607 MiB RSS. The best policy averaged
443.85 over seeds 17--36; final averaged 137.35. On each policy's own states,
the final model had *lower* one-step state MSE (`0.121` versus `0.218`) and
higher one-step target correlation (`0.880` versus `0.625`). Continuation Brier
error worsened (`0.00775` versus `0.00195`), but these cohorts have different
state and terminal distributions. The result rejects a simple global loss of
world-model fidelity; it cannot establish policy-conditioned causality.

Fixed-label generation over best replay evidence took 4:25.55, peaked at 435
MiB, and produced 317 actionable rows among 3,072 trained rows. The complete
16-cell matrix took 5:26.47 with 1,095 MiB peak RSS. On the immutable best-policy
states and real continuation labels:

| Cell | Actor BA | Posterior BA | One-step prior BA | Full dream BA |
|---|---:|---:|---:|---:|
| all best | 0.866 | 0.647 | 0.891 | 0.890 |
| final representation, best heads/critic/actor | 0.855 | 0.175 | 0.481 | 0.861 |
| final representation/critic, best heads/actor | 0.855 | 0.752 | 0.866 | 0.864 |
| all final | 0.863 | 0.752 | 0.842 | 0.866 |

The matched final system retains useful best-policy action ordering. Swapping
only the latent representation breaks the old critic coordinate system, while
the matched final representation and critic restore it. That is coadaptation,
not a portable broken critic. The final actor also remains aligned on the best
policy's states. Consequently, the 338.65-point behavioral collapse is not
encoded as one independently failed component on the old trajectory support.

This selects the policy-conditioned coverage/recovery boundary: small policy
changes move deployment onto histories not represented by the best-evidence
panel, where an error can compound even though the old corridor remains solved.
One final offline cohort is authorized before any intervention. Reuse the best
checkpoint as the immutable real continuation policy, but label
`replay_evidence_final.npz`; then run the same complete 16-cell cross with CPU,
one Torch thread, horizon 30, seed 17, 64 samples, and cap 760. This asks which
components preserve the trusted best controller's recovery ordering on states
actually visited by the collapsed policy. Stop after this cohort and select
the first failed boundary; do not train or tune.

### Final-evidence recovery result and trajectory-trace contract

The fixed best-controller labeling pass on final evidence completed from clean
commit `a60444d` in 4:25.94 with 438 MiB peak RSS. It found 287 actionable rows
among 3,072. The best actor and full imagined values retained balanced accuracy
`0.880` and `0.889`, so the collapsed policy's replay states are not inherently
outside the trusted controller's local recovery competence.

The corresponding 16-cell matrix completed in 4:56.82 with 1,092 MiB peak RSS:

| Cell | Actor BA | Posterior BA | One-step prior BA | Full dream BA |
|---|---:|---:|---:|---:|
| all best | 0.880 | 0.637 | 0.894 | 0.892 |
| final representation, best heads/critic/actor | 0.845 | 0.225 | 0.421 | 0.847 |
| final representation/critic, best heads/actor | 0.845 | 0.804 | 0.879 | 0.864 |
| all final | 0.877 | 0.804 | 0.880 | 0.882 |

Again, matching the final representation and critic restores their coordinate
system, and the final actor retains the best controller's local recovery
ordering. Neither old-corridor coverage nor isolated final-state recovery
labels reproduce the 338.65-point closed-loop collapse. Thirty-step branch
labels hold the continuation policy fixed; actual deployment does not. Rare
action changes alter the next state and therefore the later target, latent, and
action repeatedly.

The next bounded diagnostic is the existing final-policy trajectory trace, not
a training change. First generate final-controller continuation labels over the
same `replay_evidence_final.npz` (final checkpoint, horizon 30, seed 17, 64
samples, cap 760). Then select 32 histories tied under best continuation but
actionable under final continuation, seed 23, and trace both first-action
branches for 30 real steps with 64 final-prior samples. At every successor,
cross best/final actors against best/final representations. Stop after the
trace. Representation transfer selects recurrent policy-state drift; actor
transfer selects policy-head drift; neither selects a closed-loop evaluation
cross as the next diagnostic.

The final-controller label pass completed normally in 4:14.61 with 435 MiB peak
RSS. On its own continuation target, the failed final policy is highly
self-consistent: actor balanced accuracy `0.969`, posterior critic `0.818`, and
full imagined values `0.973`. The preregistered 32-history trace then failed
before evaluation because only four histories satisfy the frozen final-only
selection rule. No artifact was partially interpreted.

The bounded correction is to trace all four eligible histories with the same
checkpoints, evidence, labels, seed, horizon, samples, and actor cross. This is
an exhaustive cohort, not a smaller random sample. Its limitation must remain
explicit; no population conclusion may rest on four histories.

### Closed-loop trace result and selected engineering boundary

The exhaustive four-history trace completed from clean commit `b71b1a4` in
1.60 seconds with 386 MiB peak RSS. Across eight real first-action branches and
206 nonterminal successor decisions, matched best/final actions agree only
`0.641`; every branch diverges by depth eight, at mean depth `4.75`. Among 74
changed decisions, swapping final actor weights onto the best representation
transfers the final action `0.311` of the time, while swapping the final
representation under the best actor transfers it `0.568`. This weakly points to
recurrent representation/policy-state drift, but the four-history selection is
too small for a population claim.

The sharper mechanistic warning is prior continuation on the five actual
terminal transitions: final imagination predicts mean continuation `0.989`,
higher than `0.977` on the 206 nonterminal transitions. The final controller is
therefore locally self-consistent with its actor/value target while its model
can label rare closed-loop failure branches as especially safe. That is model
exploitation, not evidence that actor optimization stopped running.

This warning does not authorize class-balanced continuation. The retained
balanced-BCE canary already showed why: it improved terminal classification by
changing the learned class prior, then fed the uncorrected score into
imagination as a probability and destroyed live-state discount calibration.
Nor do five terminal rows establish that continuation alone is the root cause.

The selected engineering boundary is the remaining pinned-reference replay
carry mismatch. Official DreamerV3 stores and refreshes model carry entries in
replay and uses them as context. This implementation reconstructs carry from
zero over 20 rows. Its initialized-model gate passed, but trained best/final
feature p95 relative errors are `0.411`/`0.319`; every subsequent critic,
continuation, and actor measurement is therefore conditioned on a training
latent that can differ materially from deployment.

## Phase 3: exact replay carry conformance

### Scale-only requalification gate

The no-mistakes audit corrected an important evidence-labeling error: pinned
DreamerV3 RMSNorm learns scale only. The 3,500-update v1 canary used an extra
learned shift, so it is a near-reference instability result rather than an
exact-reference qualification. Its contract, checkpoints, parameter count,
and conclusions remain immutable and load through the explicit
`reference_v3_state_v1` compatibility architecture. Corrected checkpoints use
`reference_v3_state`; resume never silently converts between the two unless
semantic migration is explicitly requested.

Before cached carry changes training, run the separately frozen
`reports/contracts/cartpole_reference_v3_state_v2.yaml` canary:

- **Hypothesis:** removing the non-reference RMSNorm shifts is insufficient to
  eliminate the acquire-then-collapse failure under otherwise identical data,
  optimization, and evaluation semantics.
- **Causal variable:** `rmsnorm_learned_shift=true` to `false`; trainable
  parameters change from 639,173 to 637,381. Every authored run setting and
  behavior gate remains identical to v1.
- **Source:** implementation commit `dc076da`, run source commit `121f1dd`,
  pinned upstream `e3f02248693a79dc8b0ebd62c93683888ddaccfe`, ROCm, training
  seed 0.
- **Budget and metrics:** 3,500 updates, replay ratio 16 trained rows per
  decision, batch 8, sequence 32 with 20 context rows, 20 deterministic
  evaluation episodes every 100 updates, and periodic/best/final checkpoints
  with 256 replay evidence sequences.
- **Pass gate:** reach 475, never fall below 400 afterward, finish at least
  475, and keep best-to-final gap at most 25.
- **Stop rule:** one seed-0 run. Failure authorizes trained carry parity and
  exact cached replay carry, not tuning or more seeds. Passing authorizes only
  unchanged seeds 1 and 2.

This rerun is necessary even though scale-only RMSNorm is unlikely to explain
the entire 338.65-point collapse: scientific attribution requires measuring
the corrected architecture before combining it with the replay-carry repair.

### Corrected v2 result and carry diagnosis

The v2 seed-0 run used the frozen settings and reached all 3,500 optimizer
updates on clean source `121f1dd`. Its manifest run ID is
`ea4298e0eec6402b96b60be96f340dd8`, MLflow run ID
`b4e461097bb841afbea1b3dd5e425f3e`, and it took 27:36.67 with 3,349,780 KiB
peak RSS. Replay descriptor drops remained zero and the collector/trainer
processes stayed healthy. ROCm then reported a GPU hang during finalization,
before the final evaluation and checkpoint could be written; the manifest is
therefore an interrupted run, not a normally completed canary.

The behavioral evidence is nevertheless decisive for the frozen gate. The
evaluation curve was:

| Update | 600 | 700 | 900 | 1,000 | 1,200 | 1,900 | 2,100 | 2,300 | 2,800 | 3,300 | 3,400 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean return | 10.45 | 74.45 | 85.20 | 120.40 | 225.25 | 416.05 | 497.35 | 241.00 | 500.00 | 500.00 | 416.75 |

It first crossed 475 at update 2,100, then fell to 241 at 2,300. The formal
retention gate therefore fails, even though the policy later reacquired 500.
The scale-only correction improves acquisition relative to v1 but does not
remove the acquire/lose/reacquire cycle. Seeds 1 and 2 remain unauthorized.

The surviving best checkpoint (step 2,800) and periodic step-3,000 checkpoint
both fail trained carry parity. At 20 rows, best has median feature cosine
`0.98853`, p95 relative L2 `0.34409`, and actor agreement `0.96403`; step 3,000
has `0.98984`, `0.28936`, and `0.98561`. Extending to 24 rows improves feature
cosine but still fails the p95 L2 gate (`0.28481` and `0.22896`). The carry
diagnostics are off-policy random CartPole prefixes, so they do not prove that
carry mismatch is the sole cause; they do prove the initialized parity gate is
not retained by training. The missing final checkpoint is a hardware-evidence
limitation, not a reason to reinterpret the best/periodic results.

### Corrected v3 cached-carry result

The frozen cached-carry contract ran to completion on seed 0 with source
`f69c8eb` and MLflow run `ccedbee86d4c47de806117a58299e2aa`. Its manifest ID is
`24dfedc9f910454da12038cbd02407bb`, runtime was 26:57.90, peak RSS was
3,914,744 KiB, and the run collected 21,365 environment steps. It wrote the
periodic, best, final, and replay-evidence artifacts and stopped normally at
update 3,500.

The evaluation curve retained the same repeated acquisition/loss pattern:

| Update | 600 | 900 | 1,100 | 1,900 | 2,000 | 2,200 | 2,300 | 2,400 | 2,800 | 2,900 | 3,100 | 3,200 | 3,300 | 3,400 | 3,500 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean return | 56.30 | 18.95 | 118.45 | 464.20 | 356.95 | 229.35 | 462.35 | 301.05 | 475.55 | 333.80 | 489.70 | 500.00 | 500.00 | 500.00 | 304.20 |

The first post-solve evaluation fell below the 400 retention floor (464.20 to
356.95), and later collapses recurred. Best was 500 at update 3,200, while the
final score was 304.20, a 195.80 best-to-final gap. The frozen pass therefore
fails decisively. Compared with v2, the timing and magnitude of acquisition
shift, but the failure boundary is unchanged: useful control is learned and
then lost during continued joint training.

Operationally, the cache behaved as designed: stale writebacks remained zero,
availability was generally 75--100%, and the cache reached 20,842 rows / 50.88
MiB of NumPy payload by update 3,475. The higher peak RSS versus v2 (about
565 MiB) is a real cost, but not the cause of this run's failure; the process
completed without resource or child-process errors.

The post-run random-prefix carry probe also failed its trained-parity gate:
the best checkpoint had median cosine `0.98923`, p95 relative L2 `0.31056`, and
actor agreement `0.98561`; the final checkpoint had `0.98865`, `0.36072`, and
`0.99281`. These probes are off-policy diagnostics and do not isolate the
cause of the behavioral collapse, but they confirm that checkpoint carry
quality remains imperfect even after the cache repair.

This rejects cached carry as the next behavioral fix. Do not run seeds 1/2 or
Pong. The next investigation should compare the actor/critic targets at the
first-solve versus post-collapse checkpoints, with emphasis on imagined prior
continuation and value calibration, while keeping the v3 run as a negative
replay-boundary result.

### First-solve versus post-collapse target probe

The read-only `probe_cartpole_q.py` comparison used the same 256 random
CartPole states, 58 action-discriminative simulator labels, and seed 17 for
v3 checkpoints. The actor and learned policy-Q retain moderate action ranking,
but the model-derived one-step transition/continuation branches are nearly
uninformative:

| Checkpoint | Actor vs rollout | Policy-Q vs rollout | Q/true delta Pearson | Hybrid state | Hybrid continuation | Actor entropy |
|---|---:|---:|---:|---:|---:|---:|
| step 2,000 | 0.776 | 0.776 | 0.658 | 0.069 | 0.121 | 0.336 |
| step 2,500 | 0.776 | 0.776 | 0.577 | 0.052 | 0.086 | 0.299 |
| step 3,000 | 0.776 | 0.759 | 0.585 | 0.121 | 0.155 | 0.271 |
| final 3,500 | 0.759 | 0.759 | 0.529 | 0.086 | 0.086 | 0.237 |

Here “hybrid state/continuation” means action preference from one learned prior
step evaluated with trusted simulator continuation; values near 0.1 are far
below the actor/Q ranking and point at the transition/continuation boundary.
The actor becomes steadily lower-entropy while Q agreement and Q/true
correlation weaken. This is the evidence-selected next probe: separate prior
transition error, continuation calibration, and value-target drift at the
first-solve and post-collapse checkpoints before changing losses.

### Continuation calibration and latent representability result

The matched continuation probe was repeated from clean source `d31cf8d` on the
v3 step-2,000, step-3,000, and final checkpoints using CUDA execution,
evaluation seeds 17--36, 20 episodes, and 64 posterior/prior samples. The
durable artifacts are under
`experiments/2026-07-26_cartpole_cached_carry_continuation_probe_cuda20/`.
The result is stable across the larger cohort:

| Checkpoint | Transitions / terminals | Prior continuation, live | Prior continuation, terminal | Prior Brier | Prior failure ROC-AUC |
|---|---:|---:|---:|---:|---:|
| step 2,000 | 7,385 / 18 | 0.9717 | 0.9464 | 0.0030 | 0.904 |
| step 3,000 | 5,758 / 20 | 0.9798 | 0.9647 | 0.0037 | 0.844 |
| final 3,500 | 5,795 / 20 | 0.9852 | 0.9583 | 0.0035 | 0.921 |

The prior and observation-conditioned posterior differed by only `0.0006` to
`0.0021` mean absolute effective-discount transport error on these cohorts.
The problem is therefore not prior-to-posterior transport. Both channels
assign almost the same high continuation to physical terminal transitions;
the model supplies no useful terminal discount to imagination. The high ROC-AUC
is only ranking: at the default 0.5 discount cutoff, terminal recall remains
zero because the output never approaches the required terminal value of zero.

The frozen-head representability probe used 100 episodes from the final
checkpoint, 80/20 episode split, and the exact continuation-head topology. Its
artifacts are under
`experiments/2026-07-26_cartpole_cached_carry_continuation_supervision/`. The
jointly trained head had failure ROC-AUC `0.901`, but mean failure probability
`0.035` on terminals versus `0.013` on live rows, so its natural-prior output
is under-confident. A freshly fitted class-balanced head on the same posterior
latents reached ROC-AUC `0.979` and balanced accuracy `0.925`; a head on the
true CartPole state reached `0.999` and `0.996`. Thus terminal information is
present in the latent, but the learned continuation head does not turn it into
a calibrated Bernoulli probability during joint training. A prior-corrected
balanced head still averaged only `0.065` terminal failure probability, so a
class-weighted loss must not be used as a raw imagination discount.

This narrows the boundary: **terminal calibration is a real model weakness,
but not the sole source of the collapse**. It explains why imagined rollouts
remain alive after entering a failure state, while the low hybrid transition
scores show that the one-step prior is also wrong. The rejected adaptive
class-balanced canary remains valid as a behavioral negative: its raw weighted
output shortened live imagination. The next intervention must either preserve
the natural prior explicitly (for example, a balanced auxiliary terminal
score with a separately calibrated discount head) or target prior transition
fidelity first. No Pong or multi-seed run is authorized by this read-only
result.

### Phase 4 preregistration: one-step prior-state supervision

The matched rollout probe rejects global transition degradation: the final
checkpoint has lower one-step state error than step 2,000 on its own policy
trajectory, while the counterfactual hybrid probe remains poor. The next
controlled intervention therefore targets the prior's action-conditioned
one-step branch rather than changing the actor or continuation semantics.

- **Hypothesis:** the existing posterior decoder loss leaves the expected
  one-step prior state weakly constrained, allowing policy improvement to use
  an action-ranking target that is locally self-consistent but wrong for the
  unchosen action. The already-authored `prior_state_pred_scale` loss will
  improve counterfactual state fidelity and reduce the collapse.
- **Causal variable:** set only `train.prior_state_pred_scale=1.0` (baseline is
  `0.0`). This adds the expected-prior decoded-state symlog MSE to the world
  model loss. Preserve the v3 cached-carry contract, optimizer, replay,
  continuation head, actor loss, seed, evaluation cohort, and all budgets.
- **Run:** clean source `3170c68`, CUDA, seed `0`, 3,500 updates, batch `8`,
  sequence `32`, burn-in `20`, replay ratio `16`, cached carry, evaluation
  every `100` updates over 20 episodes. Artifact directory:
  `experiments/2026-07-26_cartpole_reference_v3_state_prior_state_loss_seed0_3500/`.
- **Behavior gate:** reach mean return `450`, never fall below `300` after
  first reaching it, finish at least `400`, and keep best-to-final gap at most
  `100`. Failure stops this seed and authorizes no seeds or Pong.
- **Boundary readout:** at steps `2,000`, `3,000`, and final, repeat the same
  q/true-action probe (256 states, 58 action-discriminative labels, seed 17).
  Report one-step hybrid state/continuation ordering, actor/Q agreement,
  one-step state MSE, and continuation calibration. A behavioral failure with
  improved hybrid state ordering rejects this loss as sufficient but retains
  it as a model-fidelity effect; no behavior or boundary improvement rejects
  the intervention outright.

The carry repair is implemented and covered by focused replay, forward-pass,
configuration, and full-suite tests. The frozen v3 behavioral rerun is now
complete and rejected as the next behavioral fix. Its implementation evidence
is:

1. Give every replay row a stable identity and cache the detached RSSM carry
   produced at that row. Sample the cached entry at the burn-in endpoint and
   return refreshed entries after training, matching the pinned
   `stepid -> enc/dyn/dec` update contract.
2. Invalidate or reset entries at genuine episode boundaries and eviction;
   reject stale identities rather than applying an update to a reused slot.
3. Preserve the current historical and bounded-burn contracts for checkpoint
   compatibility. The reference contract alone selects cached carry.
4. The implementation adds deterministic tests for stable IDs, cache writeback,
   stale-update rejection, eviction, mixed cache availability, and the forward
   path. Cache entry count, stale updates, availability, and payload memory are
   logged. On the CartPole contract, a full cache can hold roughly 256,000 rows;
   with the current 640-float carry this is about 625 MiB of NumPy payload before
   Python/container overhead. The telemetry is therefore part of the acceptance
   evidence, not an optional optimization detail.
5. Completed: the frozen seed-0 qualification reran with cached carry as the
   only causal change, using the same 3,500 updates, replay ratio, trained rows
   per update, environment-step authorization, evaluation seeds, and retention
   gate. It failed; do not run seeds 1/2.

If exact carry still collapses, the next isolated intervention is direct
one-step **prior** continuation supervision on real transitions, because
imagination consumes prior rather than posterior features. It must preserve the
natural continuation probability and be gated on held-out prior Brier error,
terminal/live calibration, and behavior. Do not revisit balanced BCE, generic
loss scaling, Pong, or hyperparameter sweeps before that decision point.

### Phase 4 one-step prior-state supervision result

The preregistered intervention completed from source `970c89c`. The first
weight run was interrupted by a turn abort at update `1,200`, after saving its
best checkpoint at update `1,100`; it was not a trainer or collector failure.
The run was resumed from that checkpoint with fresh replay and completed to
update `3,500`. This is therefore a weight-resume result, not an exact
uninterrupted trajectory replication. The resumed manifest is
`3b99f4b174c64036851efdb988d2058c` and the MLflow run is
`f9d69d3435ba47be8afb0d24de7ceec7`.

The resumed evaluation curve repeatedly acquired CartPole and repeatedly
lost it:

```text
1200: 172.8, 1300: 296.3, 1400: 233.65, 1500: 264.9,
1600: 342.9, 1700: 370.45, 1800: 323.5, 1900: 251.65,
2000: 261.65, 2100: 194.25, 2200: 345.7, 2300: 350.55,
2400: 241.75, 2500: 470.8, 2600: 237.15, 2700: 471.55,
2800: 500.0, 2900: 242.65, 3000: 500.0, 3100: 415.65,
3200: 490.4, 3300: 249.05, 3400: 198.1, 3500: 136.45
```

The best score was `500.0` at update `2,800`, but the final score was
`136.45`, giving a best-to-final gap of `363.55`. The behavior gate therefore
fails decisively; no seeds or Pong run is authorized.

The boundary probe is retained under
`experiments/2026-07-26_cartpole_prior_state_q_probe/`. Relative to the
cached-carry baseline, the auxiliary loss lowers sampled one-step decoded-state
MSE at steps `2,000`, `3,000`, and final (`0.0246`, `0.0465`, `0.0258` versus
`0.1004`, `0.0654`, `0.0623`). This confirms a model-fidelity effect. It does
not repair action-conditioned control: hybrid state ordering is only
`0.175`, `0.200`, and `0.300`, and hybrid continuation ordering is `0.300`,
`0.300`, and `0.425`. The actor remains highly aligned with the sampled
rollout preference (`0.85` at all three checkpoints), while Q ordering falls
from `0.825` to `0.550` and recovers only to `0.625`.

This rejects one-step prior-state supervision as a sufficient stability fix.
It also sharpens the diagnosis: better local state prediction does not produce
better counterfactual action values. The remaining problem is downstream in
the policy-conditioned value/continuation target, or in how that target is
optimized and retained. The next intervention must preserve the natural
continuation prior and directly measure held-out terminal calibration; do not
add seeds, tune the actor, or return to Pong.

### Phase 5 preregistration: natural-prior one-step continuation supervision

The prior-state canary improved decoded state error without improving
counterfactual action ordering. The next intervention therefore trains the
existing continuation head on the expected one-step **prior** latent used by
imagination, rather than only on the observation-conditioned posterior latent.

- **Hypothesis:** the continuation head is calibrated on posterior states but
  optimistic on the prior states that imagination visits. A direct prior-head
  loss on real transitions will reduce terminal optimism without changing the
  natural terminal/live prevalence.
- **Causal variable:** add `train.prior_continue_pred_scale=1.0`; baseline is
  `0.0`. Compute the expected prior categorical latent with the same 1% unimix,
  join it with the observed recurrent state, and apply the ordinary natural
  continuation target and `contdisc` semantics. Do not class-balance, alter
  actor/critic losses, add terminal rewards, or change replay/carry settings.
- **Frozen contract:** current v3 state architecture, cached carry, seed `0`,
  3,500 updates, batch `8`, sequence `32`, burn-in `20`, replay ratio `16`,
  20 deterministic evaluation episodes every `100` updates, and the same
  checkpoints and resource limits as the prior-state canary. Artifact prefix:
  `experiments/2026-07-26_cartpole_reference_v3_prior_continue_loss_seed0_3500`.
- **Behavior gate:** reach mean return `450`, never fall below `300` after
  first reaching it, finish at least `400`, and keep the best-to-final gap at
  most `100`. Failure stops this seed and authorizes no seeds or Pong.
- **Calibration gate:** run the held-out 20-episode, seeds `17--36`, 64-sample
  continuation probe at steps `2,000`, `3,000`, and final. At final, require
  prior Brier error no worse than the cached-carry reference (`0.00351`), mean
  terminal continuation at most `0.95`, and mean live continuation at least
  `0.98`. These are model-target gates, not substitutes for behavior.
- **Decision:** behavior plus calibration passes select this as the next
  current CartPole correction. Calibration improvement with behavioral failure
  rejects it as insufficient but retains the target-localization evidence.
Behavioral success without calibration improvement is treated as an
  incidental canary and does not authorize Pong.

### Phase 5 natural-prior continuation result

The one-seed canary completed normally from source `134677d`. Manifest
`9937969baeda443092cc349d16aa76b5` and MLflow run
`c6ba47ed906844a79965fb7f6cffee92` record 3,500 updates, 21,435 environment
steps, and normal collector shutdown.

The intervention improved acquisition and recovery but failed the behavioral
gate. The curve reached `500.0` at update `2,000`, held `488--500` through
update `2,400`, then oscillated through `274.8`, `455.8`, `495.95`, `363.4`,
`301.4`, `492.8`, `248.85`, `488.4`, `482.65`, `141.05`, and `355.95` at
updates `2,500--3,500`. The best-to-final gap was `144.05`, and the run fell
below `300` after solving.

The held-out continuation probe is retained under
`experiments/2026-07-26_cartpole_prior_continue_calibration_probe/`. At final,
prior Brier error was `0.003008`, which passes the preregistered Brier bound,
and live continuation was `0.9927`, which passes its live bound. However,
terminal continuation was `0.9915`, far above the `0.95` terminal bound and
even worse than the cached-carry reference's `0.9583`; failure ROC-AUC fell to
`0.642`. The intervention therefore did not calibrate rare terminal states.

The Q probe under
`experiments/2026-07-26_cartpole_prior_continue_q_probe/` shows why the run
could look promising before collapsing. At steps `2,000`, `3,000`, and final,
Q/true-delta correlation was `0.759`, `0.771`, and `0.825`, while one-step
decoded-state MSE was `0.0645`, `0.0404`, and `0.0152`. Hybrid continuation
ordering improved to `0.525` at final, but remained far below actor agreement
(`0.85`) and was not stable across checkpoints (`0.425`, `0.275`, `0.525`).

Disposition: reject the direct prior-continuation loss as a sufficient fix.
It improves local transition/value readouts and can produce a solved-looking
checkpoint, but it does not preserve terminal calibration or closed-loop
retention. The result rules out “the prior head simply lacks a BCE gradient”
as the complete explanation. Do not run seeds 1/2 or Pong. The next step is a
read-only source/data-flow audit of how rare terminal evidence reaches the
prior continuation target and whether the policy-conditioned value update
amplifies optimistic states before another training intervention.

### Phase 5 source/data-flow audit

The pinned upstream source confirms that Dreamer trains one natural-prior
continuation head on posterior features with the target `~obs['is_terminal']`
and consumes that head on imagined features; it does not add a separate prior
continuation loss. The local direct-prior term was therefore an intentional
diagnostic intervention, not a reference-conformance repair.

The final prior-continuation run's replay-evidence terminal fraction was
`2.1--4.3%` across retained checkpoints. In contrast, its held-out deployed
policy probe contained only `19/7342` physical terminal transitions (`0.26%`)
at final. The training stream sees terminal rows, but not the same
policy-conditioned failure corridor that emerges after the actor drifts.

The final checkpoint's 100-episode frozen-latent probe retains terminal
information: a freshly fitted balanced latent head reaches failure ROC-AUC
`0.965`, balanced accuracy `0.883`, and mean failure probability `0.759` on
held-out terminal rows versus `0.027` on live rows. A true-state head reaches
ROC-AUC `0.994`. The jointly trained original continuation head reaches only
ROC-AUC `0.753`, with terminal/live failure probabilities `0.0074`/`0.0059`
and zero terminal recall at the natural 0.5 cutoff.

This rejects static information absence and simple prior transport loss as the
proximal explanation. The current boundary is **rare, policy-conditioned
terminal calibration under joint representation/value drift**: the latent can
separate failure when a fresh head is fit, but the online head is dominated by
live examples and becomes optimistic on the states generated by its changing
policy. No further training intervention is selected from this audit alone.
The next preregistration must isolate whether a separate terminal-risk
auxiliary signal can improve the shared representation without replacing the
natural-probability discount head. Seeds 1/2 and Pong remain stopped.

### Phase 6 preregistration: detached terminal-risk auxiliary head

The audit shows that a fresh balanced head can read terminal information from
the latent while the natural continuation head remains nearly all-live. The
next intervention tests representation support without using a reweighted
classifier as an imagination discount.

- **Hypothesis:** a separate terminal-risk head trained with balanced terminal
  exposure will improve prior latent separation and stabilize the existing
  natural continuation head through shared RSSM gradients. The risk score is a
  diagnostic/representation auxiliary only; Dreamer continues to discount with
  the ordinary natural-prior continuation head.
- **Causal variable:** add `train.terminal_risk_aux_scale=0.25` (baseline `0.0`).
  The new head consumes the expected one-step prior feature, uses per-batch
  equal terminal/live mass weighting, and receives gradients through the prior
  representation. Do not alter the natural continuation head, actor/critic
  objectives, replay, carry, terminal rewards, or actor entropy.
- **Frozen contract:** current v3 cached-carry CartPole state contract, seed `0`,
  3,500 updates, batch `8`, sequence `32`, burn-in `20`, replay ratio `16`,
  20 deterministic evaluations every `100` updates, and the same resource cap.
  Artifact prefix:
  `experiments/2026-07-26_cartpole_reference_v3_terminal_risk_aux_seed0_3500`.
- **Behavior gate:** reach mean return `450`, never fall below `300` after
  first reaching it, finish at least `400`, and keep best-to-final gap at most
  `100`. Failure stops this seed and authorizes no seeds or Pong.
- **Natural-discount gate:** held-out prior continuation at final must have
  Brier error no worse than `0.00351`, terminal continuation at most `0.97`,
  and live continuation at least `0.98`. The auxiliary risk score is not
  accepted as a replacement discount.
- **Boundary readout:** report existing Q/true-action metrics plus the
  auxiliary risk head's held-out terminal/live probabilities and failure AUC.
  Improved risk separation with unchanged natural continuation and behavioral
  failure rejects the intervention as insufficient but localizes representation
  support. No seeds or Pong run follows a one-seed partial result.

### Phase 6 terminal-risk auxiliary result

The one-seed canary completed normally from source `0478ada`. Manifest
`024997e9a9df4f7bbea1a12f6ac9aa60` and MLflow run
`38faefc059224a23bb2d034c3d32fb62` record 3,500 updates, 21,358 environment
steps, and peak RSS `3,955,304 KiB`.

The behavior gate failed. The run first exceeded 400 at update 2,700 with
`422.7`, but then oscillated below and above the threshold and finished at
`365.35`; the best score was `422.7`, so it never reached the required 450.
The best-to-final gap was `57.35`, which is better than prior canaries, but
retention is not solved.

The final held-out prior continuation probe reports Brier `0.002627`, terminal
continuation `0.97149`, and live continuation `0.98896`. The Brier/live gates
pass; the terminal gate (`<=0.97`) misses narrowly. Q/true action agreement is
`0.85`, but hybrid state and continuation ordering are only `0.225` and
`0.375`, respectively, with final one-step state MSE `0.0441`.

The new auxiliary head does learn a useful readout: held-out failure ROC-AUC
is `0.698`/`0.808`/`0.952` at steps 2,000/3,000/final; final terminal/live risk
probabilities are `0.629`/`0.208`, with balanced accuracy `0.918`. Thus the
intervention improves terminal representation support but does not make the
natural discount or actor/value loop stable.

**Disposition:** reject as a sufficient fix. Keep the auxiliary risk probe as
a representation regression signal. The evidence-selected next direction is
the policy-conditioned value/actor update and retention boundary, not more
terminal-head weighting. Do not run seeds 1/2 or Pong until a new one-seed
behavior gate is preregistered and passed.
