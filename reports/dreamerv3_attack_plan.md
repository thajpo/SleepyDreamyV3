# DreamerV3 conformance and reliability plan

Status: active research plan, 2026-07-25

Local baseline: `35d7454c22d34bb9ba72d35c2f9f7254bdfac0cd`

Reference paper: DreamerV3 v2, <https://arxiv.org/abs/2301.04104>

Reference source: `danijar/dreamerv3@e3f02248693a79dc8b0ebd62c93683888ddaccfe`

## Audited conclusion

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

## What remains unresolved

- Whether value predictions lose a stable target or the policy-conditioned
  target itself changes first.
- Which of representation, reward/continuation model, critic, or actor is the
  first moving component.
- Whether replay eligibility translated into sufficient effective gradient
  exposure for the anchor histories.
- Whether the same CartPole behavior occurs in the pinned reference
  implementation.

## Confirmed conformance gaps

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
