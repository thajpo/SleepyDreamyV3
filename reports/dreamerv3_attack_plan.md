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
