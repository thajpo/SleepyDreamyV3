# Plan (paper-faithful baseline alignment)

1) Lock in latent shape refactor
- Ensure K = d_hidden // 16, L = num_latents, and all z sizes use L * K.
- Verify no remaining quadratic z sizes anywhere.

2) Add one-step shape verification
- Log/assert: posterior/prior logits [B,T,L,K], z onehot [B,T,L,K], actor/critic input dims = h_dim + L*K.

3) Replay sampling update (online queue + segment-uniform)
- Define "recent" as online queue of non-overlapping segments (length T).
- Batch sampling: n_recent from online queue, remainder uniform over all segments (not episode-uniform).
- Backfill from replay when recent queue is empty.

4) Replay ratio gating
- Add replay_ratio and action_repeat.
- Gate training steps with:
  target_train_steps = env_steps * replay_ratio / (B * T * action_repeat)
- Log actual vs target ratio.

5) Latent replay storage + writeback
- Store z (and optionally h) in buffer per step.
- Initialize replay rollouts from stored latents.
- Write back updated latents after training.

6) Actor unimix + critic stability extras
- Actor unimix (1%).
- Critic replay loss + EMA regularizer.
- Zero-init critic head.

7) Baseline mode config
- Disable non-paper extras (surprise gating, WM:AC schedules, warmups) behind a baseline flag/config.
