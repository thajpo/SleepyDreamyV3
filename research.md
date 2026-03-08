# Research Notes

## Replay Burn-In (Explicit RSSM Warm-Up)

**Idea:** When sampling a sequence from replay, the RSSM hidden state `h` starts at zero. The first few timesteps produce low-quality hidden states because the RNN hasn't accumulated enough context. Training actor-critic on imaginations rooted in those cold-start hidden states wastes gradient signal (or worse, teaches bad policy).

**Implementation:** Run the world model forward on the first N timesteps of each sampled sequence *without* computing actor-critic losses. The WM still receives gradients on burn-in steps (it learns reconstruction/prediction on all timesteps), but AC training only begins at timestep `train_start_t = burn_in_steps` where `h` is meaningful.

**Relation to DreamerV3 paper:** The paper doesn't use explicit burn-in. Instead it relies on long sequences (T=64) so the cold-start timesteps are a small fraction of the total. Explicit burn-in is a more direct approach — it decouples the "how much context does the RNN need" question from the sequence length hyperparameter. This could allow shorter sequences to work better (e.g., T=16 with burn_in=8 instead of T=64).

**Open questions:**
- Does explicit burn-in help more at shorter sequence lengths where cold-start is a larger fraction?
- Should burn-in be adaptive (based on RSSM hidden state convergence) rather than fixed?
- Does the WM benefit from gradients on burn-in timesteps, or should those be detached too?
