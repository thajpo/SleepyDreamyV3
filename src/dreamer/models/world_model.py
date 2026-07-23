import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import ObservationDecoder, StateOnlyDecoder
from .math_utils import unimix_logits


class RSSMWorldModel(nn.Module):
    """
    World model architecture from Dreamerv3, called a 'Recurrent State Space Model'

    The observe step follows the paper's graphical model:
      1. Step GRU: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
      2. Compute posterior: q(z_t | h_t, x_t) from concat(h_t, encoder_tokens)
      3. Compute prior: p(z_t | h_t) from h_t alone
      4. Sample z_t from posterior, predict heads from (h_t, z_t)
    """

    def __init__(
        self,
        models_config,
        env_config,
        batch_size,
        b_start,
        b_end,
        encoder_token_dim,
        num_bins=255,
        use_pixels=True,
    ):
        super().__init__()
        self.d_hidden = models_config.d_hidden
        num_classes = self.d_hidden // 16

        # GatedRecurrentUnit | Uses n_blocks independent GRUs computed in parallel
        self.n_blocks = models_config.rnn.n_blocks
        self.rssm_core = str(getattr(models_config, "rssm_core", "legacy"))
        h_dim = self.d_hidden * self.n_blocks
        stoch_dim = models_config.num_latents * num_classes
        if self.rssm_core == "legacy":
            gru_d_in = self.d_hidden + env_config.n_actions

            # Preserve the parameter layout of historical checkpoints.
            blocks = nn.ModuleList()
            for _ in range(self.n_blocks):
                blocks.append(
                    GatedRecurrentUnit(d_in=gru_d_in, d_hidden=self.d_hidden)
                )

            self._W_ir = nn.Parameter(torch.stack([b.W_ir.weight for b in blocks]))
            self._W_iz = nn.Parameter(torch.stack([b.W_iz.weight for b in blocks]))
            self._W_in = nn.Parameter(torch.stack([b.W_in.weight for b in blocks]))
            self._b_ir = nn.Parameter(torch.stack([b.W_ir.bias for b in blocks]))
            self._b_iz = nn.Parameter(torch.stack([b.W_iz.bias for b in blocks]))
            self._b_in = nn.Parameter(torch.stack([b.W_in.bias for b in blocks]))
            self._W_hr = nn.Parameter(torch.stack([b.W_hr.weight for b in blocks]))
            self._W_hz = nn.Parameter(torch.stack([b.W_hz.weight for b in blocks]))
            self._W_hn = nn.Parameter(torch.stack([b.W_hn.weight for b in blocks]))
            self._b_hr = nn.Parameter(torch.stack([b.W_hr.bias for b in blocks]))
            self._b_hz = nn.Parameter(torch.stack([b.W_hz.bias for b in blocks]))
            self._b_hn = nn.Parameter(torch.stack([b.W_hn.bias for b in blocks]))
            self.z_embedding = nn.Linear(stoch_dim, self.d_hidden)
        elif self.rssm_core == "reference":
            # Reference _core: normalize each input independently, apply one
            # normalized grouped hidden layer, then produce all three gates in
            # one grouped projection.
            self.dynin_deter = nn.Sequential(
                nn.Linear(h_dim, self.d_hidden),
                nn.RMSNorm(self.d_hidden),
                nn.SiLU(),
            )
            self.z_embedding = nn.Sequential(
                nn.Linear(stoch_dim, self.d_hidden),
                nn.RMSNorm(self.d_hidden),
                nn.SiLU(),
            )
            self.dynin_action = nn.Sequential(
                nn.Linear(env_config.n_actions, self.d_hidden),
                nn.RMSNorm(self.d_hidden),
                nn.SiLU(),
            )
            self.dynhid = BlockLinear(
                self.n_blocks,
                4 * self.d_hidden,
                self.d_hidden,
            )
            self.dynhid_norm = nn.RMSNorm(h_dim)
            self.dyngru = BlockLinear(
                self.n_blocks,
                self.d_hidden,
                3 * self.d_hidden,
            )
        else:
            raise ValueError("rssm_core must be 'legacy' or 'reference'")

        # Outputs prior distribution \hat{z} from the sequence model
        n_gru_blocks = models_config.rnn.n_blocks
        self.dynamics_predictor = DynamicsPredictor(
            d_in=self.d_hidden * n_gru_blocks,
            d_hidden=self.d_hidden,
            num_latents=models_config.num_latents,
            num_classes=num_classes,
        )

        self.n_latents = models_config.num_latents
        self.n_classes = num_classes

        # --- Posterior head: q(z_t | h_t, x_t) ---
        # Takes concat(h_t, encoder_tokens) and produces posterior logits.
        # This is the key architectural requirement from the paper.
        posterior_in_dim = h_dim + encoder_token_dim
        posterior_out_dim = self.n_latents * self.n_classes
        posterior_head_layers = int(
            getattr(models_config, "posterior_head_layers", 0)
        )
        if posterior_head_layers == 0:
            # Preserve the parameter layout of historical checkpoints.
            self.posterior_head = nn.Linear(posterior_in_dim, posterior_out_dim)
        elif posterior_head_layers == 1:
            self.posterior_head = nn.Sequential(
                nn.Linear(posterior_in_dim, self.d_hidden),
                nn.RMSNorm(self.d_hidden, eps=1e-4),
                nn.SiLU(),
                nn.Linear(self.d_hidden, posterior_out_dim),
            )
        else:
            raise ValueError("posterior_head_layers must be 0 or 1")

        # Initalizing network params for t=0 ; h_0 is the zero matrix
        h_prev = torch.zeros(batch_size, self.d_hidden * n_gru_blocks)
        self.register_buffer("h_prev", h_prev)
        z_prev = torch.zeros((batch_size, self.n_latents, self.n_classes))
        self.register_buffer("z_prev", z_prev)

        # Takes 2D categorical samples and projects to d_hidden for GRU input
        h_z_dim = (self.d_hidden * n_gru_blocks) + (self.n_latents * self.n_classes)

        # Rewards use two-hot encoding
        reward_out = int(num_bins)
        self.reward_predictor = nn.Linear(h_z_dim, reward_out)
        nn.init.zeros_(self.reward_predictor.weight)
        nn.init.zeros_(self.reward_predictor.bias)
        continue_head_layers = int(
            getattr(models_config, "continue_head_layers", 0)
        )
        if continue_head_layers == 0:
            # Preserve the parameter layout of historical checkpoints.
            self.continue_predictor = nn.Linear(h_z_dim, 1)
        elif continue_head_layers == 1:
            self.continue_predictor = nn.Sequential(
                nn.Linear(h_z_dim, self.d_hidden),
                nn.RMSNorm(self.d_hidden),
                nn.SiLU(),
                nn.Linear(self.d_hidden, 1),
            )
        else:
            raise ValueError("continue_head_layers must be 0 or 1")

        # Decoder. Outputs distribution of mean predictions for pixel/vector observations
        if use_pixels:
            self.decoder = ObservationDecoder(
                d_in=h_z_dim,
                mlp_config=models_config.encoder.mlp,
                cnn_config=models_config.encoder.cnn,
                env_config=env_config,
                d_hidden=models_config.d_hidden,
            )
        else:
            self.decoder = StateOnlyDecoder(
                d_in=h_z_dim,
                d_hidden=models_config.d_hidden,
                n_observations=env_config.n_observations,
            )

    def step_dynamics(self, z_embed, action, h_prev):
        """
        Step the configured recurrent dynamics core once.

        The legacy mode preserves historical checkpoint equations. Reference
        mode follows the grouped, normalized DreamerV3 core. Both modes compute
        every recurrent block in parallel.

        Args:
            z_embed: The embedded latent state z_t. Shape: (B, d_hidden)
            action: The one-hot encoded action a_t. Shape: (B, n_actions)
            h_prev: The previous hidden state h_{t-1}. Shape: (B, n_blocks * d_hidden)

        Returns:
            h: The new hidden state h_t. Shape: (B, n_blocks * d_hidden)
            prior_logits: The predicted logits for the next latent state.
        """
        B = z_embed.shape[0]
        h_blocks = h_prev.reshape(B, self.n_blocks, self.d_hidden)
        if self.rssm_core == "legacy":
            x = torch.cat((z_embed, action), dim=1)
            ir = torch.einsum("bi,kji->bkj", x, self._W_ir) + self._b_ir
            iz = torch.einsum("bi,kji->bkj", x, self._W_iz) + self._b_iz
            in_ = torch.einsum("bi,kji->bkj", x, self._W_in) + self._b_in
            hr = torch.einsum("bki,kji->bkj", h_blocks, self._W_hr) + self._b_hr
            hz = torch.einsum("bki,kji->bkj", h_blocks, self._W_hz) + self._b_hz
            hn = torch.einsum("bki,kji->bkj", h_blocks, self._W_hn) + self._b_hn
            reset = torch.sigmoid(ir + hr)
            candidate = torch.tanh(reset * (in_ + hn))
            update = torch.sigmoid(iz + hz - 1.0)
        else:
            deter_input = self.dynin_deter(h_prev)
            action_input = self.dynin_action(action)
            shared_inputs = torch.cat(
                (deter_input, z_embed, action_input), dim=-1
            ).unsqueeze(1)
            shared_inputs = shared_inputs.expand(-1, self.n_blocks, -1)
            hidden_input = torch.cat((h_blocks, shared_inputs), dim=-1)
            hidden = self.dynhid(hidden_input).reshape(B, -1)
            hidden = F.silu(self.dynhid_norm(hidden))
            gate_logits = self.dyngru(
                hidden.reshape(B, self.n_blocks, self.d_hidden)
            )
            reset_logits, candidate_logits, update_logits = gate_logits.chunk(
                3, dim=-1
            )
            reset = torch.sigmoid(reset_logits)
            candidate = torch.tanh(reset * candidate_logits)
            update = torch.sigmoid(update_logits - 1.0)

        h_new = update * candidate + (1.0 - update) * h_blocks
        h = h_new.reshape(B, -1)
        # Preserve the observed-sequence graph. The trainer resets this carry at
        # each sampled sequence, so retaining it here gives losses at later rows
        # gradient paths through the earlier recurrent states (BPTT).
        self.h_prev = h
        prior_logits = self.dynamics_predictor(h)
        return h, prior_logits

    def compute_posterior(self, h, tokens):
        """
        Compute posterior logits q(z_t | h_t, x_t) by conditioning on both
        the deterministic state and the encoder tokens.

        Args:
            h: Deterministic state h_t. Shape: (B, n_blocks * d_hidden)
            tokens: Encoder token features. Shape: (B, token_dim)

        Returns:
            posterior_logits: Shape (B, num_latents, num_classes)
        """
        x = torch.cat([h, tokens], dim=-1)
        logits = self.posterior_head(x)
        return logits.view(logits.shape[0], self.n_latents, self.n_classes)

    def predict_heads(self, h, z_sample):
        """
        Generates predictions from the model's state (h, z).
        """
        h_z_joined = self.join_h_and_z(
            h, z_sample
        )  # This is the state for actor/critic
        obs_reconstruction = self.decoder(h_z_joined)
        reward_logits = self.reward_predictor(h_z_joined)
        continue_logits = self.continue_predictor(h_z_joined)
        return obs_reconstruction, reward_logits, continue_logits, h_z_joined

    def forward(self, tokens, action, is_first=None):
        """
        Performs a full world model observe step for training.

        Follows the paper's graphical model:
          1. Step GRU with (z_{t-1}, a_{t-1}) -> h_t
          2. Compute posterior q(z_t | h_t, tokens) -> posterior_logits
          3. Sample z_t from posterior (straight-through)
          4. Predict heads from (h_t, z_t)

        Args:
            tokens: Encoder token features for current observation. Shape: (B, token_dim)
            action: One-hot action. Shape: (B, n_actions)
            is_first: Optional boolean reset mask for rows that begin an episode.

        Returns:
            obs_reconstruction, reward_logits, continue_logits, h_z_joined,
            z_sample, prior_logits, posterior_logits
        """
        # A sampled stream may cross an episode reset. Reset only the affected
        # batch rows so BPTT remains intact within each episode and cannot leak
        # across environment boundaries.
        if is_first is not None:
            reset = is_first.bool()
            if reset.ndim != 1 or reset.shape[0] != self.h_prev.shape[0]:
                raise ValueError("is_first must have shape (batch_size,)")
            keep = (~reset).to(self.h_prev.dtype).unsqueeze(-1)
            self.h_prev = self.h_prev * keep
            self.z_prev = self.z_prev * keep.unsqueeze(-1)

        # 1. Step GRU using PREVIOUS z and action
        z_prev_flat = self.z_prev.view(self.z_prev.size(0), -1)
        z_prev_embed = self.z_embedding(z_prev_flat)
        h, prior_logits = self.step_dynamics(z_prev_embed, action, self.h_prev)

        # 2. Compute posterior conditioned on h_t AND observation tokens
        posterior_logits = self.compute_posterior(h, tokens)

        # 3. Sample z_t from posterior using straight-through estimator
        posterior_logits_mixed = unimix_logits(
            posterior_logits, unimix_ratio=0.01
        )
        posterior_probs = F.softmax(posterior_logits_mixed, dim=-1)
        posterior_dist = torch.distributions.Categorical(
            probs=posterior_probs, validate_args=False
        )
        z_indices = posterior_dist.sample()  # (batch_size, latents)
        z_onehot = F.one_hot(z_indices, num_classes=self.n_classes).float()
        z_sample = z_onehot + (posterior_probs - posterior_probs.detach())
        # Cache z_t without severing the observed-sequence graph. Collection and
        # evaluation run under no_grad; training needs this path for BPTT.
        self.z_prev = z_sample

        # 4. Generate predictions using the new state
        (obs_reconstruction, reward_logits, continue_logits, h_z_joined) = (
            self.predict_heads(h, z_sample)
        )
        return (
            obs_reconstruction,
            reward_logits,
            continue_logits,
            h_z_joined,
            z_sample,
            prior_logits,
            posterior_logits,
        )

    def join_h_and_z(self, h, z):
        z_flat = z.view(z.size(0), -1)
        return torch.cat([h, z_flat], dim=-1)

    def init_state(self, batch_size, device="cpu"):
        """Initialize recurrent states to zero for a new batch sequence."""
        self.h_prev = torch.zeros(
            batch_size, self.d_hidden * self.n_blocks, device=device
        )
        self.z_prev = torch.zeros(
            batch_size, self.n_latents, self.n_classes, device=device
        )


class BlockLinear(nn.Module):
    """Independent linear projections over fixed feature groups."""

    def __init__(self, groups: int, d_in: int, d_out: int):
        super().__init__()
        self.groups = int(groups)
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        layers = [nn.Linear(self.d_in, self.d_out) for _ in range(self.groups)]
        self.weight = nn.Parameter(torch.stack([layer.weight for layer in layers]))
        self.bias = nn.Parameter(torch.stack([layer.bias for layer in layers]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-2:] != (self.groups, self.d_in):
            raise ValueError(
                "BlockLinear expected trailing shape "
                f"({self.groups}, {self.d_in}), got {tuple(inputs.shape[-2:])}"
            )
        return torch.einsum("...ki,koi->...ko", inputs, self.weight) + self.bias


class GatedRecurrentUnit(nn.Module):
    """
    The GRU is the paper's recurrent model for 'dreaming' ahead
    Takes:
        h_{t-1}: Previous hidden state
        z_{t-1}: Previous stochastic representation
        z_{t-1}: Previous *action*
    Returns:
        h_{t}: Current hidden state/16
    z: output of encoder "ObservationEncoder" class
    a: action sampled from policy: a_{t} ~ pi(a_{t} | s_{t})

    "The input to the GRU is a linear embedding of the *sampled latent z_t*,
    the action a_t, and the recurrent state"
    """

    def __init__(
        self,
        d_in,
        d_hidden,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.W_ir = nn.Linear(d_in, d_hidden, bias=True)
        self.W_hr = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W_iz = nn.Linear(d_in, d_hidden, bias=True)
        self.W_hz = nn.Linear(d_hidden, d_hidden, bias=True)
        self.W_in = nn.Linear(d_in, d_hidden, bias=True)
        self.W_hn = nn.Linear(d_hidden, d_hidden, bias=True)

    def forward(self, z, a, h_prev=None):
        batch_size = z.shape[0]
        x = torch.cat((z, a), dim=1)  # Join x and a in the hidden state axis
        if h_prev is None:
            h_prev = torch.zeros(
                batch_size, self.d_hidden, device=x.device, dtype=x.dtype
            )

        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h_prev))
        n = torch.tanh(r * (self.W_in(x) + self.W_hn(h_prev)))
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h_prev) - 1.0)
        h = z * n + (1 - z) * h_prev
        return h


class DynamicsPredictor(nn.Module):
    """
    Upscales GRU to num_latents * num_classes
    Breaks the hidden state into a distribution, and set of bins
    Takes logits over the bins (final dimension)
    """

    def __init__(self, d_in, d_hidden, num_latents, num_classes):
        super().__init__()
        d_out = num_latents * num_classes
        self.num_latents = num_latents
        self.num_classes = num_classes

        # Paper: "RMSNorm normalization, SiLU activation"
        self.layers = nn.Sequential(
            nn.Linear(d_in, d_hidden, bias=True),
            nn.RMSNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.RMSNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.shape[0], self.num_latents, self.num_classes)

        return out
