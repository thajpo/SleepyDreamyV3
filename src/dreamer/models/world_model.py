import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import ObservationDecoder, StateOnlyDecoder


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
        use_pixels=True,
    ):
        super().__init__()
        self.d_hidden = models_config.d_hidden
        num_classes = self.d_hidden // 16

        # GatedRecurrentUnit | Uses n_blocks independent GRUs computed in parallel
        self.n_blocks = models_config.rnn.n_blocks
        gru_d_in = self.d_hidden + env_config.n_actions

        # Create temporary blocks to initialize weights, then stack for batched computation
        blocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            blocks.append(GatedRecurrentUnit(d_in=gru_d_in, d_hidden=self.d_hidden))

        # Stack input projection weights: (n_blocks, d_hidden, d_in)
        self._W_ir = nn.Parameter(torch.stack([b.W_ir.weight for b in blocks]))
        self._W_iz = nn.Parameter(torch.stack([b.W_iz.weight for b in blocks]))
        self._W_in = nn.Parameter(torch.stack([b.W_in.weight for b in blocks]))
        self._b_ir = nn.Parameter(torch.stack([b.W_ir.bias for b in blocks]))
        self._b_iz = nn.Parameter(torch.stack([b.W_iz.bias for b in blocks]))
        self._b_in = nn.Parameter(torch.stack([b.W_in.bias for b in blocks]))

        # Stack hidden projection weights: (n_blocks, d_hidden, d_hidden)
        self._W_hr = nn.Parameter(torch.stack([b.W_hr.weight for b in blocks]))
        self._W_hz = nn.Parameter(torch.stack([b.W_hz.weight for b in blocks]))
        self._W_hn = nn.Parameter(torch.stack([b.W_hn.weight for b in blocks]))
        self._b_hr = nn.Parameter(torch.stack([b.W_hr.bias for b in blocks]))
        self._b_hz = nn.Parameter(torch.stack([b.W_hz.bias for b in blocks]))
        self._b_hn = nn.Parameter(torch.stack([b.W_hn.bias for b in blocks]))

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
        h_dim = self.d_hidden * n_gru_blocks
        posterior_in_dim = h_dim + encoder_token_dim
        posterior_out_dim = self.n_latents * self.n_classes
        self.posterior_head = nn.Linear(posterior_in_dim, posterior_out_dim)

        # Initalizing network params for t=0 ; h_0 is the zero matrix
        h_prev = torch.zeros(batch_size, self.d_hidden * n_gru_blocks)
        self.register_buffer("h_prev", h_prev)
        z_prev = torch.zeros((batch_size, self.n_latents, self.n_classes))
        self.register_buffer("z_prev", z_prev)

        # Linear layer to project categorical sample to embedding dimension
        self.z_embedding = nn.Linear(self.n_latents * self.n_classes, self.d_hidden)

        # Takes 2D categorical samples and projects to d_hidden for GRU input
        h_z_dim = (self.d_hidden * n_gru_blocks) + (self.n_latents * self.n_classes)

        # Rewards use two-hot encoding
        reward_out = abs(b_start - b_end)
        self.reward_predictor = nn.Linear(h_z_dim, reward_out)
        nn.init.zeros_(self.reward_predictor.weight)  # Reward is initalized to zero
        self.continue_predictor = nn.Linear(h_z_dim, 1)

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
        Steps the recurrent dynamics forward one step using batched GRU computation.

        All n_blocks GRUs are computed in parallel via einsum over stacked weights.
        The input x = cat(z_embed, action) is identical for all blocks, while
        h_prev differs per block.

        Args:
            z_embed: The embedded latent state z_t. Shape: (B, d_hidden)
            action: The one-hot encoded action a_t. Shape: (B, n_actions)
            h_prev: The previous hidden state h_{t-1}. Shape: (B, n_blocks * d_hidden)

        Returns:
            h: The new hidden state h_t. Shape: (B, n_blocks * d_hidden)
            prior_logits: The predicted logits for the next latent state.
        """
        B = z_embed.shape[0]
        x = torch.cat((z_embed, action), dim=1)  # (B, d_in)
        h_blocks = h_prev.view(
            B, self.n_blocks, self.d_hidden
        )  # (B, n_blocks, d_hidden)

        # Input projections: (B, d_in) @ (n_blocks, d_hidden, d_in).T -> (B, n_blocks, d_hidden)
        ir = torch.einsum("bi,kji->bkj", x, self._W_ir) + self._b_ir
        iz = torch.einsum("bi,kji->bkj", x, self._W_iz) + self._b_iz
        in_ = torch.einsum("bi,kji->bkj", x, self._W_in) + self._b_in

        # Hidden projections: (B, n_blocks, d_hidden) @ (n_blocks, d_hidden, d_hidden).T -> (B, n_blocks, d_hidden)
        hr = torch.einsum("bki,kji->bkj", h_blocks, self._W_hr) + self._b_hr
        hz = torch.einsum("bki,kji->bkj", h_blocks, self._W_hz) + self._b_hz
        hn = torch.einsum("bki,kji->bkj", h_blocks, self._W_hn) + self._b_hn

        # GRU equations with repo-matching modifications:
        # 1. Reset gate: standard sigmoid
        r = torch.sigmoid(ir + hr)
        # 2. Candidate: reset gates the candidate directly (source convention)
        #    tanh(reset * raw_candidate) rather than tanh(input + reset * hidden)
        n = torch.tanh(r * (in_ + hn))
        # 3. Update gate: bias of -1 makes sigmoid(x-1) ≈ 0.27 at init,
        #    causing the GRU to default to RETAINING the old state.
        #    This is the GRU analogue of the LSTM forget gate bias trick.
        z = torch.sigmoid(iz + hz - 1.0)
        # 4. Blend: update * candidate + (1-update) * old
        #    With z≈0.27 at init: mostly keeps old state, small correction from candidate
        h_new = z * n + (1 - z) * h_blocks

        h = h_new.view(B, -1)  # (B, n_blocks * d_hidden)
        # Detach and clone before storing to avoid in-place modification of computation graph
        # Gradients flow through h directly, not through stored h_prev
        self.h_prev = h.detach().clone()
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

    def forward(self, tokens, action):
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

        Returns:
            obs_reconstruction, reward_logits, continue_logits, h_z_joined,
            z_sample, prior_logits, posterior_logits
        """
        # 1. Step GRU using PREVIOUS z and action
        z_prev_flat = self.z_prev.view(self.z_prev.size(0), -1)
        z_prev_embed = self.z_embedding(z_prev_flat)
        h, prior_logits = self.step_dynamics(z_prev_embed, action, self.h_prev)

        # 2. Compute posterior conditioned on h_t AND observation tokens
        posterior_logits = self.compute_posterior(h, tokens)

        # 3. Sample z_t from posterior using straight-through estimator
        posterior_probs = F.softmax(posterior_logits, dim=-1)
        posterior_dist = torch.distributions.Categorical(
            probs=posterior_probs, validate_args=False
        )
        z_indices = posterior_dist.sample()  # (batch_size, latents)
        z_onehot = F.one_hot(z_indices, num_classes=self.n_classes).float()
        z_sample = z_onehot + (posterior_probs - posterior_probs.detach())
        # Cache z_t for the next transition step.
        self.z_prev = z_sample.detach().clone()

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

        self.layers = nn.Sequential(
            nn.Linear(d_in, d_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.shape[0], self.num_latents, self.num_classes)

        return out
