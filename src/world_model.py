import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import ObservationDecoder, StateOnlyDecoder


class RSSMWorldModel(nn.Module):
    """
    World model architecture from Dreamerv3, called a 'Recurrent State Space Model'

    1. Encode scene (image, observation vector) -> Returns binned distribution of states 'z'
    2. Pass encoded states into GRU
    3. Estimate encoded distribution, hat{z}, a learned prior of the encoded representation
    """

    def __init__(
        self,
        models_config,
        env_config,
        batch_size,
        b_start,
        b_end,
        use_pixels=True,
    ):
        super().__init__()
        self.d_hidden = models_config.d_hidden

        # GatedRecurrentUnit | Uses 8 blocks to make a pseudo-large network
        self.blocks = nn.ModuleList()
        self.n_blocks = models_config.rnn.n_blocks
        gru_d_in = self.d_hidden + env_config.n_actions
        for _ in range(self.n_blocks):
            self.blocks.append(
                GatedRecurrentUnit(d_in=gru_d_in, d_hidden=self.d_hidden)
            )

        # Outputs prior distribution \hat{z} from the sequence model
        n_gru_blocks = models_config.rnn.n_blocks
        self.dynamics_predictor = DynamicsPredictor(
            d_in=self.d_hidden * n_gru_blocks, d_hidden=self.d_hidden
        )

        self.n_latents = models_config.d_hidden
        # Initalizing network params for t=0 ; h_0 is the zero matrix
        h_prev = torch.zeros(batch_size, self.d_hidden * n_gru_blocks)
        self.register_buffer("h_prev", h_prev)
        z_prev = torch.zeros((batch_size, self.d_hidden, int(self.d_hidden / 16)))
        self.register_buffer("z_prev", z_prev)

        self.h_prev_blocks = torch.split(self.h_prev, self.d_hidden, dim=-1)
        # Linear layer to project categorical sample to embedding dimension
        self.z_embedding = nn.Linear(
            self.d_hidden * (self.d_hidden // 16), self.d_hidden
        )

        # Takes 2D categorical samples and projects to d_hidden for GRU input
        h_z_dim = (self.d_hidden * n_gru_blocks) + (
            self.d_hidden
            * (self.d_hidden // models_config.encoder.mlp.latent_categories)
        )

        # Rewards use two-hot encoding
        reward_out = abs(b_start - b_end)
        self.reward_predictor = nn.Linear(h_z_dim, reward_out)
        nn.init.zeros_(self.reward_predictor.weight)  # Reward is initalized to zero
        self.continue_predictor = nn.Linear(h_z_dim, 1)

        # Decoder. Outputs distribution of mean predictions for pixel/vector observations
        if use_pixels:
            self.decoder = ObservationDecoder(
                mlp_config=models_config.encoder.mlp,
                cnn_config=models_config.encoder.cnn,
                env_config=env_config,
                gru_config=models_config.rnn,
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
        Steps the recurrent dynamics forward one step.

        Args:
            z_embed: The embedded latent state z_t.
            action: The one-hot encoded action a_t.
            h_prev: The previous hidden state h_{t-1}.

        Returns:
            h: The new hidden state h_t.
            prior_logits: The predicted logits for the next latent state, z_{t+1}.
        """
        outputs = []
        h_prev_blocks = torch.split(h_prev, self.d_hidden, dim=-1)
        for i, block in enumerate(self.blocks):
            h_i = block(z_embed, action, h_prev_blocks[i])
            outputs.append(h_i)
        h = torch.cat(outputs, dim=-1)
        # Detach and clone before storing to avoid in-place modification of computation graph
        # Gradients flow through h directly, not through stored h_prev
        # Use clone() to ensure we create a new tensor, not a view
        self.h_prev = h.detach().clone()
        prior_logits = self.dynamics_predictor(h)
        return h, prior_logits

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

    def forward(self, posterior_dist, action):
        """
        Performs a full world model step for training.
        This involves encoding, stepping dynamics, and making predictions.
        """
        # Apply straight-through method to sample z while keeping gradients
        z_indices = posterior_dist.sample()  # (batch_size, latents)
        z_onehot = F.one_hot(z_indices, num_classes=self.d_hidden // 16).float()
        z_sample = z_onehot + (posterior_dist.probs - posterior_dist.probs.detach())
        bsz = z_onehot.shape[0]
        z_flat = z_sample.view(bsz, -1)
        z_embed = self.z_embedding(z_flat)

        # Step the dynamics to get the new hidden state and prior
        h, prior_logits = self.step_dynamics(z_embed, action, self.h_prev)

        # Generate predictions using the new state
        (
            obs_reconstruction, reward_logits, continue_logits, h_z_joined
        ) = self.predict_heads(h, z_sample)
        return obs_reconstruction, reward_logits, continue_logits, h_z_joined, prior_logits

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
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h_prev))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h_prev))
        h = (1 - z) * n + z * h_prev
        return h


class DynamicsPredictor(nn.Module):
    """
    Upscales GRU to d_hidden ** 2 / 16
    Breaks the hidden state into a distribution, and set of bins
    Takes logits over the bins (final dimension)
    """

    def __init__(self, d_in, d_hidden):
        super().__init__()
        d_out = int(d_hidden**2 / 16)
        self.n_latents = d_hidden

        self.layers = nn.Sequential(
            nn.Linear(d_in, d_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.shape[0], self.n_latents, self.n_latents // 16)

        return out
