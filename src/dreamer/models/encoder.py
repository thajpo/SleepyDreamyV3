import torch.nn as nn
import torch
import torch.nn.functional as F


class ObservationEncoder(nn.Module):
    """
    Encodes observations (pixels and/or state vectors) into a token vector.

    This is the first stage of the RSSM posterior path. The encoder extracts
    features from raw observations. The posterior logits are produced separately
    by the world model, conditioned on both these tokens AND the deterministic
    state h_t, implementing q(z_t | h_t, x_t) as specified in the paper.
    """

    def __init__(
        self,
        mlp_config,
        cnn_config,
        d_hidden,
        n_observations,
        num_latents=32,
    ):
        super().__init__()
        self.use_state = n_observations > 0

        if self.use_state:
            self.MLP = ThreeLayerMLP(
                d_in=n_observations, d_hidden=d_hidden, d_out=d_hidden
            )
        self.CNN = ObservationCNNEncoder(
            target_size=cnn_config.target_size,
            in_channels=cnn_config.input_channels,
            kernel_size=cnn_config.kernel_size,
            stride=cnn_config.stride,
            padding=cnn_config.padding,
            d_hidden=d_hidden,
            hidden_dim_ratio=mlp_config.hidden_dim_ratio,
            num_layers=cnn_config.num_layers,
            final_feature_size=cnn_config.final_feature_size,
        )

        n_channels = int(d_hidden / mlp_config.hidden_dim_ratio)
        cnn_out_features = (
            n_channels * 2 ** (cnn_config.num_layers - 1)
        ) * cnn_config.final_feature_size**2

        if self.use_state:
            self.token_dim = cnn_out_features + d_hidden  # CNN + MLP
        else:
            self.token_dim = cnn_out_features  # CNN only

        # num_latents and num_classes are stored for reference but the
        # posterior logit layer now lives in the world model.
        num_classes = d_hidden // 16
        self.num_latents = num_latents
        self.num_classes = num_classes

    def forward(self, x):
        """
        Encode observations into token vectors.

        Args:
            x: Dict with "pixels" (B, C, H, W) and optionally "state" (B, n_obs),
               OR a raw state tensor for StateOnlyEncoder compat.

        Returns:
            tokens: (B, token_dim) feature vector
        """
        # x is passed as a dict of ['state', 'pixels']
        # Expect pixels in (B, C, H, W) format - conversion happens once when loading data
        image_obs = x["pixels"] / 255.0

        x1 = self.CNN(image_obs)
        x1 = x1.reshape(x1.size(0), -1)  # Flatten all features

        if self.use_state:
            vec_obs = x["state"]
            x2 = self.MLP(vec_obs)
            tokens = torch.cat((x1, x2), dim=1)  # Join outputs along feature dimension
        else:
            tokens = x1  # CNN only

        return tokens


class ObservationCNNEncoder(nn.Module):
    """
    Observations are compressed dynamically based on config.
    Uses a series of convolutions with doubling channel progression.
    """

    def __init__(
        self,
        target_size,
        in_channels,
        kernel_size,
        stride,
        padding,
        d_hidden,
        hidden_dim_ratio=16,
        num_layers=4,
        final_feature_size=4,
    ):
        super().__init__()
        # Coerce to tuple for PyTorch compatibility (OmegaConf returns ListConfig)
        self.target_size = tuple(target_size)
        self.num_layers = num_layers
        self.final_feature_size = final_feature_size

        # Calculate base channel count from hidden dim ratio
        base_channels = int(d_hidden / hidden_dim_ratio)

        # Build sequential layers with ReLU activation after each conv
        conv_layers = []

        for i in range(num_layers):
            if i == 0:
                # First layer: input_channels -> base_channels
                out_ch = base_channels
                conv_layers.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
            else:
                # Subsequent layers: base_channels*2^(i-1) -> base_channels*2^i
                out_ch = base_channels * (2**i)
                in_ch = base_channels * (2 ** (i - 1))
                conv_layers.append(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )

            # Add ReLU activation after each conv layer except the last one
            if i < num_layers - 1:
                conv_layers.append(nn.ReLU())

        # Wrap in Sequential for clean forward pass
        self.cnn_blocks = nn.Sequential(*conv_layers)

    def forward(self, x):
        # resize image to target shape
        x = F.interpolate(
            input=x,
            size=self.target_size,
            mode="bilinear",
        )

        # Apply ReLU + convolution sequentially
        x = self.cnn_blocks(x)
        return x


class ThreeLayerMLP(nn.Module):
    """
    Observations are encoded with 3-layer MLP
    Input state is a vector of size 8
    """

    def __init__(
        self,
        d_in,
        d_hidden,
        d_out,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden, bias=True),
            nn.SiLU(),
            nn.Linear(d_hidden, d_out, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)


class StateOnlyEncoder(nn.Module):
    """
    Encoder for state-vector-only observations (no pixels).
    Used for simple environments like CartPole where state is sufficient.

    Like ObservationEncoder, this now returns tokens (not posterior logits).
    The posterior logit layer lives in the world model.
    """

    def __init__(
        self,
        mlp_config,
        d_hidden,
        n_observations,
        num_latents=32,
    ):
        super().__init__()
        self.MLP = ThreeLayerMLP(d_in=n_observations, d_hidden=d_hidden, d_out=d_hidden)

        num_classes = d_hidden // 16
        self.num_latents = num_latents
        self.num_classes = num_classes
        self.token_dim = d_hidden

    def forward(self, x):
        # x is the state vector directly (not a dict)
        tokens = self.MLP(x)
        return tokens
