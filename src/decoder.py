import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ThreeLayerMLP

class ObservationDecoder(nn.Module):
    """
    Reconstructs the image and state vector from the model's state.
    Outputs a distribution of mean values over the image and state vector.

    The output is the mean assuming both the pixels and vector values are from a gaussian distribution
    """

    def __init__(
        self,
        mlp_config,
        cnn_config,
        env_config,
        gru_config,
        d_hidden,
    ):
        super().__init__()
        n_gru_blocks = gru_config.n_blocks
        decoder_input_dim = (d_hidden * n_gru_blocks) + (
            d_hidden * (d_hidden // mlp_config.latent_categories)
        )

        self.MLP = ThreeLayerMLP(
            d_in=decoder_input_dim,
            d_hidden=d_hidden,
            d_out=env_config.n_observations,
        )
        self.CNN = ObservationCNNDecoder(
            d_in=decoder_input_dim,
            in_channels=cnn_config.input_channels,
            kernel_size=cnn_config.kernel_size,
            stride=cnn_config.stride,
            d_hidden=d_hidden,
            hidden_dim_ratio=mlp_config.hidden_dim_ratio,
            num_layers=cnn_config.num_layers,
            final_feature_size=cnn_config.final_feature_size,
        )

    def forward(self, decoder_in):
        """
        Decodes the model state (z, h) back into an observation (pixels, state vector).
        z: Sampled latent state, shape (batch, d_hidden, d_hidden/16)
        h: GRU hidden state, shape (batch, d_hidden * n_gru_blocks)
        """
        # Flatten z and concatenate with h to form the decoder input state
        # Reconstruct pixels and state vector
        pixels_rec = self.CNN(decoder_in)
        state_rec = self.MLP(decoder_in)
        return {"pixels": pixels_rec, "state": state_rec}


class ObservationCNNDecoder(nn.Module):
    """
    Reconstructs the image from the model's state. This is the inverse
    of the ObservationCNNEncoder. It uses transposed convolutions to upsample.
    """

    def __init__(
        self,
        d_in,
        in_channels,
        kernel_size,
        stride,
        d_hidden,
        hidden_dim_ratio,
        num_layers,
        final_feature_size,
    ):
        super().__init__()
        base_channels = int(d_hidden / hidden_dim_ratio)

        # This is the number of channels at the input of the decoder CNN,
        # which is the output of the encoder CNN.
        final_encoder_channels = base_channels * (2 ** (num_layers - 1))
        self.first_layer_shape = (
            final_encoder_channels,
            final_feature_size,
            final_feature_size,
        )

        # Linear layer to project model state to the shape required by the first ConvTranspose2d
        self.fc = nn.Linear(d_in, int(torch.prod(torch.tensor(self.first_layer_shape))))

        # Build the transposed convolutional layers in reverse order of the encoder
        deconv_layers = []
        for i in range(num_layers - 1, -1, -1):
            in_ch = base_channels * (2**i)
            if i > 0:
                out_ch = base_channels * (2 ** (i - 1))
            else:
                out_ch = in_channels  # Final layer outputs logits for each channel

            deconv_layers.append(nn.ReLU())
            deconv_layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )

        self.deconv_blocks = nn.Sequential(*deconv_layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, *self.first_layer_shape)
        x = self.deconv_blocks(x)  # (N, C , H, W)

        return x


class ObservationMLPDecoder(nn.Module):
    """Reconstructs the state vector from the model's state."""

    def __init__(
        self,
        d_in,
        d_hidden,
        d_out,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
        self.d_out = d_out

    def forward(self, x):
        x = self.mlp(x)
        return x


class StateOnlyDecoder(nn.Module):
    """
    Decoder for state-vector-only reconstruction (no pixels).
    Used for simple environments like CartPole where state is sufficient.
    """

    def __init__(
        self,
        d_in,
        d_hidden,
        n_observations,
    ):
        super().__init__()
        self.MLP = ThreeLayerMLP(
            d_in=d_in, d_hidden=d_hidden, d_out=n_observations
        )

    def forward(self, decoder_in):
        return {"state": self.MLP(decoder_in)}
