import torch.nn as nn
import torch
import torch.nn.functional as F


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        mlp_config,
        cnn_config,
        d_hidden,
    ):
        super().__init__()
        self.MLP = ThreeLayerMLP(
            d_in=8, d_hidden=d_hidden, d_out=d_hidden
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

        self.latents = d_hidden
        # Use dynamic calculation based on actual config parameters

        n_channels = int(d_hidden / mlp_config.hidden_dim_ratio)
        cnn_out_features = (
            n_channels * 2 ** (cnn_config.num_layers - 1)
        ) * cnn_config.final_feature_size**2

        encoder_out = cnn_out_features + d_hidden  # CNN + MLP
        self.latent_categories = mlp_config.latent_categories

        # Paper
        logit_out = self.latents * (self.latents // self.latent_categories)
        self.logit_layer = nn.Linear(in_features=encoder_out, out_features=logit_out)

    def forward(self, x):
        # x is passed as a dict of ['state', 'pixels']
        # Expect pixels in (B, C, H, W) format - conversion happens once when loading data
        image_obs = x["pixels"] / 255.0
        vec_obs = x["state"]

        x1 = self.CNN(image_obs)
        x1 = x1.reshape(x1.size(0), -1)  # Flatten all features

        x2 = self.MLP(vec_obs)
        x = torch.cat((x1, x2), dim=1)  # Join outputs along feature dimension

        # feed this through a network to get out code * latent size
        x = self.logit_layer(x)
        x = x.view(x.shape[0], self.latents, self.latents // self.latent_categories)

        # Return logits directly. The Categorical distribution will handle the softmax.
        return x


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
        self.target_size = target_size
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
