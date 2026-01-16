import torch
import torch.nn.functional as F
from .config import config
from .encoder import ThreeLayerMLP, ObservationEncoder, StateOnlyEncoder
from .world_model import RSSMWorldModel


def resize_pixels_to_target(pixels, target_size):
    """
    Resize pixel tensors to target_size. Handles both single images and batches.

    Args:
        pixels: Tensor of shape (B, C, H, W) or (T, C, H, W) or (B, T, C, H, W)
        target_size: Tuple of (height, width) to resize to

    Returns:
        Resized tensor with same batch/time dimensions but H, W resized to target_size
    """
    return F.interpolate(pixels, size=target_size, mode="bilinear", align_corners=False)


def symlog(x):
    out = torch.sign(x) * (torch.log(torch.abs(x) + 1))
    return out


def symexp(x):
    out = torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    return out


# The network is trained on twohot encoded targets8,28, a generalization of onehot encoding to
# continuous values. The twohot encoding of a scalar is a vector with |B| entries that are all 0 except
# at the indices k and k + 1 of the two bins closest to the encoded scalar. The two entries sum up
# to 1, with linearly higher weight given to the bin that is closer to the encoded continuous number.
# The network is then trained to minimize the categorical cross entropy loss for classification with
# soft targets
def twohot_encode(x, B):
    # B is a 1D tensor of bin edges.
    # x is a 1D tensor of values to encode.

    # Clamp values to be within the range of B
    x = torch.clamp(x, B.min(), B.max())

    # Find the index of the bin to the right of each value
    # The result of searchsorted is the index of the first element in B that is >= x
    right_bin_indices = torch.searchsorted(B, x)

    # The left bin is the one before it. Clamp at 0 for safety.
    left_bin_indices = torch.clamp(right_bin_indices - 1, 0)

    # Handle cases where right_bin_indices might be out of bounds if x matches B.max()
    right_bin_indices = torch.clamp(right_bin_indices, 0, len(B) - 1)

    # Get the values of the left and right bin edges
    bin_left = B[left_bin_indices]
    bin_right = B[right_bin_indices]

    # Calculate weights
    # Avoid division by zero if a value falls exactly on a bin edge
    denom = bin_right - bin_left
    denom[denom == 0] = 1.0

    weight_right = (x - bin_left) / denom
    weight_left = 1.0 - weight_right

    # Create the two-hot encoded tensor
    batch_size = x.size(0)
    n_bins = B.size(0)
    weights = torch.zeros(batch_size, n_bins, device=x.device)

    # Use scatter to place the weights in the correct locations
    weights.scatter_(1, left_bin_indices.unsqueeze(1), weight_left.unsqueeze(1))
    # Use scatter_add_ for the right bin in case left and right indices are the same
    weights.scatter_add_(1, right_bin_indices.unsqueeze(1), weight_right.unsqueeze(1))

    return weights


def initialize_actor(device, cfg):
    """Initializes the actor model."""
    # This import is here to avoid circular dependencies
    from .encoder import ThreeLayerMLP

    d_in = (cfg.models.d_hidden * cfg.models.rnn.n_blocks) + (
        cfg.models.d_hidden
        * (cfg.models.d_hidden // cfg.models.encoder.mlp.latent_categories)
    )
    return ThreeLayerMLP(
        d_in=d_in,
        d_hidden=cfg.models.d_hidden,
        d_out=cfg.environment.n_actions,
    ).to(device)


def initialize_critic(device, cfg):
    """Initializes the critic model."""
    from .encoder import ThreeLayerMLP

    d_in = (cfg.models.d_hidden * cfg.models.rnn.n_blocks) + (
        cfg.models.d_hidden
        * (cfg.models.d_hidden // cfg.models.encoder.mlp.latent_categories)
    )
    return ThreeLayerMLP(
        d_in=d_in,
        d_hidden=cfg.models.d_hidden,
        d_out=cfg.train.b_end - cfg.train.b_start,
    ).to(device)


def initialize_world_model(device, cfg, batch_size=1):
    use_pixels = cfg.general.use_pixels

    if use_pixels:
        encoder = ObservationEncoder(
            mlp_config=cfg.models.encoder.mlp,
            cnn_config=cfg.models.encoder.cnn,
            d_hidden=cfg.models.d_hidden,
            n_observations=cfg.environment.n_observations,
        ).to(device)
    else:
        encoder = StateOnlyEncoder(
            mlp_config=cfg.models.encoder.mlp,
            d_hidden=cfg.models.d_hidden,
            n_observations=cfg.environment.n_observations,
        ).to(device)

    world_model = RSSMWorldModel(
        models_config=cfg.models,
        env_config=cfg.environment,
        batch_size=batch_size,
        b_start=cfg.train.b_start,
        b_end=cfg.train.b_end,
        use_pixels=use_pixels,
    ).to(device)
    return encoder, world_model
