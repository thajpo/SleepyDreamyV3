"""Mathematical utility functions for DreamerV3 training."""

import torch
import torch.nn.functional as F


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
    """Symmetric logarithm transform: sign(x) * log(|x| + 1)."""
    return torch.sign(x) * (torch.log(torch.abs(x) + 1))


def symexp(x):
    """Symmetric exponential transform (inverse of symlog): sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def unimix_logits(logits, unimix_ratio=0.01):
    """
    Apply unimix to categorical logits (DreamerV3 Section 4).

    Mixes the categorical distribution with a uniform distribution to prevent
    deterministic collapse and maintain gradient flow.

    Formula: probs_mixed = (1 - unimix_ratio) * softmax(logits) + unimix_ratio * uniform

    Args:
        logits: Categorical logits of any shape (..., num_classes)
        unimix_ratio: Fraction of uniform distribution to mix in (default: 0.01 = 1%)

    Returns:
        Mixed logits (converted back from mixed probabilities)
    """
    probs = F.softmax(logits, dim=-1)
    num_classes = logits.shape[-1]
    uniform = torch.ones_like(probs) / num_classes
    probs_mixed = (1 - unimix_ratio) * probs + unimix_ratio * uniform
    # Convert back to logits (add small epsilon for numerical stability)
    return torch.log(probs_mixed + 1e-8)


def twohot_encode(x, B):
    """
    Two-hot encoding for continuous values into discrete bins.

    The network is trained on twohot encoded targets, a generalization of onehot encoding to
    continuous values. The twohot encoding of a scalar is a vector with |B| entries that are
    all 0 except at the indices k and k + 1 of the two bins closest to the encoded scalar.
    The two entries sum up to 1, with linearly higher weight given to the bin that is closer
    to the encoded continuous number.

    Args:
        x: 1D tensor of values to encode
        B: 1D tensor of bin edges

    Returns:
        Tensor of shape (batch_size, n_bins) with twohot encoding
    """
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
