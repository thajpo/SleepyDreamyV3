"""Model initialization functions for DreamerV3."""


def initialize_actor(device, cfg):
    """
    Initializes the actor model.

    Args:
        device: Torch device to place the model on
        cfg: Configuration object with models and environment settings

    Returns:
        Actor network (ThreeLayerMLP)
    """
    # This import is here to avoid circular dependencies
    from .encoder import ThreeLayerMLP

    num_classes = cfg.models.d_hidden // 16
    d_in = (cfg.models.d_hidden * cfg.models.rnn.n_blocks) + (
        cfg.models.num_latents * num_classes
    )
    return ThreeLayerMLP(
        d_in=d_in,
        d_hidden=cfg.models.d_hidden,
        d_out=cfg.environment.n_actions,
    ).to(device)


def initialize_critic(device, cfg):
    """
    Initializes the critic model.

    Args:
        device: Torch device to place the model on
        cfg: Configuration object with models, environment, and train settings

    Returns:
        Critic network (ThreeLayerMLP)
    """
    import torch.nn as nn
    from .encoder import ThreeLayerMLP

    num_classes = cfg.models.d_hidden // 16
    d_in = (cfg.models.d_hidden * cfg.models.rnn.n_blocks) + (
        cfg.models.num_latents * num_classes
    )
    critic = ThreeLayerMLP(
        d_in=d_in,
        d_hidden=cfg.models.d_hidden,
        d_out=cfg.train.b_end - cfg.train.b_start,
    )
    # Zero-init critic output layer (DreamerV3 paper)
    nn.init.zeros_(critic.mlp[-1].weight)
    nn.init.zeros_(critic.mlp[-1].bias)
    return critic.to(device)


def initialize_world_model(device, cfg, batch_size=1):
    """
    Initializes the encoder and world model.

    Args:
        device: Torch device to place the models on
        cfg: Configuration object with models and environment settings
        batch_size: Batch size for initializing buffers

    Returns:
        Tuple of (encoder, world_model)
    """
    from .encoder import ObservationEncoder, StateOnlyEncoder
    from .world_model import RSSMWorldModel

    use_pixels = cfg.general.use_pixels

    if use_pixels:
        encoder = ObservationEncoder(
            mlp_config=cfg.models.encoder.mlp,
            cnn_config=cfg.models.encoder.cnn,
            d_hidden=cfg.models.d_hidden,
            n_observations=cfg.environment.n_observations,
            num_latents=cfg.models.num_latents,
        ).to(device)
    else:
        encoder = StateOnlyEncoder(
            mlp_config=cfg.models.encoder.mlp,
            d_hidden=cfg.models.d_hidden,
            n_observations=cfg.environment.n_observations,
            num_latents=cfg.models.num_latents,
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
