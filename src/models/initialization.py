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
    """
    Initializes the critic model.

    Args:
        device: Torch device to place the model on
        cfg: Configuration object with models, environment, and train settings

    Returns:
        Critic network (ThreeLayerMLP)
    """
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
