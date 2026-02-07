"""Model initialization functions for DreamerV3."""


def initialize_actor(device, cfg):
    """
    Initializes the actor model.

    Args:
        device: Torch device to place the model on
        cfg: Configuration object (flat Config dataclass)

    Returns:
        Actor network (ThreeLayerMLP)
    """
    # This import is here to avoid circular dependencies
    from .encoder import ThreeLayerMLP
    from types import SimpleNamespace

    num_classes = cfg.d_hidden // 16
    d_in = (cfg.d_hidden * cfg.rnn_n_blocks) + (
        cfg.num_latents * num_classes
    )
    return ThreeLayerMLP(
        d_in=d_in,
        d_hidden=cfg.d_hidden,
        d_out=cfg.n_actions,
    ).to(device)


def initialize_critic(device, cfg):
    """
    Initializes the critic model.

    Args:
        device: Torch device to place the model on
        cfg: Configuration object (flat Config dataclass)

    Returns:
        Critic network (ThreeLayerMLP)
    """
    import torch.nn as nn
    from .encoder import ThreeLayerMLP
    from types import SimpleNamespace

    num_classes = cfg.d_hidden // 16
    d_in = (cfg.d_hidden * cfg.rnn_n_blocks) + (
        cfg.num_latents * num_classes
    )
    critic = ThreeLayerMLP(
        d_in=d_in,
        d_hidden=cfg.d_hidden,
        d_out=cfg.b_end - cfg.b_start,
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
        cfg: Configuration object (flat Config dataclass)
        batch_size: Batch size for initializing buffers

    Returns:
        Tuple of (encoder, world_model)
    """
    from .encoder import ObservationEncoder, StateOnlyEncoder
    from .world_model import RSSMWorldModel
    from types import SimpleNamespace

    use_pixels = cfg.use_pixels

    # Build nested config as SimpleNamespace objects for attribute access
    mlp_config = SimpleNamespace(
        hidden_dim_ratio=cfg.encoder_mlp_hidden_dim_ratio,
        n_layers=cfg.encoder_mlp_n_layers,
    )
    cnn_config = SimpleNamespace(
        stride=cfg.encoder_cnn_stride,
        kernel_size=cfg.encoder_cnn_kernel_size,
        padding=cfg.encoder_cnn_padding,
        input_channels=cfg.encoder_cnn_input_channels,
        num_layers=cfg.encoder_cnn_num_layers,
        final_feature_size=cfg.encoder_cnn_final_feature_size,
        target_size=cfg.encoder_cnn_target_size,
    )
    encoder_config = SimpleNamespace(mlp=mlp_config, cnn=cnn_config)
    models_config = SimpleNamespace(
        d_hidden=cfg.d_hidden,
        num_latents=cfg.num_latents,
        encoder=encoder_config,
        rnn=SimpleNamespace(n_blocks=cfg.rnn_n_blocks),
    )
    env_config = SimpleNamespace(
        n_actions=cfg.n_actions,
        n_observations=cfg.n_observations,
        environment_name=cfg.environment_name,
    )

    if use_pixels:
        encoder = ObservationEncoder(
            mlp_config=mlp_config,
            cnn_config=cnn_config,
            d_hidden=cfg.d_hidden,
            n_observations=cfg.n_observations,
            num_latents=cfg.num_latents,
        ).to(device)
    else:
        encoder = StateOnlyEncoder(
            mlp_config=mlp_config,
            d_hidden=cfg.d_hidden,
            n_observations=cfg.n_observations,
            num_latents=cfg.num_latents,
        ).to(device)

    world_model = RSSMWorldModel(
        models_config=models_config,
        env_config=env_config,
        batch_size=batch_size,
        b_start=cfg.b_start,
        b_end=cfg.b_end,
        use_pixels=use_pixels,
    ).to(device)
    return encoder, world_model
