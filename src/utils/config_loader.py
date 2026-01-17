"""Unified configuration loading for DreamerV3."""

import yaml

from ..config import config


def load_env_config(config_path, config_obj=None):
    """
    Load YAML config and apply overrides to global config.

    This is the consolidated config loader used by all entry points
    (main.py, evaluate.py, dream_visualizer.py).

    Args:
        config_path: Path to YAML config file
        config_obj: Optional config object to modify (uses global config if None)

    Returns:
        The modified config object
    """
    if config_obj is None:
        config_obj = config

    with open(config_path, "r") as f:
        overrides = yaml.safe_load(f)

    # Apply nested overrides for each section
    if "general" in overrides:
        for key, value in overrides["general"].items():
            if hasattr(config_obj.general, key):
                setattr(config_obj.general, key, value)

    if "environment" in overrides:
        for key, value in overrides["environment"].items():
            if hasattr(config_obj.environment, key):
                setattr(config_obj.environment, key, value)

    if "models" in overrides:
        for key, value in overrides["models"].items():
            if hasattr(config_obj.models, key):
                setattr(config_obj.models, key, value)

    if "train" in overrides:
        for key, value in overrides["train"].items():
            if hasattr(config_obj.train, key):
                setattr(config_obj.train, key, value)

    return config_obj


def print_config_summary(config_obj=None):
    """
    Print a summary of the current configuration.

    Args:
        config_obj: Config object to summarize (uses global config if None)
    """
    if config_obj is None:
        config_obj = config

    print(f"Environment: {config_obj.environment.environment_name}")
    print(
        f"  Actions: {config_obj.environment.n_actions}, "
        f"Observations: {config_obj.environment.n_observations}"
    )
    print(f"  d_hidden: {config_obj.models.d_hidden}")
    print(f"  use_pixels: {config_obj.general.use_pixels}")
