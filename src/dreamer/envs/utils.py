"""Unified environment creation for DreamerV3."""

import gymnasium as gym
from gymnasium.wrappers import AddRenderObservation

# Register ALE (Atari) environments if available
try:
    import ale_py

    gym.register_envs(ale_py)
except ImportError:
    pass


def create_env(env_name, render_mode="rgb_array", use_pixels=True):
    """
    Create gymnasium environment with optional vision wrapper.

    This is the consolidated environment creation function used by all entry points
    (environment.py collector, evaluate.py, dream_visualizer.py).

    Args:
        env_name: Name of the gymnasium environment (e.g., "CartPole-v1")
        render_mode: Render mode for the environment (default: "rgb_array")
        use_pixels: If True, wrap with AddRenderObservation for pixel observations

    Returns:
        Configured gymnasium environment
    """
    if use_pixels:
        base_env = gym.make(env_name, render_mode=render_mode)
        env = AddRenderObservation(
            base_env, render_only=False, render_key="pixels", obs_key="state"
        )
    else:
        # State-only mode: no rendering overhead
        env = gym.make(env_name)
    return env
