"""Unified environment creation for DreamerV3."""

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AddRenderObservation, AtariPreprocessing

# Register ALE (Atari) environments if available
try:
    import ale_py

    gym.register_envs(ale_py)
except ImportError:
    pass


class AtariPixelsStateWrapper(gym.ObservationWrapper):
    """Expose Atari observations as {'pixels', 'state'} for pixel pipelines."""

    def __init__(self, env):
        super().__init__(env)
        pixel_space = env.observation_space
        self.observation_space = gym.spaces.Dict(
            {
                "pixels": pixel_space,
                "state": gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, observation):
        return {
            "pixels": observation,
            "state": np.zeros((1,), dtype=np.float32),
        }


class FireResetWrapper(gym.Wrapper):
    """Take FIRE on reset when the environment supports it."""

    def __init__(self, env):
        super().__init__(env)
        get_action_meanings = getattr(env.unwrapped, "get_action_meanings", None)
        self._fire_action = None
        if callable(get_action_meanings):
            try:
                action_meanings = get_action_meanings()
                if isinstance(action_meanings, list) and all(
                    isinstance(x, str) for x in action_meanings
                ):
                    if "FIRE" in action_meanings:
                        self._fire_action = action_meanings.index("FIRE")
            except Exception:
                self._fire_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._fire_action is None:
            return obs, info
        obs, _reward, terminated, truncated, step_info = self.env.step(
            self._fire_action
        )
        if terminated or truncated:
            return self.env.reset(**kwargs)
        if isinstance(info, dict) and isinstance(step_info, dict):
            info = {**info, **step_info}
        return obs, info


def create_env(env_name, render_mode="rgb_array", use_pixels=True, config=None):
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
        atari_compat = bool(getattr(config, "atari_compat_mode", False))
        is_ale = env_name.startswith("ALE/")

        if atari_compat and is_ale:
            sticky_prob = float(getattr(config, "atari_sticky_action_prob", 0.25))
            full_action_space = bool(getattr(config, "atari_full_action_space", False))
            # Use frameskip=1 at ALE level, then apply frame skipping in AtariPreprocessing.
            base_env = gym.make(
                env_name,
                render_mode=render_mode,
                frameskip=1,
                repeat_action_probability=sticky_prob,
                full_action_space=full_action_space,
            )
            env = AtariPreprocessing(
                base_env,
                noop_max=int(getattr(config, "atari_noop_max", 30)),
                frame_skip=int(getattr(config, "atari_frame_skip", 4)),
                terminal_on_life_loss=bool(
                    getattr(config, "atari_terminal_on_life_loss", False)
                ),
                screen_size=84,
                grayscale_obs=False,
                scale_obs=False,
            )
            if bool(getattr(config, "atari_fire_reset", True)):
                env = FireResetWrapper(env)
            env = AtariPixelsStateWrapper(env)
        else:
            base_env = gym.make(env_name, render_mode=render_mode)
            env = AddRenderObservation(
                base_env, render_only=False, render_key="pixels", obs_key="state"
            )
    else:
        # State-only mode: no rendering overhead
        env = gym.make(env_name)
    return env
