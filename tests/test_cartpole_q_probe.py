import gymnasium as gym
import numpy as np
import pytest

from scripts.probe_cartpole_q import (
    action_preference,
    hybrid_state_score,
    one_step_outcome,
    rollout_score,
)


@pytest.mark.parametrize(
    ("state", "action"),
    [
        (np.array([0.0, 0.0, 0.03, 0.0], dtype=np.float32), 0),
        (np.array([0.0, 0.0, -0.03, 0.0], dtype=np.float32), 1),
        (np.array([2.39, 1.0, 0.0, 0.0], dtype=np.float32), 1),
    ],
)
def test_perfect_one_step_prediction_preserves_real_rollout_score(state, action):
    env = gym.make("CartPole-v1")
    horizon = 10
    try:
        real_score = rollout_score(env, state, action, horizon)
        real_next_state, _done = one_step_outcome(env, state, action)
        hybrid_score = hybrid_state_score(env, real_next_state, horizon)
    finally:
        env.close()

    assert hybrid_score == real_score


def test_perfect_one_step_predictions_preserve_action_preference():
    env = gym.make("CartPole-v1")
    state = np.array([0.0, 0.1, 0.04, -0.1], dtype=np.float32)
    horizon = 30
    try:
        real_scores = [rollout_score(env, state, action, horizon) for action in (0, 1)]
        hybrid_scores = []
        for action in (0, 1):
            next_state, _done = one_step_outcome(env, state, action)
            hybrid_scores.append(hybrid_state_score(env, next_state, horizon))
    finally:
        env.close()

    assert action_preference(hybrid_scores) == action_preference(real_scores)
