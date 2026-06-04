from dreamer.runtime.env import create_env


def test_easy_bandit_env_rewards_action_one():
    env = create_env("DreamerEasyBandit-v0", use_pixels=False)
    obs, _info = env.reset()

    assert obs.shape == (1,)
    assert env.action_space.n == 2

    _obs, reward_0, terminated, truncated, _info = env.step(0)
    assert reward_0 == 0.0
    assert not terminated
    assert not truncated

    _obs, reward_1, _terminated, _truncated, _info = env.step(1)
    assert reward_1 == 1.0


def test_easy_survival_env_action_one_survives():
    env = create_env("DreamerEasySurvival-v0", use_pixels=False)
    env.reset()

    _obs, reward, terminated, truncated, _info = env.step(1)
    assert reward == 1.0
    assert not terminated
    assert not truncated

    env.reset()
    _obs, reward, terminated, truncated, _info = env.step(0)
    assert reward == 1.0
    assert terminated
    assert not truncated
