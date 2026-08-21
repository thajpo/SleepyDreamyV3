from dreamer.trainer.core import evaluation_episode_seed


def test_evaluation_seed_is_stable_per_episode():
    assert evaluation_episode_seed(7, 0) == 1_000_007
    assert evaluation_episode_seed(7, 19) == 1_000_026
    assert evaluation_episode_seed(8, 0) != evaluation_episode_seed(7, 0)
