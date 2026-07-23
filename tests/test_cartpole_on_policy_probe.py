from pathlib import Path

import pytest

from scripts.probe_cartpole_on_policy import summarize_on_policy_rows


def _row(
    episode: int,
    *,
    episode_return: float,
    true_delta: float,
    true_pref: int,
    actor_correct: bool,
    steps_to_end: int,
    regret: float = 0.0,
    chosen_done: bool = False,
    other_done: bool = False,
) -> dict:
    return {
        "episode": episode,
        "episode_return": episode_return,
        "true_delta": true_delta,
        "true_pref": true_pref,
        "actor_correct": int(actor_correct),
        "steps_to_end": steps_to_end,
        "trusted_regret": regret,
        "actor_confidence": 0.8,
        "actor_action": true_pref if actor_correct and true_pref >= 0 else 0,
        "chosen_action_terminated": int(chosen_done),
        "other_action_terminated": int(other_done),
    }


def test_on_policy_summary_separates_critical_and_terminal_errors():
    rows = [
        _row(
            0,
            episode_return=100,
            true_delta=20,
            true_pref=1,
            actor_correct=False,
            steps_to_end=3,
            regret=20,
            chosen_done=True,
        ),
        _row(
            0,
            episode_return=100,
            true_delta=2,
            true_pref=1,
            actor_correct=True,
            steps_to_end=2,
        ),
        _row(
            1,
            episode_return=500,
            true_delta=0,
            true_pref=-1,
            actor_correct=True,
            steps_to_end=1,
        ),
    ]

    summary = summarize_on_policy_rows(
        rows,
        checkpoint_path=Path("checkpoint.pt"),
        train_step=123,
        critical_margin=15,
        terminal_window=10,
    )

    assert summary["episodes"] == 2
    assert summary["mean_episode_return"] == 300
    assert summary["solved_episode_fraction"] == 0.5
    assert summary["actionable_states"] == 2
    assert summary["actor_vs_rollout_accuracy"] == 0.5
    assert summary["critical_states"] == 1
    assert summary["critical_accuracy"] == 0.0
    assert summary["terminal_window_accuracy"] == 0.5
    assert summary["avoidable_immediate_failures"] == 1
    assert summary["episodes_with_actionable_error"] == 1
    assert summary["episodes_with_terminal_window_error"] == 1
    assert summary["mean_actionable_errors_per_episode"] == 0.5
    assert summary["mean_regret_per_episode"] == 10


def test_on_policy_summary_rejects_empty_rows():
    with pytest.raises(ValueError, match="no states"):
        summarize_on_policy_rows(
            [],
            checkpoint_path=Path("checkpoint.pt"),
            train_step=None,
            critical_margin=15,
            terminal_window=10,
        )


def test_on_policy_summary_reports_confident_policy_q_agreement():
    correct = _row(
        0,
        episode_return=100,
        true_delta=10,
        true_pref=1,
        actor_correct=True,
        steps_to_end=10,
    )
    correct.update(
        {
            "policy_q_delta": 2.0,
            "policy_q_delta_se": 0.1,
            "policy_q_pref": 1,
        }
    )
    mismatch = _row(
        1,
        episode_return=50,
        true_delta=-10,
        true_pref=0,
        actor_correct=False,
        steps_to_end=5,
    )
    mismatch.update(
        {
            "actor_action": 1,
            "policy_q_delta": -2.0,
            "policy_q_delta_se": 0.1,
            "policy_q_pref": 0,
        }
    )

    summary = summarize_on_policy_rows(
        [correct, mismatch],
        checkpoint_path=Path("checkpoint.pt"),
        train_step=123,
        critical_margin=15,
        terminal_window=10,
    )

    assert summary["policy_q_confident_states"] == 2
    assert summary["policy_q_confident_actor_agreement"] == 0.5
    assert summary["policy_q_confident_actionable_states"] == 2
    assert summary["policy_q_confident_vs_rollout_balanced_accuracy"] == 1.0


def test_on_policy_summary_compares_decomposed_value_preferences():
    correct = _row(
        0,
        episode_return=100,
        true_delta=10,
        true_pref=1,
        actor_correct=True,
        steps_to_end=20,
    )
    correct.update(
        {
            "q_delta": -2.0,
            "q_pref": 0,
            "hybrid_state_delta": 4.0,
            "hybrid_state_pref": 1,
            "decomp_model_return_h1_delta": 0.0,
            "decomp_model_return_h1_pref": -1,
            "decomp_critic_bootstrap_h1_delta": -2.0,
            "decomp_critic_bootstrap_h1_pref": 0,
        }
    )
    wrong = _row(
        1,
        episode_return=50,
        true_delta=-8,
        true_pref=0,
        actor_correct=False,
        steps_to_end=5,
    )
    wrong.update(
        {
            "actor_action": 1,
            "q_delta": 3.0,
            "q_pref": 1,
            "hybrid_state_delta": -5.0,
            "hybrid_state_pref": 0,
            "decomp_model_return_h1_delta": 0.0,
            "decomp_model_return_h1_pref": -1,
            "decomp_critic_bootstrap_h1_delta": 3.0,
            "decomp_critic_bootstrap_h1_pref": 1,
        }
    )

    summary = summarize_on_policy_rows(
        [correct, wrong],
        checkpoint_path=Path("checkpoint.pt"),
        train_step=123,
        critical_margin=15,
        terminal_window=10,
    )

    assert summary["hybrid_state_vs_rollout_accuracy"] == 1.0
    assert summary["hybrid_state_vs_rollout_balanced_accuracy"] == 1.0
    assert summary["q_vs_rollout_accuracy"] == 0.0
    assert summary["q_vs_rollout_balanced_accuracy"] == 0.0
    assert summary["decomp_critic_bootstrap_h1_vs_rollout_accuracy"] == 0.0
    assert summary["decomp_model_return_h1_margined_actionable_states"] == 0
    assert summary["actor_vs_q_accuracy"] == 0.5
