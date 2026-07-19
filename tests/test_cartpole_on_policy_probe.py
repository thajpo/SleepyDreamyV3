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
