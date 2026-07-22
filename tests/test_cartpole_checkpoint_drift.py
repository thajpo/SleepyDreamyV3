from pathlib import Path

import pytest

from scripts.probe_cartpole_checkpoint_drift import (
    position_bin,
    summarize_fixed_history_rows,
)


def _row(
    *,
    episode: int,
    x: float,
    true_pref: int,
    q_pref: int,
    actor_action: int,
    q_delta: float,
    true_delta: float,
    policy_q_delta: float | None = None,
    policy_q_delta_se: float = 0.1,
) -> dict:
    row = {
        "episode": episode,
        "source_episode_return": 10.0 + episode,
        "x": x,
        "abs_x_bin": position_bin(x),
        "true_pref": true_pref,
        "q_pref": q_pref,
        "target_actor_action": actor_action,
        "q_delta": q_delta,
        "true_delta": true_delta,
    }
    for state_name in ("x", "x_dot", "theta", "theta_dot"):
        row[f"current_posterior_{state_name}_mse"] = 1.0
        row[f"one_step_prior_{state_name}_mse"] = 2.0
        row[f"one_step_posterior_{state_name}_mse"] = 3.0
    if policy_q_delta is not None:
        row["policy_q_delta"] = policy_q_delta
        row["policy_q_delta_se"] = policy_q_delta_se
        row["policy_q_pref"] = int(policy_q_delta > 0.0)
    return row


def test_position_bin_uses_absolute_position_and_stable_edges():
    assert position_bin(-0.49) == "0.0-0.5"
    assert position_bin(0.5) == "0.5-1.0"
    assert position_bin(-1.5) == "1.5-2.0"
    assert position_bin(2.0) == "2.0+"


def test_fixed_history_summary_preserves_balancing_and_position_strata():
    rows = [
        _row(
            episode=0,
            x=0.1,
            true_pref=0,
            q_pref=0,
            actor_action=0,
            q_delta=-2.0,
            true_delta=-2.0,
        ),
        _row(
            episode=0,
            x=-0.7,
            true_pref=0,
            q_pref=1,
            actor_action=0,
            q_delta=1.0,
            true_delta=-1.0,
        ),
        _row(
            episode=1,
            x=1.2,
            true_pref=1,
            q_pref=1,
            actor_action=0,
            q_delta=2.0,
            true_delta=1.0,
        ),
        _row(
            episode=1,
            x=-2.1,
            true_pref=1,
            q_pref=1,
            actor_action=1,
            q_delta=3.0,
            true_delta=3.0,
        ),
    ]

    summary = summarize_fixed_history_rows(
        rows,
        source_checkpoint=Path("source.pt"),
        target_checkpoint=Path("target.pt"),
        train_step=2500,
    )

    assert summary["episodes"] == 2
    assert summary["mean_source_episode_return"] == 10.5
    assert summary["q_vs_real_balanced_accuracy"] == 0.75
    assert summary["actor_vs_real_balanced_accuracy"] == 0.75
    assert summary["actor_vs_q_accuracy"] == 0.5
    assert summary["mean_current_posterior_x_mse"] == 1.0
    assert summary["mean_one_step_prior_x_mse"] == 2.0
    assert summary["mean_one_step_posterior_x_mse"] == 3.0
    assert summary["by_abs_x_bin"]["2.0+"]["states"] == 1


def test_fixed_history_summary_rejects_empty_input():
    with pytest.raises(ValueError, match="no states"):
        summarize_fixed_history_rows(
            [],
            source_checkpoint=Path("source.pt"),
            target_checkpoint=Path("target.pt"),
            train_step=None,
        )


def test_fixed_history_summary_reports_confident_policy_target_boundary():
    rows = [
        _row(
            episode=0,
            x=0.1,
            true_pref=0,
            q_pref=0,
            actor_action=0,
            q_delta=-1.0,
            true_delta=-1.0,
            policy_q_delta=-1.0,
        ),
        _row(
            episode=0,
            x=0.2,
            true_pref=1,
            q_pref=1,
            actor_action=0,
            q_delta=1.0,
            true_delta=1.0,
            policy_q_delta=1.0,
        ),
        _row(
            episode=0,
            x=0.3,
            true_pref=1,
            q_pref=1,
            actor_action=1,
            q_delta=1.0,
            true_delta=1.0,
            policy_q_delta=0.1,
        ),
    ]

    summary = summarize_fixed_history_rows(
        rows,
        source_checkpoint=Path("source.pt"),
        target_checkpoint=Path("target.pt"),
        train_step=1000,
    )

    assert summary["policy_q_confident_states"] == 2
    assert summary["policy_q_confident_actionable_states"] == 2
    assert summary["actor_vs_policy_q_confident_accuracy"] == 0.5
    assert summary["policy_q_vs_real_confident_balanced_accuracy"] == 1.0
    assert summary["policy_q_pref_hist"] == {"0": 1, "1": 2}
