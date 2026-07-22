import pytest

from scripts.probe_cartpole_recovery_value_boundary import summarize_boundary_rows


def _row(
    true_pref: int,
    actor: int,
    posterior: int,
    prior: int,
    policy: int,
    *,
    true_delta: float,
    policy_se: float = 0.1,
) -> dict:
    def delta(pref: int) -> float:
        return 1.0 if pref == 1 else -1.0

    return {
        "real_policy_pref": true_pref,
        "real_policy_delta": true_delta,
        "actor_action": actor,
        "actor_probability_0": 0.75 if actor == 0 else 0.25,
        "actor_probability_1": 0.75 if actor == 1 else 0.25,
        "posterior_critic_pref": posterior,
        "posterior_critic_delta": delta(posterior),
        "prior_critic_pref": prior,
        "prior_critic_delta": delta(prior),
        "policy_q_pref": policy,
        "policy_q_delta": delta(policy),
        "policy_q_delta_se": policy_se,
    }


def test_boundary_summary_identifies_the_first_degraded_boundary():
    rows = [
        _row(0, 1, 0, 0, 1, true_delta=-2.0),
        _row(0, 0, 0, 1, 1, true_delta=-1.0),
        _row(1, 1, 1, 0, 0, true_delta=1.0),
        _row(1, 0, 1, 1, 0, true_delta=2.0),
    ]

    summary = summarize_boundary_rows(rows)

    assert summary["actionable_states"] == 4
    assert summary["posterior_critic_vs_real_policy_balanced_accuracy"] == 1.0
    assert summary["posterior_critic_vs_real_policy_accuracy"] == 1.0
    assert summary["prior_critic_vs_real_policy_balanced_accuracy"] == 0.5
    assert summary["policy_q_vs_real_policy_balanced_accuracy"] == 0.0
    assert summary["actor_vs_real_policy_balanced_accuracy"] == 0.5
    assert summary["actor_vs_policy_q_accuracy"] == 0.5
    assert summary["policy_q_confident_states"] == 4
    assert summary["actor_preferred_action_probability_mean"] == 0.5


def test_boundary_summary_excludes_tied_real_branches_from_accuracy():
    rows = [
        _row(-1, 0, 0, 0, 0, true_delta=0.0, policy_se=1.0),
        _row(1, 1, 1, 1, 1, true_delta=2.0, policy_se=1.0),
    ]

    summary = summarize_boundary_rows(rows)

    assert summary["states"] == 2
    assert summary["actionable_states"] == 1
    assert summary["actor_vs_real_policy_balanced_accuracy"] == 1.0
    assert summary["policy_q_confident_states"] == 0


def test_boundary_summary_rejects_empty_rows():
    with pytest.raises(ValueError, match="produced no states"):
        summarize_boundary_rows([])
