import pytest

from scripts.probe_cartpole_carry_horizon import summarize_context_rows


def test_carry_summary_reports_action_feature_and_value_drift():
    rows = [
        {
            "checkpoint": "step_10",
            "context_rows": 4,
            "time_bin": "16-31",
            "abs_x_bin": "0.5-1.0",
            "full_actor_action": 0,
            "local_actor_action": 1,
            "abs_actor_logit_delta_difference": 2.0,
            "feature_rms_distance": 3.0,
            "abs_critic_value_difference": 4.0,
            "full_decoder_mse": 5.0,
            "local_decoder_mse": 1.0,
            "full_decoder_x_mse": 6.0,
            "local_decoder_x_mse": 2.0,
        },
        {
            "checkpoint": "step_10",
            "context_rows": 4,
            "time_bin": "16-31",
            "abs_x_bin": "0.5-1.0",
            "full_actor_action": 1,
            "local_actor_action": 1,
            "abs_actor_logit_delta_difference": 0.0,
            "feature_rms_distance": 1.0,
            "abs_critic_value_difference": 2.0,
            "full_decoder_mse": 3.0,
            "local_decoder_mse": 1.0,
            "full_decoder_x_mse": 4.0,
            "local_decoder_x_mse": 2.0,
        },
    ]

    summary = summarize_context_rows(rows)
    context = summary["checkpoints"]["step_10"]["contexts"]["4"]

    assert context["states"] == 2
    assert context["actor_action_disagreement"] == 0.5
    assert context["mean_abs_actor_logit_delta_difference"] == 1.0
    assert context["mean_feature_rms_distance"] == 2.0
    assert context["mean_abs_critic_value_difference"] == 3.0
    assert context["mean_full_decoder_mse"] == 4.0
    assert context["mean_local_decoder_mse"] == 1.0
    assert context["by_time"]["16-31"]["states"] == 2
    assert context["by_abs_x"]["0.5-1.0"]["states"] == 2


def test_carry_summary_rejects_empty_rows():
    with pytest.raises(ValueError, match="no rows"):
        summarize_context_rows([])
