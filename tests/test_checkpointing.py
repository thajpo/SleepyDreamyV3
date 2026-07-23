from dataclasses import asdict
from pathlib import Path

import torch
import pytest

from dreamer.config import Config
from dreamer.trainer.checkpoints import load_checkpoint, save_checkpoint
from dreamer.trainer.core import EvaluationResult


def _components():
    encoder = torch.nn.Linear(2, 2)
    world_model = torch.nn.Linear(2, 2)
    actor = torch.nn.Linear(2, 2)
    critic = torch.nn.Linear(2, 2)
    critic_ema = torch.nn.Linear(2, 2)
    q_critic = torch.nn.Linear(2, 2)
    q_critic_ema = torch.nn.Linear(2, 2)
    wm_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(world_model.parameters()), lr=1e-3
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(
        list(critic.parameters()) + list(q_critic.parameters()), lr=1e-3
    )
    return {
        "encoder": encoder,
        "world_model": world_model,
        "actor": actor,
        "critic": critic,
        "critic_ema": critic_ema,
        "q_critic": q_critic,
        "q_critic_ema": q_critic_ema,
        "wm_optimizer": wm_optimizer,
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
    }


def _save(checkpoint_dir: Path, parts: dict, **kwargs) -> str:
    return save_checkpoint(
        checkpoint_dir,
        12,
        parts["encoder"],
        parts["world_model"],
        parts["actor"],
        parts["critic"],
        parts["critic_ema"],
        parts["q_critic"],
        parts["q_critic_ema"],
        parts["wm_optimizer"],
        parts["actor_optimizer"],
        parts["critic_optimizer"],
        1.5,
        -2.0,
        7.0,
        "mlflow-test",
        run_id="manifest-test",
        best_eval_metric="episode_reward",
        **kwargs,
    )


def _load(path: str, parts: dict):
    return load_checkpoint(
        path,
        "cpu",
        parts["encoder"],
        parts["world_model"],
        parts["actor"],
        parts["critic"],
        parts["critic_ema"],
        parts["q_critic"],
        parts["q_critic_ema"],
        parts["wm_optimizer"],
        parts["actor_optimizer"],
        parts["critic_optimizer"],
        0.0,
        None,
        None,
    )


def test_checkpoint_round_trip_restores_models_and_run_state(tmp_path):
    original = _components()
    with torch.no_grad():
        original["actor"].weight.fill_(3.25)

    path = _save(
        tmp_path,
        original,
        label="best",
        best_eval_score=475.5,
        best_eval_step=12,
    )
    assert Path(path).name == "checkpoint_best.pt"
    assert not list(tmp_path.glob("*.tmp"))

    restored = _components()
    state = _load(path, restored)

    assert torch.allclose(
        restored["actor"].weight, torch.full_like(restored["actor"].weight, 3.25)
    )
    assert state == {
        "step": 12,
        "return_scale": 1.5,
        "ret_lo": -2.0,
        "ret_hi": 7.0,
        "best_eval_score": 475.5,
        "best_eval_step": 12,
        "best_eval_metric": "episode_reward",
        "run_id": "manifest-test",
        "continuation_terminal_ema": 0.5,
    }


def test_checkpoint_round_trip_restores_continuation_prevalence(tmp_path):
    path = _save(
        tmp_path,
        _components(),
        continuation_terminal_ema=0.0125,
    )

    state = _load(path, _components())

    assert state["continuation_terminal_ema"] == pytest.approx(0.0125)


def test_checkpoint_carries_portable_config_snapshot(tmp_path):
    snapshot = asdict(Config())
    path = _save(tmp_path, _components(), config_snapshot=snapshot)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["config_snapshot"] == snapshot


def test_final_checkpoint_does_not_overwrite_best_checkpoint(tmp_path):
    parts = _components()
    with torch.no_grad():
        parts["actor"].weight.fill_(5.0)
    best_path = _save(
        tmp_path,
        parts,
        label="best",
        best_eval_score=500.0,
        best_eval_step=12,
    )

    with torch.no_grad():
        parts["actor"].weight.fill_(-1.0)
    final_path = _save(
        tmp_path,
        parts,
        final=True,
        best_eval_score=500.0,
        best_eval_step=12,
    )

    best_parts = _components()
    final_parts = _components()
    _load(best_path, best_parts)
    _load(final_path, final_parts)

    assert torch.allclose(
        best_parts["actor"].weight,
        torch.full_like(best_parts["actor"].weight, 5.0),
    )
    assert torch.allclose(
        final_parts["actor"].weight,
        torch.full_like(final_parts["actor"].weight, -1.0),
    )


def test_evaluation_result_uses_an_explicit_checkpoint_metric():
    result = EvaluationResult(avg_length=500.0, avg_reward=21.0, win_rate=0.75)

    assert result.metric_value("episode_reward") == 21.0
    assert result.metric_value("episode_length") == 500.0
    assert result.metric_value("win_rate") == 0.75
    with pytest.raises(ValueError, match="Unsupported eval_metric"):
        result.metric_value("loss")
