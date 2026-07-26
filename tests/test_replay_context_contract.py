import torch

from dreamer.config import Config
from dreamer.models import (
    initialize_actor,
    initialize_critic,
    initialize_world_model,
    symexp_twohot_bins,
)
from dreamer.runtime.replay_buffer import EnvData
from dreamer.trainer.forward import dreamer_step
from dreamer.trainer.logging import create_step_metrics


def test_burn_in_reconstructs_detached_carry_without_training_context_rows() -> None:
    torch.manual_seed(19)
    config = Config(
        batch_size=1,
        sequence_length=2,
        replay_burn_in=1,
        d_hidden=32,
        num_latents=4,
        rnn_n_blocks=1,
        n_observations=4,
        n_actions=2,
        use_pixels=False,
        num_dream_steps=2,
        critic_replay_scale=0.0,
    )
    encoder, world_model = initialize_world_model("cpu", config, batch_size=1)
    actor = initialize_actor("cpu", config)
    critic = initialize_critic("cpu", config)
    critic_ema = initialize_critic("cpu", config)
    critic_ema.load_state_dict(critic.state_dict())
    states = torch.randn(1, 2, 4)
    batch = EnvData(
        states=states,
        actions=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        rewards=torch.tensor([[0.0, 1.0]]),
        is_first=torch.tensor([[False, False]]),
        is_last=torch.tensor([[False, False]]),
        is_terminal=torch.tensor([[False, False]]),
        future_returns=None,
        continue_weights=torch.ones(1, 2),
        mask=torch.ones(1, 2),
    )
    all_tokens = encoder(states.reshape(2, 4)).reshape(1, 2, -1)
    all_tokens.retain_grad()
    world_model.init_state(1)

    result = dreamer_step(
        encoder=encoder,
        world_model=world_model,
        actor=actor,
        critic=critic,
        critic_ema=critic_ema,
        batch=batch,
        metrics=create_step_metrics(torch.device("cpu"), False),
        all_tokens=all_tokens,
        B=1,
        T=2,
        train_start_t=1,
        skip_actor=True,
        skip_critic=True,
        bins=symexp_twohot_bins(-20, 20, config.num_bins),
        return_scale=1.0,
        config=config,
        device=torch.device("cpu"),
        use_pixels=False,
        do_log_images=False,
    )
    result.total_wm_loss.backward()

    assert all_tokens.grad is not None
    torch.testing.assert_close(
        all_tokens.grad[:, 0], torch.zeros_like(all_tokens.grad[:, 0])
    )
    assert all_tokens.grad[:, 1].norm().item() > 0.0


def test_cached_carry_path_restores_context_and_emits_stable_row_update() -> None:
    torch.manual_seed(23)
    config = Config(
        batch_size=1,
        sequence_length=2,
        replay_burn_in=1,
        d_hidden=32,
        num_latents=4,
        rnn_n_blocks=1,
        n_observations=4,
        n_actions=2,
        use_pixels=False,
        num_dream_steps=2,
        critic_replay_scale=0.0,
    )
    encoder, world_model = initialize_world_model("cpu", config, batch_size=1)
    actor = initialize_actor("cpu", config)
    critic = initialize_critic("cpu", config)
    critic_ema = initialize_critic("cpu", config)
    critic_ema.load_state_dict(critic.state_dict())
    states = torch.randn(1, 2, 4)
    batch = EnvData(
        states=states,
        actions=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        rewards=torch.tensor([[0.0, 1.0]]),
        is_first=torch.tensor([[False, False]]),
        is_last=torch.tensor([[False, False]]),
        is_terminal=torch.tensor([[False, False]]),
        future_returns=None,
        continue_weights=torch.ones(1, 2),
        mask=torch.ones(1, 2),
        step_ids=torch.tensor([[[8, 13, 4], [8, 13, 5]]]),
        replay_carry_h=torch.randn(1, config.d_hidden * config.rnn_n_blocks),
        replay_carry_z=torch.randn(
            1, config.num_latents, config.d_hidden // 16
        ),
        replay_carry_available=torch.tensor([True]),
    )
    all_tokens = encoder(states.reshape(2, 4)).reshape(1, 2, -1)
    world_model.init_state(1)

    result = dreamer_step(
        encoder=encoder,
        world_model=world_model,
        actor=actor,
        critic=critic,
        critic_ema=critic_ema,
        batch=batch,
        metrics=create_step_metrics(torch.device("cpu"), False),
        all_tokens=all_tokens,
        B=1,
        T=2,
        train_start_t=1,
        skip_actor=True,
        skip_critic=True,
        bins=symexp_twohot_bins(-20, 20, config.num_bins),
        return_scale=1.0,
        config=config,
        device=torch.device("cpu"),
        use_pixels=False,
        do_log_images=False,
    )

    assert result.replay_carry_updates is not None
    assert len(result.replay_carry_updates) == 1
    emitted_ids, emitted_h, emitted_z = result.replay_carry_updates[0]
    assert emitted_ids.tolist() == [[8, 13, 5]]
    assert emitted_h.shape == (1, config.d_hidden * config.rnn_n_blocks)
    assert emitted_z.shape == (
        1,
        config.num_latents,
        config.d_hidden // 16,
    )
