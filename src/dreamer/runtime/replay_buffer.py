"""
Episode Replay Buffer for Dreamer training.

Stores complete episodes, samples fixed-length subsequences.
This matches DreamerV3's approach for consistent batch shapes.
"""

import threading
import random
import numpy as np
import torch
from collections import deque
from queue import Empty
from typing import NamedTuple, Optional

from ..models.math_utils import resize_pixels_to_target


class EnvData(NamedTuple):
    """Immutable batch of environment data sampled from replay."""

    states: torch.Tensor  # (B, T, n_obs) — raw env vectors
    actions: torch.Tensor  # (B, T, n_actions)
    rewards: torch.Tensor  # (B, T)
    is_last: torch.Tensor  # (B, T)
    is_terminal: torch.Tensor  # (B, T)
    future_returns: Optional[torch.Tensor]  # (B, T), when exact targets enabled
    mask: torch.Tensor  # (B, T) — 1=real, 0=padded
    pixels: Optional[torch.Tensor] = None  # (B, T, C, H, W)
    pixels_original: Optional[torch.Tensor] = None  # (B, T, C, H, W)




class EpisodeReplayBuffer:
    """
    Thread-safe replay buffer that stores complete episodes and samples
    fixed-length subsequences.

    Design:
    - Background thread continuously drains the mp.Queue (never blocks collectors)
    - Circular buffer stores up to max_episodes (FIFO eviction)
    - sample() returns fixed-length subsequences (pads short episodes)
    - Blocks only on startup until min_episodes collected
    """

    def __init__(
        self,
        data_queue=None,
        max_episodes=1000,
        min_episodes=64,
        sequence_length=25,
        gamma=0.997,
        compute_future_returns=False,
        throttle_collection=False,
    ):
        """
        Args:
            data_queue: mp.Queue to drain episodes from
            max_episodes: Maximum episodes to store (older episodes evicted)
            min_episodes: Block sampling until buffer has this many episodes
            sequence_length: Fixed length of sampled subsequences
            gamma: Discount used for full-episode future return targets
            compute_future_returns: Whether to annotate stored episodes with returns
            throttle_collection: Apply trainer-issued environment-step budgets after
                the startup population is ready
        """
        self.data_queue = data_queue
        self.max_episodes = max_episodes
        self.min_episodes = min_episodes
        self.sequence_length = sequence_length
        self.gamma = float(gamma)
        self.compute_future_returns = bool(compute_future_returns)
        self.throttle_collection = bool(throttle_collection)

        self.buffer = deque(maxlen=max_episodes)
        self.lock = threading.Lock()
        self._budget_changed = threading.Condition(self.lock)
        self.ready_event = threading.Event()  # Signals when min_episodes reached

        self._stop = False
        self._thread = None
        self._episodes_added = 0
        self._total_steps = 0  # For tracking average episode length
        self._recent_ep_lengths = deque(maxlen=100)  # Track recent episode lengths
        self._max_sequence_length = 256  # Cap for adaptive growth
        self._env_step_budget: float | None = None

    def start(self):
        """Start background queue draining thread."""
        if self.data_queue is None:
            return
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background thread gracefully."""
        with self._budget_changed:
            self._stop = True
            self._budget_changed.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _drain_loop(self):
        """Background thread: continuously move episodes from queue to buffer."""
        data_queue = self.data_queue
        if data_queue is None:
            return
        while not self._stop:
            try:
                # Short timeout so we can check stop flag regularly
                episode = data_queue.get(timeout=0.1)
                if not self._wait_for_collection_budget(episode):
                    break
                self.add_episode(episode)
            except Empty:
                continue

    def _wait_for_collection_budget(self, episode) -> bool:
        """Wait until training has budgeted enough steps for one whole episode."""
        if not self.throttle_collection:
            return True

        episode_steps = float(episode[6] if len(episode) > 6 else len(episode[1]))
        with self._budget_changed:
            while (
                not self._stop
                and self._env_step_budget is not None
                and self._total_steps + episode_steps > self._env_step_budget
            ):
                self._budget_changed.wait(timeout=0.1)
            return not self._stop

    def allow_env_steps(self, steps: float) -> None:
        """Increase the post-startup collection budget after a trainer update."""
        if not self.throttle_collection:
            return
        if steps < 0:
            raise ValueError("collection budget increment must be non-negative")
        with self._budget_changed:
            if self._env_step_budget is None:
                return
            self._env_step_budget += float(steps)
            self._budget_changed.notify_all()

    def add_episode(self, episode):
        """Insert one complete episode into the replay buffer."""
        pixels, states, actions, rewards, is_last, is_terminal = episode[:6]
        ep_len = episode[6] if len(episode) > 6 else len(states)
        future_returns = None
        if self.compute_future_returns:
            future_returns = np.zeros(len(rewards), dtype=np.float32)
            for index in range(len(rewards) - 2, -1, -1):
                next_index = index + 1
                future_returns[index] = rewards[next_index] + self.gamma * (
                    1.0 - float(is_last[next_index])
                ) * future_returns[next_index]
        stored_episode = (
            pixels,
            states,
            actions,
            rewards,
            is_last,
            is_terminal,
            ep_len,
            future_returns,
        )
        with self.lock:
            self.buffer.append(stored_episode)
            self._episodes_added += 1
            self._total_steps += ep_len
            self._recent_ep_lengths.append(ep_len)
            if len(self.buffer) >= self.min_episodes and not self.ready_event.is_set():
                if self.throttle_collection:
                    # Startup is intentionally unrestricted. Once the minimum
                    # population exists, every later episode must be paid for by
                    # completed trainer updates.
                    self._env_step_budget = float(self._total_steps)
                self.ready_event.set()
                print(
                    f"Replay buffer ready: {len(self.buffer)} episodes collected",
                    flush=True,
                )

    def _sample_subsequence(self, episode):
        """
        Sample a fixed-length subsequence from an episode.

        If episode is shorter than sequence_length, pads with zeros and
        marks as terminated. Returns mask indicating real (1) vs padded (0) steps.
        """
        # Replay adds future returns to the seven-field collector episode tuple.
        pixels, states, actions, rewards, is_last, is_terminal = episode[:6]
        future_returns = episode[7]
        # Use states for length - pixels may be None in state-only mode
        ep_len = len(states)
        seq_len = self.sequence_length

        if ep_len >= seq_len:
            # Sample random start point
            start = random.randint(0, ep_len - seq_len)
            mask = np.ones(seq_len, dtype=np.float32)  # All real steps
            return (
                pixels[start : start + seq_len] if pixels is not None else None,
                states[start : start + seq_len],
                actions[start : start + seq_len],
                rewards[start : start + seq_len],
                is_last[start : start + seq_len],
                is_terminal[start : start + seq_len],
                (
                    future_returns[start : start + seq_len]
                    if future_returns is not None
                    else None
                ),
                mask,
            )
        else:
            # Pad short episode
            pad_len = seq_len - ep_len

            # Create padding arrays (pixels may be None in state-only mode)
            if pixels is not None:
                pixels_pad = np.zeros((pad_len,) + pixels.shape[1:], dtype=pixels.dtype)
                pixels_out = np.concatenate([pixels, pixels_pad], axis=0)
            else:
                pixels_out = None

            states_pad = np.zeros((pad_len,) + states.shape[1:], dtype=states.dtype)
            actions_pad = np.zeros((pad_len,) + actions.shape[1:], dtype=actions.dtype)
            rewards_pad = np.zeros(pad_len, dtype=rewards.dtype)
            is_last_pad = np.ones(
                pad_len, dtype=is_last.dtype
            )  # Mark padded as terminated
            is_terminal_pad = np.ones(
                pad_len, dtype=is_terminal.dtype
            )  # Padded steps should not bootstrap
            if future_returns is not None:
                future_returns_out = np.concatenate(
                    [future_returns, np.zeros(pad_len, dtype=future_returns.dtype)],
                    axis=0,
                )
            else:
                future_returns_out = None

            # Mask: 1 for real steps, 0 for padded
            mask = np.concatenate(
                [np.ones(ep_len, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)]
            )

            return (
                pixels_out,
                np.concatenate([states, states_pad], axis=0),
                np.concatenate([actions, actions_pad], axis=0),
                np.concatenate([rewards, rewards_pad], axis=0),
                np.concatenate([is_last, is_last_pad], axis=0),
                np.concatenate([is_terminal, is_terminal_pad], axis=0),
                future_returns_out,
                mask,
            )

    def sample(self, batch_size, recent_fraction=0.2):
        """
        Sample a batch of fixed-length subsequences with recent bias.

        Blocks until buffer has min_episodes, then returns instantly.
        Samples recent_fraction of batch from newest episodes (recency bias).

        Args:
            batch_size: Number of subsequences to sample
            recent_fraction: Fraction of batch to sample from recent episodes (default 0.2)

        Returns:
            List of (pixels, states, actions, rewards, is_last, is_terminal,
            future_returns, mask) tuples, each with shape (sequence_length, ...)
        """
        # Wait for buffer to have enough episodes (only blocks on startup)
        self.ready_event.wait()

        # Guard buffer sampling to keep episode list/metadata consistent.
        with self.lock:
            buffer_len = len(self.buffer)
            n_recent = int(batch_size * recent_fraction)
            n_uniform = batch_size - n_recent

            # Recent episodes: newest 20% of buffer (or at least 1)
            recent_count = max(1, buffer_len // 5)
            recent_start = buffer_len - recent_count

            # Sample indices: n_recent from recent, n_uniform from all
            if recent_count < n_recent:
                recent_indices = random.choices(
                    range(recent_start, buffer_len), k=n_recent
                )
            else:
                recent_indices = random.sample(
                    range(recent_start, buffer_len), k=n_recent
                )

            if buffer_len < n_uniform:
                uniform_indices = random.choices(range(buffer_len), k=n_uniform)
            else:
                uniform_indices = random.sample(range(buffer_len), k=n_uniform)

            indices = recent_indices + uniform_indices

            return [self._sample_subsequence(self.buffer[i]) for i in indices]

    def sample_tensors(self, batch_size, device, use_pixels=False, target_size=None, recent_fraction=0.2) -> EnvData:
        """
        Samples a batch and returns an EnvData namedtuple with ready-to-use PyTorch tensors.
        Handles pixels resizing and symlog of state.
        """
        raw_batch = self.sample(batch_size, recent_fraction)

        batch_pixels, batch_pixels_original = [], []
        batch_states, batch_actions, batch_rewards = [], [], []
        batch_is_last, batch_is_terminal = [], []
        batch_future_returns, batch_mask = [], []

        for (
            pixels,
            states,
            actions,
            rewards,
            is_last,
            is_terminal,
            future_returns,
            mask,
        ) in raw_batch:
            if use_pixels and pixels is not None:
                pixels_tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2)
                batch_pixels_original.append(pixels_tensor)
                if target_size and pixels_tensor.shape[-2:] != target_size:
                    pixels_tensor = resize_pixels_to_target(pixels_tensor, target_size)
                batch_pixels.append(pixels_tensor)

            batch_states.append(torch.from_numpy(states))
            batch_actions.append(torch.from_numpy(actions))
            batch_rewards.append(torch.from_numpy(rewards))
            batch_is_last.append(torch.from_numpy(is_last))
            batch_is_terminal.append(torch.from_numpy(is_terminal))
            if future_returns is not None:
                batch_future_returns.append(torch.from_numpy(future_returns))
            batch_mask.append(torch.from_numpy(mask))

        if use_pixels and batch_pixels:
            pixels_out = torch.stack(batch_pixels).to(device).float()
            pixels_original_out = torch.stack(batch_pixels_original).to(device).float()
        else:
            pixels_out, pixels_original_out = None, None

        states_out = torch.stack(batch_states).to(device)
        future_returns_out = (
            torch.stack(batch_future_returns).to(device)
            if len(batch_future_returns) == len(raw_batch)
            else None
        )

        return EnvData(
            states=states_out,
            actions=torch.stack(batch_actions).to(device),
            rewards=torch.stack(batch_rewards).to(device),
            is_last=torch.stack(batch_is_last).to(device),
            is_terminal=torch.stack(batch_is_terminal).to(device),
            future_returns=future_returns_out,
            mask=torch.stack(batch_mask).to(device),
            pixels=pixels_out,
            pixels_original=pixels_original_out,
        )

    def __len__(self):
        """Current number of episodes in buffer."""
        with self.lock:
            return len(self.buffer)

    @property
    def total_episodes_added(self):
        """Total episodes that have passed through the buffer."""
        with self.lock:
            return self._episodes_added

    @property
    def is_ready(self):
        """Whether buffer has minimum episodes for sampling."""
        return self.ready_event.is_set()

    @property
    def avg_episode_length(self):
        """Average length of all episodes added to buffer."""
        with self.lock:
            if self._episodes_added == 0:
                return 0
            return self._total_steps / self._episodes_added

    @property
    def total_env_steps(self):
        """Total environment steps collected (for replay ratio gating)."""
        with self.lock:
            return self._total_steps

    @property
    def recent_avg_episode_length(self):
        """Average length of recent 100 episodes (for tracking learning progress)."""
        with self.lock:
            if not self._recent_ep_lengths:
                return 0
            return sum(self._recent_ep_lengths) / len(self._recent_ep_lengths)

    def maybe_increase_sequence_length(self, threshold=0.8, increment=8):
        """
        Increase sequence_length if avg episode length approaches it.

        Args:
            threshold: Trigger when avg_len > threshold * seq_len
            increment: How much to increase seq_len

        Returns:
            New sequence_length (unchanged if no increase)
        """
        avg_len = self.avg_episode_length
        if avg_len > threshold * self.sequence_length:
            old_len = self.sequence_length
            self.sequence_length = min(
                self.sequence_length + increment, self._max_sequence_length
            )
            if self.sequence_length > old_len:
                print(
                    f"Increased sequence_length: {old_len} -> {self.sequence_length} "
                    f"(avg_ep_len={avg_len:.1f})",
                    flush=True,
                )
        return self.sequence_length
