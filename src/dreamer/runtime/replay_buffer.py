"""
Episode Replay Buffer for Dreamer training.

Stores complete episodes, samples fixed-length subsequences.
This matches DreamerV3's approach for consistent batch shapes.
"""

import threading
import random
import numpy as np
import torch
from collections import defaultdict, deque
from queue import Empty
from typing import NamedTuple, Optional

from ..models.math_utils import resize_pixels_to_target


class EnvData(NamedTuple):
    """Immutable batch of environment data sampled from replay."""

    states: torch.Tensor  # (B, T, n_obs) — raw env vectors
    actions: torch.Tensor  # (B, T, n_actions)
    rewards: torch.Tensor  # (B, T)
    is_first: torch.Tensor  # (B, T), recurrent reset before this row
    is_last: torch.Tensor  # (B, T)
    is_terminal: torch.Tensor  # (B, T)
    future_returns: Optional[torch.Tensor]  # (B, T), when exact targets enabled
    continue_weights: torch.Tensor  # (B, T), replay-window inclusion correction
    mask: torch.Tensor  # (B, T) — 1=real, 0=padded
    pixels: Optional[torch.Tensor] = None  # (B, T, C, H, W)
    pixels_original: Optional[torch.Tensor] = None  # (B, T, C, H, W)


def continuation_inclusion_weights(
    episode_length: int,
    sequence_length: int,
    start: int,
) -> np.ndarray:
    """Correct per-episode window-edge bias for continuation supervision.

    Full windows include interior transitions more often than episode-edge
    transitions. The returned inverse-inclusion weights make every transition's
    aggregate weight equal when enumerating all valid windows, while preserving
    mean weight one over the sampled rows. Short padded episodes have one window
    and therefore need no correction.
    """
    if episode_length <= 0 or sequence_length <= 0:
        raise ValueError("episode and sequence lengths must be positive")
    if episode_length < sequence_length:
        if start != 0:
            raise ValueError("short padded episodes must start at zero")
        return np.concatenate(
            [
                np.ones(episode_length, dtype=np.float32),
                np.zeros(sequence_length - episode_length, dtype=np.float32),
            ]
        )

    valid_starts = episode_length - sequence_length + 1
    if not 0 <= start < valid_starts:
        raise ValueError("subsequence start is outside the episode")
    mean_multiplicity = sequence_length * valid_starts / episode_length
    weights = np.empty(sequence_length, dtype=np.float32)
    for offset in range(sequence_length):
        index = start + offset
        first_start = max(0, index - sequence_length + 1)
        last_start = min(index, valid_starts - 1)
        multiplicity = last_start - first_start + 1
        weights[offset] = mean_multiplicity / multiplicity
    return weights


class EpisodeReplayBuffer:
    """
    Thread-safe replay buffer that stores complete episodes and samples
    fixed-length subsequences.

    Design:
    - Background thread continuously drains the mp.Queue (never blocks collectors)
    - Circular buffer stores up to max_episodes (FIFO eviction)
    - Samples uniformly over valid sequence starts, not uniformly over episodes
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
        sequence_mode="episode",
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
            sequence_mode: ``episode`` for historical contained windows or
                ``stream`` for per-collector windows that cross episode resets
        """
        self.data_queue = data_queue
        self.max_episodes = max_episodes
        self.min_episodes = min_episodes
        self.sequence_length = sequence_length
        self.gamma = float(gamma)
        self.compute_future_returns = bool(compute_future_returns)
        self.throttle_collection = bool(throttle_collection)
        self.sequence_mode = str(sequence_mode)
        if self.sequence_mode not in {"episode", "stream"}:
            raise ValueError("sequence_mode must be 'episode' or 'stream'")

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
        self._last_episode_id: dict[int, int] = defaultdict(lambda: -1)

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
                if not self._wait_for_collection_budget():
                    break
                self.add_episode(episode)
            except Empty:
                continue

    def _wait_for_collection_budget(self) -> bool:
        """Wait until training has budgeted collection, allowing one-episode debt."""
        if not self.throttle_collection:
            return True

        with self._budget_changed:
            while (
                not self._stop
                and self._env_step_budget is not None
                and self._total_steps >= self._env_step_budget
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
        collector_id = int(episode[7]) if len(episode) > 7 else 0
        if len(episode) > 8:
            episode_id = int(episode[8])
        else:
            episode_id = self._last_episode_id[collector_id] + 1
        self._last_episode_id[collector_id] = max(
            self._last_episode_id[collector_id], episode_id
        )
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
            collector_id,
            episode_id,
        )
        with self.lock:
            self.buffer.append(stored_episode)
            self._episodes_added += 1
            self._total_steps += ep_len
            self._recent_ep_lengths.append(ep_len)
            has_complete_sequence = self.sequence_mode == "episode" or bool(
                self._stream_start_candidates()
            )
            if (
                len(self.buffer) >= self.min_episodes
                and has_complete_sequence
                and not self.ready_event.is_set()
            ):
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
        is_first = np.concatenate(
            [
                np.ones(1, dtype=np.bool_),
                np.zeros(seq_len - 1, dtype=np.bool_),
            ]
        )

        if ep_len >= seq_len:
            # Sample random start point
            start = random.randint(0, ep_len - seq_len)
            mask = np.ones(seq_len, dtype=np.float32)  # All real steps
            continue_weights = continuation_inclusion_weights(
                ep_len, seq_len, start
            )
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
                continue_weights,
                mask,
                is_first,
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
            continue_weights = continuation_inclusion_weights(ep_len, seq_len, 0)

            return (
                pixels_out,
                np.concatenate([states, states_pad], axis=0),
                np.concatenate([actions, actions_pad], axis=0),
                np.concatenate([rewards, rewards_pad], axis=0),
                np.concatenate([is_last, is_last_pad], axis=0),
                np.concatenate([is_terminal, is_terminal_pad], axis=0),
                future_returns_out,
                continue_weights,
                mask,
                is_first,
            )

    def _stream_segments(self):
        """Return ordered, gap-free episode segments for each collector."""
        by_collector = defaultdict(list)
        for buffer_index, episode in enumerate(self.buffer):
            by_collector[int(episode[8])].append(
                (int(episode[9]), buffer_index, episode)
            )

        segments = []
        for entries in by_collector.values():
            entries.sort(key=lambda item: item[0])
            current = []
            previous_id = None
            for episode_id, buffer_index, episode in entries:
                if previous_id is not None and episode_id != previous_id + 1:
                    if current:
                        segments.append(current)
                    current = []
                current.append((buffer_index, episode))
                previous_id = episode_id
            if current:
                segments.append(current)
        return segments

    def _stream_start_candidates(self):
        """Enumerate episode-local ranges containing every valid stream start."""
        candidates = []
        for segment in self._stream_segments():
            remaining = 0
            valid_counts = [0] * len(segment)
            for index in range(len(segment) - 1, -1, -1):
                episode_length = len(segment[index][1][1])
                remaining += episode_length
                valid_counts[index] = min(
                    episode_length,
                    max(0, remaining - self.sequence_length + 1),
                )
            for episode_index, valid_count in enumerate(valid_counts):
                if valid_count:
                    buffer_index, _episode = segment[episode_index]
                    candidates.append(
                        (segment, episode_index, valid_count, buffer_index)
                    )
        return candidates

    def _sample_stream_subsequence(self, candidate):
        """Sample one full sequence from a same-collector episode stream."""
        segment, episode_index, valid_count, _buffer_index = candidate
        offset = random.randint(0, valid_count - 1)
        remaining = self.sequence_length
        pieces = []
        while remaining:
            if episode_index >= len(segment):
                raise RuntimeError("stream candidate does not contain a full sequence")
            _index, episode = segment[episode_index]
            available = len(episode[1]) - offset
            used = min(remaining, available)
            pieces.append((episode, offset, offset + used))
            remaining -= used
            episode_index += 1
            offset = 0

        def concatenate(field):
            if pieces[0][0][field] is None:
                return None
            values = [episode[field][start:stop] for episode, start, stop in pieces]
            return np.concatenate(values, axis=0)

        is_first_parts = []
        for _episode, start, stop in pieces:
            part = np.zeros(stop - start, dtype=np.bool_)
            if start == 0:
                part[0] = True
            is_first_parts.append(part)
        is_first = np.concatenate(is_first_parts)
        # Sampled carries always begin at zero, even for a mid-episode start.
        is_first[0] = True
        return (
            concatenate(0),
            concatenate(1),
            concatenate(2),
            concatenate(3),
            concatenate(4),
            concatenate(5),
            concatenate(7),
            np.ones(self.sequence_length, dtype=np.float32),
            np.ones(self.sequence_length, dtype=np.float32),
            is_first,
        )

    def _sample_stream(self, batch_size, recent_fraction):
        candidates = self._stream_start_candidates()
        if not candidates:
            raise RuntimeError("replay stream has no complete sequence")

        n_recent = int(batch_size * recent_fraction)
        n_uniform = batch_size - n_recent
        recent_start = len(self.buffer) - max(1, len(self.buffer) // 5)
        recent = [item for item in candidates if item[3] >= recent_start]
        if not recent:
            recent = candidates

        selected = []
        if n_recent:
            selected.extend(
                random.choices(
                    recent,
                    weights=[item[2] for item in recent],
                    k=n_recent,
                )
            )
        if n_uniform:
            selected.extend(
                random.choices(
                    candidates,
                    weights=[item[2] for item in candidates],
                    k=n_uniform,
                )
            )
        return [self._sample_stream_subsequence(item) for item in selected]

    def sample(self, batch_size, recent_fraction=0.2):
        """
        Sample a batch of fixed-length subsequences with recent bias.

        Blocks until buffer has min_episodes, then returns instantly.
        Samples recent_fraction of batch from newest episodes (recency bias).
        Within each pool, episode probability is proportional to the number of
        valid starts so every stored training window is equally likely.

        Args:
            batch_size: Number of subsequences to sample
            recent_fraction: Fraction of batch to sample from recent episodes (default 0.2)

        Returns:
            List of (pixels, states, actions, rewards, is_last, is_terminal,
            future_returns, continue_weights, mask, is_first) tuples, each with
            shape (sequence_length, ...)
        """
        # Wait for buffer to have enough episodes (only blocks on startup)
        self.ready_event.wait()

        # Guard buffer sampling to keep episode list/metadata consistent.
        with self.lock:
            if self.sequence_mode == "stream":
                return self._sample_stream(batch_size, recent_fraction)

            buffer_len = len(self.buffer)
            n_recent = int(batch_size * recent_fraction)
            n_uniform = batch_size - n_recent

            # Recent episodes: newest 20% of buffer (or at least 1)
            recent_count = max(1, buffer_len // 5)
            recent_start = buffer_len - recent_count
            start_counts = [
                max(1, len(episode[1]) - self.sequence_length + 1)
                for episode in self.buffer
            ]

            # Sampling with replacement matches uniform selection over valid
            # starts: the same long episode may supply multiple subsequences.
            if n_recent:
                recent_indices = random.choices(
                    range(recent_start, buffer_len),
                    weights=start_counts[recent_start:],
                    k=n_recent,
                )
            else:
                recent_indices = []

            if n_uniform:
                uniform_indices = random.choices(
                    range(buffer_len), weights=start_counts, k=n_uniform
                )
            else:
                uniform_indices = []

            indices = recent_indices + uniform_indices

            return [self._sample_subsequence(self.buffer[i]) for i in indices]

    def sample_tensors(
        self,
        batch_size,
        device,
        use_pixels=False,
        target_size=None,
        recent_fraction=0.2,
    ) -> EnvData:
        """
        Samples a batch and returns an EnvData namedtuple with ready-to-use PyTorch tensors.
        Handles pixels resizing and symlog of state.
        """
        raw_batch = self.sample(batch_size, recent_fraction)

        batch_pixels, batch_pixels_original = [], []
        batch_states, batch_actions, batch_rewards = [], [], []
        batch_is_first, batch_is_last, batch_is_terminal = [], [], []
        batch_future_returns, batch_continue_weights, batch_mask = [], [], []

        for (
            pixels,
            states,
            actions,
            rewards,
            is_last,
            is_terminal,
            future_returns,
            continue_weights,
            mask,
            is_first,
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
            batch_is_first.append(torch.from_numpy(is_first))
            batch_is_last.append(torch.from_numpy(is_last))
            batch_is_terminal.append(torch.from_numpy(is_terminal))
            if future_returns is not None:
                batch_future_returns.append(torch.from_numpy(future_returns))
            batch_continue_weights.append(torch.from_numpy(continue_weights))
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
            is_first=torch.stack(batch_is_first).to(device),
            is_last=torch.stack(batch_is_last).to(device),
            is_terminal=torch.stack(batch_is_terminal).to(device),
            future_returns=future_returns_out,
            continue_weights=torch.stack(batch_continue_weights).to(device),
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
