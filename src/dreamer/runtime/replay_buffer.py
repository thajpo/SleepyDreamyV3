"""Episode-backed replay for fixed-length DreamerV3 training sequences.

Historical episode mode samples contained subsequences and pads short episodes.
Reference-style stream mode joins only consecutive episodes from one collector,
may cross resets, and marks each boundary so the RSSM can reset its carry.
"""

import logging
import threading
import random
import numpy as np
import torch
from collections import defaultdict, deque
from queue import Empty
from typing import NamedTuple, Optional

from ..models.math_utils import resize_pixels_to_target


logger = logging.getLogger(__name__)


class EnvData(NamedTuple):
    """Immutable batch of environment data sampled from replay."""

    states: torch.Tensor  # (B, T, n_obs) — raw env vectors
    actions: torch.Tensor  # (B, T, n_actions)
    rewards: torch.Tensor  # (B, T)
    is_first: torch.Tensor  # (B, T), recurrent reset before this row
    is_last: torch.Tensor  # (B, T)
    is_terminal: torch.Tensor  # (B, T)
    future_returns: Optional[torch.Tensor]  # (B, T), when exact targets enabled
    continue_weights: torch.Tensor  # (B, T), continuation supervision weights
    mask: torch.Tensor  # (B, T) — 1=real, 0=padded
    pixels: Optional[torch.Tensor] = None  # (B, T, C, H, W)
    pixels_original: Optional[torch.Tensor] = None  # (B, T, C, H, W)


class _ChunkedArray:
    """Append-only logical array that stores each collector chunk exactly once."""

    def __init__(self, value: np.ndarray):
        self._chunks: list[np.ndarray] = []
        self._length = 0
        self._tail_shape = value.shape[1:]
        self._dtype = value.dtype
        self.append(value)

    def append(self, value: np.ndarray) -> None:
        if value.ndim == 0 or len(value) == 0:
            raise ValueError("replay chunks must contain at least one row")
        if value.shape[1:] != self._tail_shape or value.dtype != self._dtype:
            raise ValueError("replay chunk array shape or dtype changed mid-episode")
        self._chunks.append(value)
        self._length += len(value)

    def __len__(self) -> int:
        return self._length

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._length, *self._tail_shape)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def __getitem__(self, key):
        if isinstance(key, int):
            index = key if key >= 0 else self._length + key
            if not 0 <= index < self._length:
                raise IndexError(index)
            for chunk in self._chunks:
                if index < len(chunk):
                    return chunk[index]
                index -= len(chunk)
            raise IndexError(key)
        if not isinstance(key, slice):
            raise TypeError("chunked replay arrays support integer and slice access")
        start, stop, step = key.indices(self._length)
        if step != 1:
            raise ValueError("chunked replay slices require a unit step")
        if start >= stop:
            return np.empty((0, *self._tail_shape), dtype=self._dtype)
        pieces = []
        cursor = 0
        for chunk in self._chunks:
            chunk_stop = cursor + len(chunk)
            if chunk_stop > start and cursor < stop:
                local_start = max(0, start - cursor)
                local_stop = min(len(chunk), stop - cursor)
                pieces.append(chunk[local_start:local_stop])
            cursor = chunk_stop
            if cursor >= stop:
                break
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)


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
    Thread-safe replay buffer that stores complete episodes and samples fixed-
    length episode-contained or same-collector stream subsequences.

    Design:
    - Background thread continuously drains the mp.Queue (never blocks collectors)
    - Circular buffer stores up to max_episodes (FIFO eviction)
    - Episode mode pads short episodes and corrects window-edge inclusion bias
    - Stream mode uses every gap-free start with unit continuation weights
    - Sampling is uniform over valid starts, not uniform over episodes
    - Readiness clears after eviction if no complete configured sequence remains
    - sample() blocks whenever the configured replay population is not sampleable
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
        online_replay=False,
        continuous_delivery=False,
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
            online_replay: Prefer each new non-overlapping stream sequence once
                before filling batches from the configured replay selector.
            continuous_delivery: Accept append-only partial-episode chunk packets.
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
        self.online_replay = bool(online_replay)
        if self.online_replay and self.sequence_mode != "stream":
            raise ValueError("online_replay requires sequence_mode='stream'")
        self.continuous_delivery = bool(continuous_delivery)
        if self.continuous_delivery and self.sequence_mode != "stream":
            raise ValueError("continuous_delivery requires sequence_mode='stream'")
        if self.continuous_delivery and self.compute_future_returns:
            raise ValueError(
                "continuous_delivery cannot compute full-episode future returns"
            )

        self.buffer = deque(maxlen=max_episodes)
        self.lock = threading.Lock()
        self._budget_changed = threading.Condition(self.lock)
        self.ready_event = threading.Event()

        self._stop = False
        self._thread = None
        self._episodes_added = 0
        self._total_steps = 0  # For tracking average episode length
        self._completed_steps = 0
        self._recent_ep_lengths = deque(maxlen=100)  # Track recent episode lengths
        self._max_sequence_length = 256  # Cap for adaptive growth
        self._env_step_budget: float | None = None
        self._last_episode_id: dict[int, int] = defaultdict(lambda: -1)
        # Keep only tiny stream-position descriptors, never copied observations.
        # The cap prevents an unconsumed startup/backpressure queue from scaling
        # without bound while remaining far above the expected live backlog.
        self._online_queue = deque(maxlen=max_episodes * max(1, sequence_length))
        self._online_pending_start: dict[int, tuple[int, int, int]] = {}
        self._online_pending_length: dict[int, int] = defaultdict(int)
        self._online_last_episode_id: dict[int, int] = {}
        self._online_last_episode_complete: dict[int, bool] = {}
        self._online_descriptors_dropped = 0
        self._online_samples = 0
        self._sequence_samples = 0
        self._last_online_sample_fraction = 0.0
        self._active_partial_episodes: dict[tuple[int, int], list] = {}
        self._chunks_added = 0
        self._chunk_rows_added = 0
        self._preterminal_chunks_added = 0

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
                and self.ready_event.is_set()
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

    def allow_env_steps_until(self, total_steps: float) -> None:
        """Extend the collection budget to an absolute environment-step target."""
        if not self.throttle_collection:
            return
        if total_steps < 0:
            raise ValueError("collection budget target must be non-negative")
        with self._budget_changed:
            if self._env_step_budget is None:
                return
            target = float(total_steps)
            if target > self._env_step_budget:
                self._env_step_budget = target
                self._budget_changed.notify_all()

    def add_episode(self, episode):
        """Insert a complete episode or one append-only partial-episode chunk."""
        if len(episode) >= 11:
            self._add_episode_chunk(episode)
            return
        self._add_complete_episode(episode)

    def _add_complete_episode(self, episode):
        """Insert one historical complete-episode packet."""
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
        stored_episode = [
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
            True,
        ]
        with self._budget_changed:
            was_ready = self.ready_event.is_set()
            self._make_room_for_episode()
            self.buffer.append(stored_episode)
            if self.online_replay:
                self._enqueue_online_sequences(
                    collector_id, episode_id, len(states), episode_complete=True
                )
            self._episodes_added += 1
            self._total_steps += ep_len
            self._completed_steps += ep_len
            self._recent_ep_lengths.append(ep_len)
            self._update_readiness(was_ready)

    def _add_episode_chunk(self, packet) -> None:
        """Append a non-overlapping collector chunk to one logical episode."""
        if not self.continuous_delivery:
            raise ValueError("received replay chunk while continuous delivery is off")
        pixels, states, actions, rewards, is_last, is_terminal = packet[:6]
        chunk_env_steps = int(packet[6])
        collector_id = int(packet[7])
        episode_id = int(packet[8])
        episode_complete = bool(packet[9])
        row_offset = int(packet[10])
        arrays = (states, actions, rewards, is_last, is_terminal)
        row_count = len(states)
        if row_count <= 0 or any(len(value) != row_count for value in arrays):
            raise ValueError("replay chunk fields must have the same positive length")
        if pixels is not None and len(pixels) != row_count:
            raise ValueError("replay pixel chunk length does not match state rows")
        if chunk_env_steps <= 0:
            raise ValueError("replay chunk environment-step count must be positive")
        if episode_complete:
            if not bool(is_last[-1]):
                raise ValueError("completed replay chunk must end with is_last")
        elif np.any(is_last) or np.any(is_terminal):
            raise ValueError("pre-terminal replay chunk contains a terminal marker")

        key = (collector_id, episode_id)
        with self._budget_changed:
            was_ready = self.ready_event.is_set()
            stored_episode = self._active_partial_episodes.get(key)
            if stored_episode is None:
                if row_offset != 0:
                    raise ValueError("first replay chunk must have row_offset=0")
                if any(
                    active_collector == collector_id
                    for active_collector, _active_episode in self._active_partial_episodes
                ):
                    raise ValueError(
                        "cannot start a new replay episode before its prior chunk stream ends"
                    )
                if key in self._episode_lookup():
                    raise ValueError("received a chunk for an already completed episode")
                previous_id = self._last_episode_id[collector_id]
                if previous_id >= 0 and episode_id != previous_id + 1:
                    raise ValueError("replay chunk episode IDs must be consecutive")
                self._make_room_for_episode()
                chunked_pixels = _ChunkedArray(pixels) if pixels is not None else None
                stored_episode = [
                    chunked_pixels,
                    _ChunkedArray(states),
                    _ChunkedArray(actions),
                    _ChunkedArray(rewards),
                    _ChunkedArray(is_last),
                    _ChunkedArray(is_terminal),
                    chunk_env_steps,
                    None,
                    collector_id,
                    episode_id,
                    episode_complete,
                ]
                self.buffer.append(stored_episode)
                self._last_episode_id[collector_id] = episode_id
                if not episode_complete:
                    self._active_partial_episodes[key] = stored_episode
            else:
                if row_offset != len(stored_episode[1]):
                    raise ValueError(
                        "replay chunk row_offset does not match stored episode length"
                    )
                if bool(stored_episode[10]):
                    raise ValueError("cannot append to a completed replay episode")
                chunk_fields = (pixels, states, actions, rewards, is_last, is_terminal)
                for field, value in enumerate(chunk_fields):
                    target = stored_episode[field]
                    if (target is None) != (value is None):
                        raise ValueError("replay pixel presence changed mid-episode")
                    if target is not None:
                        target.append(value)
                stored_episode[6] += chunk_env_steps
                stored_episode[10] = episode_complete

            if self.online_replay:
                self._enqueue_online_sequences(
                    collector_id,
                    episode_id,
                    row_count,
                    episode_complete=episode_complete,
                )
            self._total_steps += chunk_env_steps
            self._chunks_added += 1
            self._chunk_rows_added += row_count
            if not episode_complete:
                self._preterminal_chunks_added += 1
            else:
                self._active_partial_episodes.pop(key, None)
                self._episodes_added += 1
                self._completed_steps += int(stored_episode[6])
                self._recent_ep_lengths.append(int(stored_episode[6]))
            self._update_readiness(was_ready)

    def _make_room_for_episode(self) -> None:
        if len(self.buffer) < self.max_episodes:
            return
        for index, episode in enumerate(self.buffer):
            if bool(episode[10]):
                del self.buffer[index]
                return
        raise RuntimeError("replay capacity contains only active partial episodes")

    def _update_readiness(self, was_ready: bool) -> None:
        completed_in_buffer = sum(bool(episode[10]) for episode in self.buffer)
        has_complete_sequence = self.sequence_mode == "episode" or bool(
            self._stream_start_candidates()
        )
        is_ready = (
            completed_in_buffer >= self.min_episodes and has_complete_sequence
        )
        if is_ready:
            self.ready_event.set()
        else:
            self.ready_event.clear()
        if is_ready and not was_ready:
            if self.throttle_collection and self._env_step_budget is None:
                self._env_step_budget = float(self._total_steps)
            logger.info(
                "replay_ready completed_episodes=%d preterminal_chunks=%d "
                "chunk_rows=%d",
                completed_in_buffer,
                self._preterminal_chunks_added,
                self._chunk_rows_added,
            )
        if is_ready != was_ready:
            self._budget_changed.notify_all()

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

    def _episode_lookup(self):
        return {
            (int(episode[8]), int(episode[9])): episode
            for episode in self.buffer
        }

    def _advance_stream_position(
        self, position: tuple[int, int, int], steps: int
    ) -> tuple[int, int, int]:
        """Advance a same-collector descriptor across consecutive episodes."""
        collector_id, episode_id, offset = position
        lookup = self._episode_lookup()
        remaining = int(steps)
        while remaining > 0:
            episode = lookup.get((collector_id, episode_id))
            if episode is None:
                raise RuntimeError("online replay position is outside stored stream")
            available = len(episode[1]) - offset
            if remaining < available:
                return collector_id, episode_id, offset + remaining
            remaining -= available
            if bool(episode[10]):
                episode_id += 1
                offset = 0
            else:
                offset = len(episode[1])
                if remaining:
                    raise RuntimeError(
                        "online replay advanced beyond available partial episode"
                    )
        return collector_id, episode_id, offset

    def _enqueue_online_sequences(
        self,
        collector_id: int,
        episode_id: int,
        rows_added: int,
        *,
        episode_complete: bool,
    ) -> None:
        """Register each new non-overlapping sequence exactly once."""
        previous_id = self._online_last_episode_id.get(collector_id)
        previous_complete = self._online_last_episode_complete.get(collector_id, True)
        is_same_partial = previous_id == episode_id and not previous_complete
        is_next_episode = (
            previous_id is not None
            and episode_id == previous_id + 1
            and previous_complete
        )
        if previous_id is None or not (is_same_partial or is_next_episode):
            self._online_pending_start[collector_id] = (
                collector_id,
                episode_id,
                0,
            )
            self._online_pending_length[collector_id] = 0

        pending = self._online_pending_length[collector_id] + int(rows_added)
        start = self._online_pending_start[collector_id]
        while pending >= self.sequence_length:
            if len(self._online_queue) == self._online_queue.maxlen:
                self._online_descriptors_dropped += 1
            self._online_queue.append(start)
            start = self._advance_stream_position(start, self.sequence_length)
            pending -= self.sequence_length
        self._online_pending_start[collector_id] = start
        self._online_pending_length[collector_id] = pending
        self._online_last_episode_id[collector_id] = episode_id
        self._online_last_episode_complete[collector_id] = episode_complete

    def _compose_stream_pieces(self, pieces):
        """Assemble aligned replay fields from consecutive episode slices."""
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
        # Every sampled sequence starts with a zero carry. Preserve additional
        # true episode boundaries inside the sequence.
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

    def _sample_stream_position(self, position):
        """Resolve an online descriptor, returning None after eviction or a gap."""
        collector_id, episode_id, offset = position
        lookup = self._episode_lookup()
        remaining = self.sequence_length
        pieces = []
        while remaining:
            episode = lookup.get((collector_id, episode_id))
            if episode is None:
                return None
            available = len(episode[1]) - offset
            if available <= 0:
                return None
            used = min(remaining, available)
            pieces.append((episode, offset, offset + used))
            remaining -= used
            episode_id += 1
            offset = 0
        return self._compose_stream_pieces(pieces)

    def _take_online_stream_samples(self, count: int):
        samples = []
        while self._online_queue and len(samples) < count:
            descriptor = self._online_queue.popleft()
            sample = self._sample_stream_position(descriptor)
            if sample is not None:
                samples.append(sample)
        return samples

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

    def _sample_stream_subsequence_with_position(self, candidate, *, rng=None):
        """Sample a stream and return its stable collector/episode position."""
        if rng is None:
            rng = random
        segment, episode_index, valid_count, _buffer_index = candidate
        offset = rng.randint(0, valid_count - 1)
        start_episode = segment[episode_index][1]
        position = (int(start_episode[8]), int(start_episode[9]), int(offset))
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

        return self._compose_stream_pieces(pieces), position

    def _sample_stream_subsequence(self, candidate, *, rng=None):
        """Sample one full sequence from a same-collector episode stream."""
        sample, _position = self._sample_stream_subsequence_with_position(
            candidate, rng=rng
        )
        return sample

    def sample_state_evidence(self, count: int, *, seed: int) -> dict[str, np.ndarray]:
        """Read-only uniform stream sample for checkpoint-local diagnostics.

        A private RNG prevents observability from changing the trainer's future
        replay choices. Online descriptors are intentionally neither consumed nor
        copied: this estimates the uniform valid-start population that supplies
        the large majority of authored replay batches.
        """
        if count <= 0:
            raise ValueError("replay evidence sample count must be positive")
        if self.sequence_mode != "stream":
            raise ValueError("replay evidence sampling requires stream mode")

        rng = random.Random(int(seed))
        with self.lock:
            candidates = self._stream_start_candidates()
            if not candidates:
                raise RuntimeError("replay stream has no complete sequence")
            selected = rng.choices(
                candidates,
                weights=[item[2] for item in candidates],
                k=count,
            )
            sampled = [
                self._sample_stream_subsequence_with_position(candidate, rng=rng)
                for candidate in selected
            ]
            samples = [sample for sample, _position in sampled]
            positions = [position for _sample, position in sampled]

        def stack(field: int) -> np.ndarray:
            values = [sample[field] for sample in samples]
            if any(value is None for value in values):
                raise ValueError("state replay evidence cannot contain missing fields")
            return np.stack([np.asarray(value) for value in values])

        states = stack(1)
        if states.shape[-1] == 0:
            raise ValueError("replay evidence sampling requires vector observations")
        return {
            "states": states,
            "actions": stack(2),
            "rewards": stack(3),
            "is_last": stack(4),
            "is_terminal": stack(5),
            "mask": stack(8),
            "is_first": stack(9),
            "start_collector_id": np.asarray(
                [position[0] for position in positions], dtype=np.int64
            ),
            "start_episode_id": np.asarray(
                [position[1] for position in positions], dtype=np.int64
            ),
            "start_offset": np.asarray(
                [position[2] for position in positions], dtype=np.int64
            ),
            "sample_count": np.asarray(count, dtype=np.int64),
            "sequence_length": np.asarray(self.sequence_length, dtype=np.int64),
            "seed": np.asarray(seed, dtype=np.int64),
            "selector": np.asarray("uniform_stream"),
        }

    def _sample_stream(self, batch_size, recent_fraction, candidates=None):
        if candidates is None:
            candidates = self._stream_start_candidates()
        if not candidates:
            raise RuntimeError("replay stream has no complete sequence")

        online_samples = (
            self._take_online_stream_samples(batch_size)
            if self.online_replay
            else []
        )
        remaining_batch = batch_size - len(online_samples)
        n_recent = int(remaining_batch * recent_fraction)
        n_uniform = remaining_batch - n_recent
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
        selector_samples = [
            self._sample_stream_subsequence(item) for item in selected
        ]
        self._online_samples += len(online_samples)
        self._sequence_samples += batch_size
        self._last_online_sample_fraction = len(online_samples) / batch_size
        return online_samples + selector_samples

    def sample(self, batch_size, recent_fraction=0.0):
        """
        Sample fixed-length subsequences, optionally with explicit recent bias.

        Blocks until the configured replay population can provide a sequence.
        Samples recent_fraction of the batch from starts in newer episodes.
        Within each pool, selection is proportional to valid-start count so
        every stored episode window or same-collector stream start is equally
        likely.

        Args:
            batch_size: Number of subsequences to sample
            recent_fraction: Fraction of batch forced from recent episodes (default 0)

        Returns:
            List of (pixels, states, actions, rewards, is_last, is_terminal,
            future_returns, continue_weights, mask, is_first) tuples, each with
            shape (sequence_length, ...)
        """
        if self.sequence_mode == "stream":
            while True:
                self.ready_event.wait()
                with self.lock:
                    if not self.ready_event.is_set():
                        continue
                    candidates = self._stream_start_candidates()
                    if candidates:
                        return self._sample_stream(
                            batch_size, recent_fraction, candidates=candidates
                        )
                    self.ready_event.clear()
                    self._budget_changed.notify_all()

        self.ready_event.wait()
        with self.lock:
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
        recent_fraction=0.0,
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
        """Whether the configured replay population can be sampled."""
        return self.ready_event.is_set()

    @property
    def avg_episode_length(self):
        """Average length of all episodes added to buffer."""
        with self.lock:
            if self._episodes_added == 0:
                return 0
            return self._completed_steps / self._episodes_added

    @property
    def total_env_steps(self):
        """Total environment steps collected (for replay ratio gating)."""
        with self.lock:
            return self._total_steps

    @property
    def last_online_sample_fraction(self) -> float:
        """Fraction of the most recent sequence batch supplied by online FIFO."""
        with self.lock:
            return self._last_online_sample_fraction

    @property
    def online_sample_fraction(self) -> float:
        """Cumulative fraction of sampled sequences supplied by online FIFO."""
        with self.lock:
            if self._sequence_samples == 0:
                return 0.0
            return self._online_samples / self._sequence_samples

    @property
    def online_queue_size(self) -> int:
        """Number of bounded online descriptors awaiting first consumption."""
        with self.lock:
            return len(self._online_queue)

    @property
    def online_descriptors_dropped(self) -> int:
        """Descriptors discarded only because the bounded FIFO was already full."""
        with self.lock:
            return self._online_descriptors_dropped

    @property
    def replay_chunks_added(self) -> int:
        """Number of transport chunks appended to logical episodes."""
        with self.lock:
            return self._chunks_added

    @property
    def replay_chunk_rows_added(self) -> int:
        """Number of transition rows received through chunk transport."""
        with self.lock:
            return self._chunk_rows_added

    @property
    def preterminal_chunks_added(self) -> int:
        """Chunks that made experience visible before its episode terminated."""
        with self.lock:
            return self._preterminal_chunks_added

    @property
    def active_partial_episodes(self) -> int:
        """Logical episodes currently awaiting a final collector chunk."""
        with self.lock:
            return len(self._active_partial_episodes)

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
