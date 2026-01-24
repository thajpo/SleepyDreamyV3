"""
Episode Replay Buffer for Dreamer training.

Stores complete episodes, samples fixed-length subsequences.
This matches DreamerV3's approach for consistent batch shapes.
"""

import threading
import random
import numpy as np
from collections import deque
from queue import Empty


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

    def __init__(self, data_queue, max_episodes=1000, min_episodes=64, sequence_length=25):
        """
        Args:
            data_queue: mp.Queue to drain episodes from
            max_episodes: Maximum episodes to store (older episodes evicted)
            min_episodes: Block sampling until buffer has this many episodes
            sequence_length: Fixed length of sampled subsequences
        """
        self.data_queue = data_queue
        self.max_episodes = max_episodes
        self.min_episodes = min_episodes
        self.sequence_length = sequence_length

        self.buffer = deque(maxlen=max_episodes)
        self.lock = threading.Lock()
        self.ready_event = threading.Event()  # Signals when min_episodes reached

        self._stop = False
        self._thread = None
        self._episodes_added = 0
        self._total_steps = 0  # For tracking average episode length
        self._recent_ep_lengths = deque(maxlen=100)  # Track recent episode lengths
        self._max_sequence_length = 256  # Cap for adaptive growth

    def start(self):
        """Start background queue draining thread."""
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background thread gracefully."""
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _drain_loop(self):
        """Background thread: continuously move episodes from queue to buffer."""
        while not self._stop:
            try:
                # Short timeout so we can check stop flag regularly
                episode = self.data_queue.get(timeout=0.1)
                with self.lock:
                    self.buffer.append(episode)
                    self._episodes_added += 1
                    # Episode length is now passed as 6th element, fallback to states length
                    ep_len = episode[5] if len(episode) > 5 else len(episode[1])
                    self._total_steps += ep_len
                    self._recent_ep_lengths.append(ep_len)
                    if len(self.buffer) >= self.min_episodes and not self.ready_event.is_set():
                        self.ready_event.set()
                        print(f"Replay buffer ready: {len(self.buffer)} episodes collected", flush=True)
            except Empty:
                continue

    def _sample_subsequence(self, episode):
        """
        Sample a fixed-length subsequence from an episode.

        If episode is shorter than sequence_length, pads with zeros and
        marks as terminated.
        """
        # Episode tuple: (pixels, states, actions, rewards, terminated, episode_length)
        # episode_length is for tracking only, not needed for subsequence sampling
        pixels, states, actions, rewards, terminated = episode[:5]
        # Use states for length - pixels may be None in state-only mode
        ep_len = len(states)
        seq_len = self.sequence_length

        if ep_len >= seq_len:
            # Sample random start point
            start = random.randint(0, ep_len - seq_len)
            return (
                pixels[start:start + seq_len] if pixels is not None else None,
                states[start:start + seq_len],
                actions[start:start + seq_len],
                rewards[start:start + seq_len],
                terminated[start:start + seq_len],
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
            terminated_pad = np.ones(pad_len, dtype=terminated.dtype)  # Mark padded as terminated

            return (
                pixels_out,
                np.concatenate([states, states_pad], axis=0),
                np.concatenate([actions, actions_pad], axis=0),
                np.concatenate([rewards, rewards_pad], axis=0),
                np.concatenate([terminated, terminated_pad], axis=0),
            )

    def sample(self, batch_size):
        """
        Sample a batch of fixed-length subsequences.

        Blocks until buffer has min_episodes, then returns instantly.

        Args:
            batch_size: Number of subsequences to sample

        Returns:
            List of (pixels, states, actions, rewards, terminated) tuples,
            each with shape (sequence_length, ...)
        """
        # Wait for buffer to have enough episodes (only blocks on startup)
        self.ready_event.wait()

        with self.lock:
            # Sample episodes with replacement if needed
            if len(self.buffer) < batch_size:
                indices = random.choices(range(len(self.buffer)), k=batch_size)
            else:
                indices = random.sample(range(len(self.buffer)), k=batch_size)

            # Extract fixed-length subsequences
            return [self._sample_subsequence(self.buffer[i]) for i in indices]

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
                self.sequence_length + increment,
                self._max_sequence_length
            )
            if self.sequence_length > old_len:
                print(f"Increased sequence_length: {old_len} -> {self.sequence_length} "
                      f"(avg_ep_len={avg_len:.1f})", flush=True)
        return self.sequence_length
