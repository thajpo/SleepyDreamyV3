"""Unified profiling utilities for DreamerV3 training and data collection."""

import os
import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)


class ProfilerManager:
    """
    Unified PyTorch profiler for training and collection.

    Wraps PyTorch's torch.profiler.profile() to trace CPU/CUDA operations
    and save to TensorBoard-compatible format.

    Example usage:
        with ProfilerManager(enabled=True, log_dir="runs/exp1", component_name="trainer") as profiler:
            for step in range(num_steps):
                # ... training code ...
                profiler.step()
    """

    def __init__(self, enabled: bool, log_dir: str, component_name: str, chunk_steps: int = 200):
        """
        Initialize the profiler manager.

        Args:
            enabled: Whether profiling is enabled
            log_dir: Base directory for logs (profiler traces go to log_dir/profiler/component_name)
            component_name: Name of the component (e.g., "trainer", "collector")
            chunk_steps: Number of steps between trace dumps (default: 200)
        """
        self.enabled = enabled
        self.log_dir = log_dir
        self.component_name = component_name
        self.chunk_steps = chunk_steps
        self.profiler = None
        self.profile_dir = None

    def __enter__(self):
        """Start profiling context."""
        if not self.enabled:
            return self

        self.profile_dir = os.path.join(self.log_dir, "profiler", self.component_name)
        os.makedirs(self.profile_dir, exist_ok=True)

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        trace_handler = tensorboard_trace_handler(self.profile_dir)
        self.profiler = profile(
            activities=activities,
            schedule=schedule(
                wait=0, warmup=0, active=self.chunk_steps, repeat=0
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        self.profiler.__enter__()
        print(
            f"Profiler enabled for {self.component_name}; saving a trace every "
            f"{self.chunk_steps} steps. Traces: {self.profile_dir}"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup profiler."""
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)
        return False

    def step(self):
        """Called each training/collection step."""
        if self.profiler is not None:
            self.profiler.step()


class TimingAccumulator:
    """
    Coarse timing accumulator for data/forward/backward phases.

    Provides simple percentage breakdown of where time is spent
    without the overhead of full profiling.

    Example usage:
        timing = TimingAccumulator(print_interval=50)

        t0 = time.perf_counter()
        # ... data loading ...
        torch.cuda.synchronize()
        timing.log_phase("data", time.perf_counter() - t0)

        t0 = time.perf_counter()
        # ... forward pass ...
        torch.cuda.synchronize()
        timing.log_phase("forward", time.perf_counter() - t0)

        t0 = time.perf_counter()
        # ... backward pass ...
        torch.cuda.synchronize()
        timing.log_phase("backward", time.perf_counter() - t0)

        timing.maybe_print(step)
    """

    def __init__(self, print_interval: int = 50):
        """
        Initialize the timing accumulator.

        Args:
            print_interval: Number of steps between printing timing summary
        """
        self.print_interval = print_interval
        self.t_data = 0.0
        self.t_forward = 0.0
        self.t_backward = 0.0
        self._step_count = 0

    def log_phase(self, phase: str, elapsed: float):
        """
        Accumulate timing for a phase.

        Args:
            phase: Phase name ("data", "forward", or "backward")
            elapsed: Elapsed time in seconds
        """
        if phase == "data":
            self.t_data += elapsed
        elif phase == "forward":
            self.t_forward += elapsed
        elif phase == "backward":
            self.t_backward += elapsed

    def maybe_print(self, step: int = 0) -> bool:
        """
        Print timing summary if at print interval.

        Args:
            step: Current step number (unused, for API compatibility)

        Returns:
            True if summary was printed, False otherwise
        """
        _ = step  # Unused, kept for API compatibility
        self._step_count += 1
        if self._step_count >= self.print_interval:
            self.print_summary()
            return True
        return False

    def print_summary(self):
        """Print percentage breakdown and reset accumulators."""
        total_t = self.t_data + self.t_forward + self.t_backward
        if total_t > 0:
            print(
                f"[PROFILE] data: {self.t_data / total_t * 100:5.1f}% | "
                f"fwd: {self.t_forward / total_t * 100:5.1f}% | "
                f"bwd: {self.t_backward / total_t * 100:5.1f}% | "
                f"total: {total_t:.2f}s/{self._step_count} steps"
            )
        self.reset()

    def reset(self):
        """Reset all accumulators."""
        self.t_data = 0.0
        self.t_forward = 0.0
        self.t_backward = 0.0
        self._step_count = 0
