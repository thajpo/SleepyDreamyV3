#!/usr/bin/env python3
"""Measure resource use and interference for concurrent CartPole training runs.

The probe launches the normal ``dreamer-train`` entry point in dry-run mode, so
it exercises collectors, replay, model updates, and shutdown without retaining
MLflow runs or checkpoints. It records host and process-tree memory, CPU load,
and aggregate AMD GPU utilization/VRAM use in a small JSON evidence file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil


THROUGHPUT_RE = re.compile(r"Step (\d+)/(\d+) \| ([0-9.]+) steps/s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def gpu_snapshot() -> dict[str, int] | None:
    """Return aggregate card-0 activity, or None when ROCm SMI is unavailable."""
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi is None:
        return None
    result = subprocess.run(
        [rocm_smi, "--showmeminfo", "vram", "--showuse", "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return None
    try:
        card = json.loads(result.stdout)["card0"]
        return {
            "utilization_percent": int(card["GPU use (%)"]),
            "vram_total_bytes": int(card["VRAM Total Memory (B)"]),
            "vram_used_bytes": int(card["VRAM Total Used Memory (B)"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def process_tree_stats(pid: int) -> dict[str, int]:
    """Aggregate a launcher and all of its currently live descendants."""
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {"processes": 0, "threads": 0, "rss_bytes": 0, "uss_bytes": 0}

    rss = 0
    uss = 0
    threads = 0
    live = 0
    for process in processes:
        try:
            with process.oneshot():
                rss += process.memory_info().rss
                uss += process.memory_full_info().uss
                threads += process.num_threads()
                live += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return {
        "processes": live,
        "threads": threads,
        "rss_bytes": rss,
        "uss_bytes": uss,
    }


def training_command(seed: int, steps: int) -> list[str]:
    """Build the fixed, full-actor CartPole capacity workload."""
    return [
        sys.executable,
        "-m",
        "dreamer.main",
        "env=cartpole_state_only",
        "general.dry_run=true",
        "general.device=cuda",
        f"general.seed={seed}",
        "models.d_hidden=128",
        f"train.max_train_steps={steps}",
        "train.batch_size=8",
        "train.sequence_length=16",
        "train.replay_burn_in=8",
        "train.replay_buffer_size=1000",
        "train.min_buffer_episodes=16",
        "train.num_collectors=1",
        "train.actor_warmup_steps=0",
        "train.replay_ratio=1.0",
        "train.eval_every=0",
    ]


def terminate_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def summarize(
    samples: list[dict], baseline: dict, run_records: list[dict]
) -> dict:
    gpu_samples = [sample["gpu"] for sample in samples if sample["gpu"]]
    cpu_samples = [sample["host_cpu_percent"] for sample in samples]
    min_available = min(sample["host_available_ram_bytes"] for sample in samples)
    summary = {
        "sample_count": len(samples),
        "wall_seconds_max": max(record["wall_seconds"] for record in run_records),
        "host_cpu_percent_mean": statistics.fmean(cpu_samples),
        "host_cpu_percent_peak": max(cpu_samples),
        "host_ram_delta_peak_bytes": max(
            0, baseline["host_available_ram_bytes"] - min_available
        ),
        "run_tree_rss_peak_bytes": [],
        "run_tree_uss_peak_bytes": [],
        "run_process_count_peak": [],
        "run_thread_count_peak": [],
    }
    for index in range(len(run_records)):
        trees = [sample["runs"][index] for sample in samples]
        summary["run_tree_rss_peak_bytes"].append(
            max(tree["rss_bytes"] for tree in trees)
        )
        summary["run_tree_uss_peak_bytes"].append(
            max(tree["uss_bytes"] for tree in trees)
        )
        summary["run_process_count_peak"].append(
            max(tree["processes"] for tree in trees)
        )
        summary["run_thread_count_peak"].append(
            max(tree["threads"] for tree in trees)
        )
    if gpu_samples and baseline["gpu"]:
        summary.update(
            {
                "gpu_utilization_percent_mean": statistics.fmean(
                    sample["utilization_percent"] for sample in gpu_samples
                ),
                "gpu_utilization_percent_peak": max(
                    sample["utilization_percent"] for sample in gpu_samples
                ),
                "gpu_vram_delta_peak_bytes": max(
                    0,
                    max(sample["vram_used_bytes"] for sample in gpu_samples)
                    - baseline["gpu"]["vram_used_bytes"],
                ),
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    if args.concurrency < 1 or args.steps < 1:
        raise SystemExit("concurrency and steps must both be positive")
    if args.torch_threads < 1:
        raise SystemExit("torch-threads must be positive")
    if args.sample_interval <= 0 or args.timeout <= 0:
        raise SystemExit("sample-interval and timeout must both be positive")

    baseline = {
        "host_available_ram_bytes": psutil.virtual_memory().available,
        "gpu": gpu_snapshot(),
    }
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(args.torch_threads),
            "MKL_NUM_THREADS": str(args.torch_threads),
            "OPENBLAS_NUM_THREADS": str(args.torch_threads),
            "NUMEXPR_NUM_THREADS": str(args.torch_threads),
            "PYTHONUNBUFFERED": "1",
        }
    )

    started_at = time.time()
    processes: list[subprocess.Popen[str]] = []
    log_handles = []
    log_paths: list[Path] = []
    samples: list[dict] = []
    psutil.cpu_percent(interval=None)

    with tempfile.TemporaryDirectory(prefix="dreamer_capacity_") as temp_dir:
        try:
            for index in range(args.concurrency):
                log_path = Path(temp_dir) / f"run_{index}.log"
                log_handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    training_command(seed=index, steps=args.steps),
                    cwd=Path.cwd(),
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                processes.append(process)
                log_handles.append(log_handle)
                log_paths.append(log_path)

            while any(process.poll() is None for process in processes):
                elapsed = time.time() - started_at
                if elapsed > args.timeout:
                    raise TimeoutError(f"profile exceeded {args.timeout:.0f}s timeout")
                samples.append(
                    {
                        "elapsed_seconds": elapsed,
                        "host_cpu_percent": psutil.cpu_percent(interval=None),
                        "host_available_ram_bytes": psutil.virtual_memory().available,
                        "gpu": gpu_snapshot(),
                        "runs": [
                            process_tree_stats(process.pid) for process in processes
                        ],
                    }
                )
                time.sleep(args.sample_interval)
        except BaseException:
            for process in processes:
                terminate_group(process)
            raise
        finally:
            for process in processes:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    terminate_group(process)
                    process.wait(timeout=15)
            for log_handle in log_handles:
                log_handle.close()

        finished_at = time.time()
        run_records = []
        for index, (process, log_path) in enumerate(zip(processes, log_paths)):
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            throughput = [
                {"step": int(step), "steps_per_second": float(rate)}
                for step, _, rate in THROUGHPUT_RE.findall(log_text)
            ]
            run_records.append(
                {
                    "index": index,
                    "seed": index,
                    "exit_code": process.returncode,
                    "wall_seconds": finished_at - started_at,
                    "throughput": throughput,
                    "log_tail": log_text.splitlines()[-20:],
                }
            )

    evidence = {
        "schema_version": 1,
        "recorded_at_unix": started_at,
        "command": {
            "concurrency": args.concurrency,
            "steps": args.steps,
            "torch_threads": args.torch_threads,
            "sample_interval": args.sample_interval,
            "training_argv": training_command(seed=0, steps=args.steps),
        },
        "host": {
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "total_ram_bytes": psutil.virtual_memory().total,
            "python": sys.version,
        },
        "baseline": baseline,
        "runs": run_records,
        "summary": summarize(samples, baseline, run_records),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence["summary"], indent=2))
    return 0 if all(record["exit_code"] == 0 for record in run_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
