#!/usr/bin/env python3
"""
Parallel sweep runner that bypasses Hydra's joblib launcher.

Spawns independent training processes to avoid nested multiprocessing conflicts.

Usage:
    # Run the ac_params sweep with 2 parallel jobs
    uv run python scripts/parallel_sweep.py --sweep ac_params --jobs 2

    # Custom parameter sweep
    uv run python scripts/parallel_sweep.py --params "train.actor_lr=1e-5,3e-5" "train.actor_entropy_coef=1e-3,5e-3" --jobs 4

    # Dry run to see what would be executed
    uv run python scripts/parallel_sweep.py --sweep ac_params --dry-run
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


# Predefined sweeps (mirror conf/sweep/*.yaml)
SWEEPS = {
    "ac_params": {
        "train.actor_lr": ["1e-5", "3e-5", "1e-4"],
        "train.actor_entropy_coef": ["1e-3", "5e-3", "1e-2", "3e-2"],
    },
}


def parse_param_string(param: str) -> tuple[str, list[str]]:
    """Parse 'key=val1,val2,val3' into (key, [val1, val2, val3])."""
    key, values = param.split("=", 1)
    return key, values.split(",")


def generate_configs(params: dict[str, list[str]]) -> list[dict[str, str]]:
    """Generate all combinations of parameter values."""
    keys = list(params.keys())
    value_lists = [params[k] for k in keys]

    configs = []
    for combo in itertools.product(*value_lists):
        configs.append(dict(zip(keys, combo)))
    return configs


def run_training(config: dict[str, str], base_args: list[str], output_dir: Path, job_id: int) -> tuple[int, dict, int]:
    """Run a single training job. Returns (job_id, config, return_code)."""
    # Build command
    cmd = ["uv", "run", "python", "-m", "src.train"]
    cmd.extend(base_args)

    # Add config overrides
    for key, value in config.items():
        cmd.append(f"{key}={value}")

    # Create unique output directory for this run
    # Use underscores instead of = to avoid Hydra parsing issues
    config_str = ",".join(f"{k}={v}" for k, v in sorted(config.items()))
    dir_name = "_".join(f"{k.split('.')[-1]}-{v}" for k, v in sorted(config.items()))
    run_dir = output_dir / dir_name
    cmd.append(f"++hydra.run.dir={run_dir}")

    print(f"[Job {job_id}] Starting: {config_str}")

    # Run the training
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[Job {job_id}] FAILED: {config_str}")
        print(f"  stderr: {result.stderr[-500:] if result.stderr else 'none'}")
    else:
        print(f"[Job {job_id}] Completed: {config_str}")

    return job_id, config, result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run parallel hyperparameter sweeps")
    parser.add_argument("--sweep", type=str, choices=list(SWEEPS.keys()),
                        help="Predefined sweep name (from conf/sweep/)")
    parser.add_argument("--params", nargs="+", type=str,
                        help="Custom params: 'key=val1,val2' 'key2=a,b,c'")
    parser.add_argument("--jobs", "-j", type=int, default=2,
                        help="Number of parallel jobs (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs without running")
    parser.add_argument("--base-args", nargs="*", default=[],
                        help="Additional args passed to all runs")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory (default: runs/sweeps/MM-DD_HHMM)")

    args = parser.parse_args()

    # Build parameter dict
    if args.sweep:
        params = SWEEPS[args.sweep]
    elif args.params:
        params = {}
        for p in args.params:
            key, values = parse_param_string(p)
            params[key] = values
    else:
        parser.error("Must specify --sweep or --params")

    # Generate all configs
    configs = generate_configs(params)
    print(f"Generated {len(configs)} configurations from {len(params)} parameters")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%m-%d_%H%M")
        output_dir = Path("runs/sweeps") / timestamp

    if args.dry_run:
        print(f"\nOutput directory: {output_dir}")
        print(f"Would run {len(configs)} jobs with {args.jobs} parallel workers:\n")
        for i, config in enumerate(configs):
            config_str = " ".join(f"{k}={v}" for k, v in config.items())
            print(f"  [{i+1}] {config_str}")
        return 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Running {len(configs)} jobs with {args.jobs} parallel workers\n")

    # Run in parallel
    failed = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_training, config, args.base_args, output_dir, i): i
            for i, config in enumerate(configs)
        }

        for future in as_completed(futures):
            job_id, config, returncode = future.result()
            if returncode != 0:
                failed.append((job_id, config))

    # Summary
    print(f"\n{'='*50}")
    print(f"Completed: {len(configs) - len(failed)}/{len(configs)} jobs")
    if failed:
        print(f"Failed jobs:")
        for job_id, config in failed:
            print(f"  [{job_id}] {config}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
