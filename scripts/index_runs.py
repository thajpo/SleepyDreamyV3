#!/usr/bin/env python3
"""Generate a non-destructive registry of local training evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from dreamer.run_registry import generate_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("runs"), Path("experiments")],
        help="Run roots to scan (default: runs experiments)",
    )
    parser.add_argument("--csv", type=Path, default=Path("reports/runs.csv"))
    parser.add_argument(
        "--markdown", type=Path, default=Path("reports/runs.md")
    )
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    records = generate_registry(
        roots=[root.resolve() for root in args.roots],
        repo_root=repo_root,
        csv_path=args.csv,
        markdown_path=args.markdown,
    )
    print(f"Indexed {len(records)} runs into {args.csv} and {args.markdown}")


if __name__ == "__main__":
    main()
