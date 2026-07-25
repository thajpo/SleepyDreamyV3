#!/usr/bin/env python3
"""Validate frozen DreamerV3 research contracts and their accounting units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


CONTRACT_DIR = Path(__file__).resolve().parents[1] / "reports" / "contracts"
CONTRACT_NAMES = (
    "paper_v2_atari100k",
    "official_e3f0224_atari100k",
    "cartpole_drift_v1",
)


def expected_gradient_updates(contract: dict) -> float:
    """Return optimizer updates implied by decisions, ratio, and trained rows."""
    environment = contract["environment"]
    training = contract["training"]
    return (
        float(environment["agent_decisions"])
        * float(training["replay_ratio"])
        / (
            int(training["batch_size"])
            * int(training["trained_batch_length"])
        )
    )


def expected_raw_frames(contract: dict) -> int:
    """Return raw environment frames implied by decisions and action repeat."""
    environment = contract["environment"]
    return int(environment["agent_decisions"]) * int(environment["action_repeat"])


def validate_contract(contract: dict) -> dict:
    """Fail closed when authored totals disagree with their declared units."""
    if int(contract["contract_version"]) != 1:
        raise ValueError("unsupported contract_version")
    frames = expected_raw_frames(contract)
    authored_frames = int(contract["environment"]["raw_environment_frames"])
    if frames != authored_frames:
        raise ValueError(
            f"{contract['name']}: raw frames {authored_frames} != implied {frames}"
        )
    updates = expected_gradient_updates(contract)
    authored_updates = float(contract["training"]["expected_gradient_updates"])
    if abs(updates - authored_updates) > 1e-9:
        raise ValueError(
            f"{contract['name']}: updates {authored_updates} != implied {updates}"
        )
    return {
        "name": str(contract["name"]),
        "agent_decisions": int(contract["environment"]["agent_decisions"]),
        "raw_environment_frames": frames,
        "expected_gradient_updates": updates,
    }


def load_contract(path: Path) -> dict:
    config = OmegaConf.load(path)
    return dict(OmegaConf.to_container(config, resolve=True))


def validate_all(contract_dir: Path = CONTRACT_DIR) -> list[dict]:
    results = []
    for name in CONTRACT_NAMES:
        results.append(validate_contract(load_contract(contract_dir / f"{name}.yaml")))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=CONTRACT_DIR)
    args = parser.parse_args()
    print(json.dumps(validate_all(args.contract_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
