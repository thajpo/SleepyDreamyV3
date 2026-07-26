from copy import deepcopy

import pytest

from scripts.verify_dreamerv3_contracts import (
    CONTRACT_DIR,
    expected_gradient_updates,
    load_contract,
    validate_all,
    validate_contract,
)


def test_frozen_contract_totals_are_internally_consistent():
    results = {row["name"]: row for row in validate_all()}

    assert results["paper_v2_atari100k"]["raw_environment_frames"] == 400_000
    assert results["paper_v2_atari100k"]["expected_gradient_updates"] == 12_500
    assert results["official_e3f0224_atari100k"]["raw_environment_frames"] == 440_000
    assert results["official_e3f0224_atari100k"]["expected_gradient_updates"] == 27_500
    assert results["cartpole_drift_v1"]["expected_gradient_updates"] == 3_500
    assert (
        results["cartpole_reference_v3_state_v1"]["expected_gradient_updates"]
        == 3_500
    )


def test_paper_and_current_source_contracts_remain_distinct():
    paper = load_contract(CONTRACT_DIR / "paper_v2_atari100k.yaml")
    source = load_contract(CONTRACT_DIR / "official_e3f0224_atari100k.yaml")

    assert paper["training"]["replay_ratio"] == 128.0
    assert source["training"]["replay_ratio"] == 256.0
    assert paper["training"]["beta2"] == 0.99
    assert source["training"]["beta2"] == 0.999
    assert paper["purpose"] != source["purpose"]


def test_cartpole_contract_exposes_world_model_accounting_mismatch():
    contract = load_contract(CONTRACT_DIR / "cartpole_drift_v1.yaml")
    training = contract["training"]

    world_model_ratio = (
        expected_gradient_updates(contract)
        * training["batch_size"]
        * training["world_model_rows_per_update"]
        / contract["environment"]["agent_decisions"]
    )

    assert world_model_ratio == pytest.approx(16.0 * 16 / 12)
    assert world_model_ratio == pytest.approx(
        training["effective_world_model_replay_ratio"]
    )


def test_reference_cartpole_contract_aligns_all_trained_row_counts():
    contract = load_contract(
        CONTRACT_DIR / "cartpole_reference_v3_state_v1.yaml"
    )
    training = contract["training"]

    assert training["sampled_sequence_length"] - training["replay_context_rows"] == 12
    assert training["trained_rows_per_update"] == 8 * 12
    assert training["world_model_rows_per_update"] == 8 * 12
    assert training["actor_value_starts_per_update"] == 8 * 12


def test_reference_cartpole_contract_preserves_qualified_architecture():
    contract = load_contract(
        CONTRACT_DIR / "cartpole_reference_v3_state_v1.yaml"
    )

    assert contract["implementation"]["commit"] == "326d0b2"
    assert contract["model"]["trainable_parameters_without_slow_value"] == 639_173
    assert contract["model"]["rmsnorm_learned_shift"] is True


def test_contract_validation_fails_closed_on_unit_drift():
    contract = load_contract(CONTRACT_DIR / "paper_v2_atari100k.yaml")
    broken = deepcopy(contract)
    broken["environment"]["raw_environment_frames"] += 1

    with pytest.raises(ValueError, match="raw frames"):
        validate_contract(broken)
