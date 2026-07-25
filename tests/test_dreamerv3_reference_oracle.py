import json
from pathlib import Path

import torch

from dreamer.models import (
    calculate_lambda_returns,
    symexp,
    symexp_twohot_bins,
    symlog,
    twohot_encode,
    twohot_expectation,
)
from dreamer.trainer.forward import (
    calculate_replay_lambda_targets,
    calculate_return_normalizer_update,
)


FIXTURE = Path(__file__).parent / "fixtures" / "dreamerv3_e3f0224_oracle.json"
SOURCE_COMMIT = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def tensor(values, *, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype)


def test_fixture_identifies_pinned_source_and_jax_runtime() -> None:
    fixture = load_fixture()
    assert fixture["schema_version"] == 1
    assert fixture["source_commit"] == SOURCE_COMMIT
    assert fixture["jax_version"] == "0.4.33"


def test_transforms_match_independent_jax_fixture() -> None:
    fixture = load_fixture()["transforms"]
    inputs = tensor(fixture["input"])
    torch.testing.assert_close(
        symlog(inputs), tensor(fixture["symlog"]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        symexp(symlog(inputs)),
        tensor(fixture["roundtrip_symexp"]),
        rtol=1e-5,
        atol=1e-3,
    )


def test_twohot_matches_independent_jax_fixture() -> None:
    fixture = load_fixture()["twohot"]
    bins = symexp_twohot_bins(-20, 20, 9)
    torch.testing.assert_close(bins, tensor(fixture["bins"]), rtol=1e-6, atol=1e-4)
    torch.testing.assert_close(
        twohot_encode(tensor(fixture["targets"]), bins),
        tensor(fixture["target_weights"]),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        twohot_expectation(tensor(fixture["logits"]), bins),
        tensor(fixture["predictions"]),
        rtol=1e-5,
        atol=1e-2,
    )


def test_replay_lambda_return_matches_independent_jax_fixture() -> None:
    fixture = load_fixture()["replay_lambda_return"]
    actual = calculate_replay_lambda_targets(
        tensor(fixture["rewards"]).transpose(0, 1),
        tensor(fixture["is_last"], dtype=torch.bool).transpose(0, 1),
        tensor(fixture["is_terminal"], dtype=torch.bool).transpose(0, 1),
        tensor(fixture["values"]).transpose(0, 1),
        fixture["gamma"],
        fixture["lambda"],
    )
    expected = tensor(fixture["returns"]).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_imagination_lambda_return_matches_independent_jax_fixture() -> None:
    fixture = load_fixture()["imagination_lambda_return"]
    rewards = tensor(fixture["rewards"]).transpose(0, 1)
    values = tensor(fixture["values"]).transpose(0, 1)
    continues = tensor(fixture["continues"]).transpose(0, 1)
    actual = calculate_lambda_returns(
        rewards,
        values,
        continues,
        gamma=fixture["gamma"],
        lam=fixture["lambda"],
        num_dream_steps=rewards.shape[0],
        continues_are_logits=False,
    )
    expected = tensor(fixture["returns"]).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_percentile_update_matches_independent_jax_fixture() -> None:
    fixture = load_fixture()["percentiles"]
    values = tensor(fixture["input"]).reshape(2, 4)
    update = calculate_return_normalizer_update(
        [(values, torch.ones(4))], 0.0, 0.0, rate=1.0
    )
    assert update is not None
    scale, lo, hi = update
    assert abs(lo - fixture["p05"]) < 1e-6
    assert abs(hi - fixture["p95"]) < 1e-6
    assert scale == max(1.0, hi - lo)
