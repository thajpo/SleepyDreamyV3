# pyright: reportMissingImports=false
"""Generate deterministic numerical fixtures from pinned DreamerV3 equations.

This script intentionally has no imports from the local ``dreamer`` package.
Run it in a temporary JAX environment so the committed fixture remains an
independent cross-framework oracle:

    uv run --python 3.12 --with 'jax[cpu]==0.4.33' \
      python scripts/reference/generate_dreamerv3_e3f0224_oracle.py

The equations are transcribed from danijar/dreamerv3 at commit
e3f02248693a79dc8b0ebd62c93683888ddaccfe:

- dreamerv3/agent.py:482-490 (lambda_return)
- embodied/jax/nets.py:59-64 and 361-405 (symlog, symexp, RMSNorm)
- embodied/jax/heads.py:132-143 and embodied/jax/outs.py:273-321
  (symmetric bins and TwoHot)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp


SOURCE_COMMIT = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"


def symlog(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def lambda_return(
    last: jax.Array,
    term: jax.Array,
    rew: jax.Array,
    val: jax.Array,
    boot: jax.Array,
    disc: float,
    lam: float,
) -> jax.Array:
    del val  # Present in the pinned signature and used for its shape assertion.
    rets = [boot[:, -1]]
    live = (1 - term.astype(jnp.float32))[:, 1:] * disc
    cont = (1 - last.astype(jnp.float32))[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return jnp.stack(list(reversed(rets))[:-1], 1)


def symmetric_bins(count: int) -> jax.Array:
    if count % 2 == 1:
        half = jnp.linspace(-20, 0, (count - 1) // 2 + 1, dtype=jnp.float32)
        half = symexp(half)
        return jnp.concatenate([half, -half[:-1][::-1]], 0)
    half = jnp.linspace(-20, 0, count // 2, dtype=jnp.float32)
    half = symexp(half)
    return jnp.concatenate([half, -half[::-1]], 0)


def twohot_target(target: jax.Array, bins: jax.Array) -> jax.Array:
    below = (bins <= target[..., None]).astype(jnp.int32).sum(-1) - 1
    above = len(bins) - (bins > target[..., None]).astype(jnp.int32).sum(-1)
    below = jnp.clip(below, 0, len(bins) - 1)
    above = jnp.clip(above, 0, len(bins) - 1)
    equal = below == above
    dist_below = jnp.where(equal, 1, jnp.abs(bins[below] - target))
    dist_above = jnp.where(equal, 1, jnp.abs(bins[above] - target))
    total = dist_below + dist_above
    weight_below = dist_above / total
    weight_above = dist_below / total
    return (
        jax.nn.one_hot(below, len(bins)) * weight_below[..., None]
        + jax.nn.one_hot(above, len(bins)) * weight_above[..., None]
    )


def twohot_prediction(logits: jax.Array, bins: jax.Array) -> jax.Array:
    probs = jax.nn.softmax(logits)
    n = logits.shape[-1]
    if n % 2 == 1:
        m = (n - 1) // 2
        center = (probs[..., m : m + 1] * bins[m : m + 1]).sum(-1)
        paired = (
            (probs[..., :m] * bins[:m])[..., ::-1]
            + probs[..., m + 1 :] * bins[m + 1 :]
        ).sum(-1)
        return center + paired
    return (
        (probs[..., : n // 2] * bins[: n // 2])[..., ::-1]
        + probs[..., n // 2 :] * bins[n // 2 :]
    ).sum(-1)


def rms_norm(
    x: jax.Array, scale: jax.Array, shift: jax.Array, eps: float = 1e-4
) -> jax.Array:
    mean2 = jnp.square(x.astype(jnp.float32)).mean(-1, keepdims=True)
    return x * (jax.lax.rsqrt(mean2 + eps) * scale) + shift


def as_list(x: jax.Array) -> list:
    return jax.device_get(x).tolist()


def build_fixture() -> dict:
    transform_input = jnp.array(
        [-1e6, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 1e6],
        dtype=jnp.float32,
    )

    norm_input = jnp.array(
        [[-3.0, -1.0, 0.0, 2.0, 5.0], [0.1, 0.2, 0.3, 0.4, 0.5]],
        dtype=jnp.float32,
    )
    norm_scale = jnp.array([0.75, 1.0, 1.25, 1.5, 2.0], dtype=jnp.float32)
    norm_shift = jnp.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=jnp.float32)

    bins = symmetric_bins(9)
    targets = jnp.array(
        [-1e9, -100.0, -1.5, 0.0, 0.25, 12.0, 1e9], dtype=jnp.float32
    )
    logits = jnp.array(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0, 1.5, 0.5, -0.5, -1.5],
            [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            [3.0, -2.0, 1.0, -1.0, 0.5, -0.5, 2.0, -3.0, 1.5],
        ],
        dtype=jnp.float32,
    )

    # B=2, T=6. Row zero is replay context; the returned targets align with
    # rows 0..4 and consume rewards/termination flags from rows 1..5.
    rewards = jnp.array(
        [[0.0, 1.0, -0.5, 2.0, 0.25, -1.0], [0.0, 0.2, 0.4, 0.8, 1.6, 3.2]],
        dtype=jnp.float32,
    )
    values = jnp.array(
        [[0.1, 0.3, 0.7, 1.1, 1.5, 1.9], [2.0, 1.5, 1.0, 0.5, 0.0, -0.5]],
        dtype=jnp.float32,
    )
    is_last = jnp.array(
        [[False, False, False, True, False, False], [False] * 6]
    )
    is_terminal = jnp.array(
        [[False, False, False, True, False, False], [False, False, False, False, False, True]]
    )
    replay_returns = lambda_return(
        is_last, is_terminal, rewards, values, values, 0.997, 0.95
    )

    imagine_rewards = rewards[:, 1:]
    # Local Dreamer exposes H rewards for H imagined successors and H+1 values
    # including the starting latent. This is equivalent to the pinned source's
    # common H+1 layout after lambda_return drops index zero from rewards/boot.
    imagine_values = values
    imagine_continues = jnp.array(
        [[0.99, 0.85, 0.40, 0.95, 0.10], [0.60, 0.70, 0.80, 0.90, 1.0]],
        dtype=jnp.float32,
    )
    imagine_last = jnp.zeros((2, 6), dtype=bool)
    imagine_term = jnp.concatenate(
        [jnp.zeros((2, 1), jnp.float32), 1.0 - imagine_continues], axis=1
    )
    imagine_returns = lambda_return(
        imagine_last,
        imagine_term,
        rewards,
        values,
        values,
        1.0,
        0.95,
    )

    percentile_input = jnp.array(
        [-4.0, -1.0, 0.0, 0.5, 1.0, 2.0, 8.0, 20.0], dtype=jnp.float32
    )

    return {
        "schema_version": 1,
        "source_commit": SOURCE_COMMIT,
        "jax_version": jax.__version__,
        "transforms": {
            "input": as_list(transform_input),
            "symlog": as_list(symlog(transform_input)),
            "roundtrip_symexp": as_list(symexp(symlog(transform_input))),
        },
        "rms_norm": {
            "eps": 1e-4,
            "input": as_list(norm_input),
            "scale": as_list(norm_scale),
            "shift": as_list(norm_shift),
            "output": as_list(rms_norm(norm_input, norm_scale, norm_shift)),
        },
        "twohot": {
            "bins": as_list(bins),
            "targets": as_list(targets),
            "target_weights": as_list(twohot_target(targets, bins)),
            "logits": as_list(logits),
            "predictions": as_list(twohot_prediction(logits, bins)),
        },
        "replay_lambda_return": {
            "gamma": 0.997,
            "lambda": 0.95,
            "rewards": as_list(rewards),
            "values": as_list(values),
            "is_last": as_list(is_last),
            "is_terminal": as_list(is_terminal),
            "returns": as_list(replay_returns),
        },
        "imagination_lambda_return": {
            "gamma": 1.0,
            "lambda": 0.95,
            "rewards": as_list(imagine_rewards),
            "values": as_list(imagine_values),
            "continues": as_list(imagine_continues),
            "returns": as_list(imagine_returns),
        },
        "percentiles": {
            "input": as_list(percentile_input),
            "p05": float(jnp.percentile(percentile_input, 5.0)),
            "p95": float(jnp.percentile(percentile_input, 95.0)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/dreamerv3_e3f0224_oracle.json"),
    )
    args = parser.parse_args()
    fixture = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {args.output} from {SOURCE_COMMIT} with JAX {jax.__version__}")


if __name__ == "__main__":
    main()
