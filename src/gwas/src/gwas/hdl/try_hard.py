from typing import Callable, NamedTuple

import jax
from jax import numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Bool, Float64, Integer

from .minimize import minimize
from .minimize import parameter_count as parameter_count


class Carry(NamedTuple):
    i: Integer[Array, ""]
    value: Float64[Array, ""]
    error: Float64[Array, ""]
    params: Float64[Array, " 2"]


def try_hard(
    objective_function: Callable[
        [Float64[Array, " parameter_count"]], Float64[Array, ""]
    ],
    initial_params: Float64[Array, " parameter_count"],
    key: jax.Array,
    lower_bound: Float64[Array, " parameter_count"] | None = None,
    upper_bound: Float64[Array, " parameter_count"] | None = None,
    tolerance: Float64[Array, ""] | None = None,
    maximum_iterations: Integer[Array, ""] | None = None,
    tries: Integer[Array, ""] | None = None,
    loc: Float64[Array, ""] | None = None,
    scale: Float64[Array, ""] | None = None,
) -> tuple[Float64[Array, ""], Float64[Array, " parameter_count"]]:
    if tolerance is None:
        tolerance = jnp.array(1e-8)
    if maximum_iterations is None:
        maximum_iterations = jnp.array(1000, dtype=jnp.int32)
    if lower_bound is None:
        lower_bound = jnp.full_like(initial_params, -jnp.inf)
    if upper_bound is None:
        upper_bound = jnp.full_like(initial_params, jnp.inf)

    # Adapted from https://github.com/OpenMx/OpenMx/blob/master/R/MxTryHard.R#L34-L35
    if tries is None:
        tries = jnp.array(10, dtype=jnp.int32)
    if loc is None:
        loc = jnp.array(1.0)
    if scale is None:
        scale = jnp.array(0.25)

    m = Partial(
        minimize,
        objective_function,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
    )

    def body_fun(carry: Carry) -> Carry:
        i, value, error, params = carry

        slope = jax.random.uniform(
            key, shape=params.shape, minval=1.0 - scale, maxval=1.0 + scale
        )
        intercept = jax.random.uniform(
            key, shape=params.shape, minval=0.0 - scale, maxval=0.0 + scale
        )
        candidate_value, candidate_error, candidate_params = m(
            params * slope + intercept
        )
        candidate_params = jnp.where(
            jnp.isnan(candidate_params), params, candidate_params
        )

        params = jnp.where(candidate_value < value, candidate_params, params)
        error = jnp.where(candidate_value < value, candidate_error, error)
        value = jnp.where(candidate_value < value, candidate_value, value)

        return Carry(i + 1, value, error, params)

    def cond_fun(carry: Carry) -> Bool[Array, ""]:
        i, _, error, _ = carry
        return i < maximum_iterations & (error >= tolerance)

    initial_carry = Carry(jnp.array(0), *m(initial_params))
    final_carry = jax.lax.while_loop(cond_fun, body_fun, initial_carry)
    _, value, _, params = final_carry

    # jax.debug.print("value={value} params={params}", value=value, params=params)

    return value, params
