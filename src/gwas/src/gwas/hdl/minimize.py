from typing import Callable, NamedTuple, TypeVar

import jax
import optax
from jaxtyping import Array, Bool, Float64, Integer
from optax import tree_utils as otu

parameter_count = TypeVar("parameter_count")


class Carry(NamedTuple):
    params: Float64[Array, " 2"]
    state: optax.OptState


def minimize(
    objective_function: Callable[
        [Float64[Array, " parameter_count"]], Float64[Array, ""]
    ],
    initial_params: Float64[Array, " parameter_count"],
    lower_bound: Float64[Array, " parameter_count"],
    upper_bound: Float64[Array, " parameter_count"],
    tolerance: Float64[Array, ""],
    maximum_iterations: Integer[Array, ""],
) -> tuple[Float64[Array, ""], Float64[Array, ""], Float64[Array, " parameter_count"]]:
    optimizer = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(objective_function)

    def body_fun(carry: Carry) -> Carry:
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        with jax.debug_nans(False):
            updates, state = optimizer.update(
                grad, state, params, value=value, grad=grad, value_fn=objective_function
            )
        params = optax.apply_updates(params, updates)
        params = optax.projections.projection_box(params, lower_bound, upper_bound)
        return Carry(params, state)

    def cond_fun(carry: Carry) -> Bool[Array, ""]:
        _, state = carry
        i = otu.tree_get(state, "count")
        gradient = otu.tree_get(state, "grad")
        error = otu.tree_l2_norm(gradient)

        return (i == 0) | (i < maximum_iterations) & (error >= tolerance)

    initial_params = optax.projections.projection_box(
        initial_params, lower_bound, upper_bound
    )
    initial_state = optimizer.init(initial_params)
    initial_carry = Carry(initial_params, initial_state)
    final_carry = jax.lax.while_loop(cond_fun, body_fun, initial_carry)

    params, state = final_carry
    value = otu.tree_get(state, "value")
    gradient = otu.tree_get(state, "grad")
    error = otu.tree_l2_norm(gradient)
    return value, error, params
