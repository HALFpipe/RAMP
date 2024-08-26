from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import numpy as np
import scipy
from chex import register_dataclass_type_with_jax_tree_util
from jax import numpy as jnp
from jax.lax import select
from jaxtyping import Array, Float
from numpy import typing as npt

from .mlb import (
    MaximumLikelihoodBase,
    MinusTwoLogLikelihoodTerms,
    OptimizeInput,
    OptimizeResult,
)
from .mlb import terms_count as terms_count

threshold = jnp.asarray(20.0)


def softplus(x: Float[Array, ""], beta: Float[Array, ""] | None) -> Float[Array, ""]:
    if beta is None:
        beta = jnp.asarray(1.0)
    # Taken from https://github.com/google/jax/issues/18443 and
    # mirroring the pytorch implementation at
    # https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
    x_safe = select(x * beta < threshold, x, jnp.ones_like(x))
    return select(
        x * beta < threshold,
        1 / beta * jnp.log(1 + jnp.exp(beta * x_safe)),
        x,
    )


@dataclass(frozen=True, eq=True, kw_only=True)
class ProfileMaximumLikelihood(MaximumLikelihoodBase):
    @partial(jax.jit, static_argnums=0)
    def grid_search(self, o: OptimizeInput) -> Float[Array, "..."]:
        _, _, rotated_phenotype = o
        variance: Float[Array, ""] = rotated_phenotype.var()

        variance_ratios = jnp.linspace(0.01, 0.99, self.grid_search_size)
        variances = jnp.linspace(
            self.minimum_variance,
            variance * self.maximum_variance_multiplier,
            self.grid_search_size,
        )
        grid = jnp.meshgrid(variance_ratios, variances)
        grid_variance_ratios, grid_variances = map(jnp.ravel, grid)

        genetic_variance = (1 - grid_variance_ratios) * grid_variances
        error_variance = grid_variance_ratios * grid_variances

        vec_terms: Float[Array, " grid_search_size 2"] = jnp.vstack(
            [error_variance, genetic_variance]
        ).transpose()

        if self.vec_func is None:
            raise RuntimeError("vec_func is not compiled")
        minus_two_log_likelihoods = self.vec_func(vec_terms, o)
        i = jnp.argmin(minus_two_log_likelihoods)

        return vec_terms[i, :]

    def bounds(self, o: OptimizeInput) -> list[tuple[float, float]]:
        _, _, rotated_phenotype = o
        variance: float = rotated_phenotype.var().item()
        return [
            (self.minimum_variance, variance * self.maximum_variance_multiplier),
            (0, variance * self.maximum_variance_multiplier),
        ]

    def hessian_wrapper(
        self, numpy_terms: npt.NDArray[np.float64], o: OptimizeInput
    ) -> npt.NDArray[np.float64]:
        if self.hessian is None:
            raise RuntimeError("hessian is not compiled")
        terms = self.terms_to_tensor(numpy_terms)
        hess = self.hessian(terms, o)
        return np.asarray(hess)

    def optimize(
        self,
        o: OptimizeInput,
        method: str = "L-BFGS-B",
        enable_hessian: bool = False,
        disp: bool = False,
    ) -> OptimizeResult:
        init = self.grid_search(o)
        bounds = self.bounds(o)

        minimizer_kwargs: dict[str, Any] = dict(
            method=method,
            jac=True,
            bounds=bounds,
            args=(o,),
            # options=dict(disp=disp),
        )
        if enable_hessian:
            minimizer_kwargs.update(dict(hess=self.hessian_wrapper))

        if self.func_with_grad is None:
            raise RuntimeError("func_with_grad is not compiled")
        func_with_grad = self.func_with_grad

        def wrapper_with_grad(
            numpy_terms: npt.NDArray[np.float64], o: OptimizeInput
        ) -> tuple[float, npt.NDArray[np.float64]]:
            try:
                terms = self.terms_to_tensor(numpy_terms)
                value, grad = func_with_grad(terms, o)
                return value.item(), np.asarray(grad)
            except RuntimeError:
                return np.nan, np.full_like(numpy_terms, np.nan)

        optimize_result = scipy.optimize.basinhopping(
            wrapper_with_grad,
            init,
            minimizer_kwargs=minimizer_kwargs,
            stepsize=float(init.mean()) / 8,
            niter=2**10,
            niter_success=2**4,
            disp=disp,
        )

        return OptimizeResult(x=optimize_result.x, fun=optimize_result.fun)

    @partial(jax.jit, static_argnums=0)
    def softplus_penalty(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> Float[Array, "..."]:
        _, _, rotated_phenotype = o
        maximum_variance = rotated_phenotype.var() * jnp.asarray(
            self.maximum_variance_multiplier
        )
        beta = jnp.asarray(self.softplus_beta)
        upper_penalty = softplus(terms[:2] - maximum_variance, beta=beta)
        lower_penalty = softplus(-terms[:2], beta=beta)
        penalty = jnp.asarray(self.softplus_beta) * (
            lower_penalty.sum() + upper_penalty.sum()
        )
        return penalty

    @partial(jax.jit, static_argnums=0)
    def get_minus_two_log_likelihood_terms(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> MinusTwoLogLikelihoodTerms:
        _, _, rotated_phenotype = o
        sample_count = jnp.asarray(rotated_phenotype.size)
        genetic_variance = terms[1]
        r = self.get_regression_weights(terms, o)

        logarithmic_determinant = jnp.log(r.variance).sum()
        deviation = jnp.square(r.scaled_residuals).sum()

        return MinusTwoLogLikelihoodTerms(
            sample_count=sample_count,
            genetic_variance=genetic_variance,
            logarithmic_determinant=logarithmic_determinant,
            deviation=deviation,
            r=r,
        )

    @partial(jax.jit, static_argnums=0)
    def minus_two_log_likelihood(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> Float[Array, ""]:
        t = self.get_minus_two_log_likelihood_terms(terms, o)

        minus_two_log_likelihood = t.logarithmic_determinant + t.deviation

        if self.enable_softplus_penalty:
            minus_two_log_likelihood += self.softplus_penalty(terms, o)

        return jnp.where(
            jnp.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            jnp.inf,
        )


register_dataclass_type_with_jax_tree_util(ProfileMaximumLikelihood)
