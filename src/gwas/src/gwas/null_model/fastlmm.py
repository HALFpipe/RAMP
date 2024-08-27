from dataclasses import dataclass
from functools import partial
from typing import override

import jax
import numpy as np
import scipy
from chex import register_dataclass_type_with_jax_tree_util
from jax import numpy as jnp
from jaxtyping import Array, Float
from numpy import typing as npt

from ..log import logger
from .mlb import (
    MinusTwoLogLikelihoodTerms,
    OptimizeInput,
    OptimizeResult,
    StandardErrors,
)
from .pml import ProfileMaximumLikelihood


@dataclass(frozen=True, eq=True)
class FaSTLMM(ProfileMaximumLikelihood):
    step: float = 0.2

    @override
    @partial(jax.jit, static_argnums=0)
    def minus_two_log_likelihood(
        self, terms: Float[Array, " 2"], o: OptimizeInput
    ) -> Float[Array, ""]:
        t: MinusTwoLogLikelihoodTerms = self.get_minus_two_log_likelihood_terms(terms, o)

        variance_ratio = terms[0]
        genetic_variance = t.deviation / t.sample_count
        error_variance = variance_ratio * genetic_variance

        minus_two_log_likelihood = (
            t.sample_count
            + t.logarithmic_determinant
            + t.sample_count * jnp.log(genetic_variance)
        )

        if self.enable_softplus_penalty:
            minus_two_log_likelihood += self.softplus_penalty(
                jnp.stack((error_variance, genetic_variance)), o
            )

        return jnp.where(
            jnp.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            jnp.inf,
        )

    def wrapper(self, numpy_terms: npt.NDArray[np.float64], o: OptimizeInput) -> float:
        variance_ratio = np.power(10, numpy_terms)
        terms: Float[Array, " 2"] = jnp.asarray([variance_ratio, 1])
        return float(np.asarray(self.minus_two_log_likelihood(terms, o)))

    @override
    def optimize(
        self,
        o: OptimizeInput,
        method: str = "",
        enable_hessian: bool = False,
        disp: bool = False,
    ) -> OptimizeResult:
        _, _, rotated_phenotype = o
        upper_bound = float(np.log10(rotated_phenotype.var()))
        lower_bound = float(np.log10(self.minimum_variance) - upper_bound)
        xa = np.arange(lower_bound, upper_bound, step=self.step, dtype=np.float64)
        logger.debug(
            f"FaSTLMM will optimize between {lower_bound} to {upper_bound} "
            f"in {xa.size} steps of size {self.step}"
        )

        fmin: float = np.inf
        best_optimize_result: OptimizeResult | None = None
        with np.errstate(all="ignore"):
            for bounds in zip(xa, xa + self.step, strict=True):
                try:
                    optimize_result: OptimizeResult = scipy.optimize.minimize_scalar(
                        self.wrapper,
                        args=(o,),
                        bounds=bounds,
                        options=dict(disp=disp, xatol=1e-15),
                    )
                    if optimize_result.fun < fmin:
                        fmin = optimize_result.fun
                        best_optimize_result = optimize_result
                except FloatingPointError:
                    pass

        if best_optimize_result is None:
            return OptimizeResult(
                x=np.full((2,), np.nan),
                fun=np.nan,
            )

        log_variance_ratio = best_optimize_result.x
        variance_ratio = np.power(10, log_variance_ratio)

        # Scale by genetic variance
        terms = jnp.asarray([variance_ratio, 1])
        se: StandardErrors = self.get_standard_errors(terms, o)
        genetic_variance = float(np.square(se.halfway_scaled_residuals).mean())
        terms *= genetic_variance

        logger.debug(f"Found FaSTLMM minimum at {fmin}")
        return OptimizeResult(
            x=np.asarray(terms),
            fun=float(fmin),
        )


register_dataclass_type_with_jax_tree_util(FaSTLMM)
