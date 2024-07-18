# -*- coding: utf-8 -*-
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, TypeVar

import numpy as np
import scipy
from jax import jit
from jax import numpy as jnp
from jaxtyping import Array, Float
from numpy import typing as npt

from ..log import logger
from .pml import OptimizeInput, OptimizeResult, ProfileMaximumLikelihood

terms_count = TypeVar("terms_count")


@dataclass(frozen=True, eq=True)
class FaSTLMM(ProfileMaximumLikelihood):
    step: float = 0.2

    def minus_two_log_likelihood(
        self, terms: Float[Array, " 2"], o: OptimizeInput
    ) -> Float[Array, ""]:
        t = self.get_minus_two_log_likelihood_terms(terms, o)

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

    @cached_property
    def func(
        self,
    ) -> Callable[[Float[Array, " terms_count"], OptimizeInput], Float[Array, ""]]:
        return jit(self.minus_two_log_likelihood)

    def wrapper(self, numpy_terms: npt.NDArray[np.float64], o: OptimizeInput) -> float:
        variance_ratio = np.power(10, numpy_terms)
        terms = jnp.asarray([variance_ratio, 1])
        return float(np.asarray(self.func(terms, o)))

    def optimize(
        self,
        o: OptimizeInput,
        method: str = "",
        enable_hessian: bool = False,
        disp: bool = False,
    ) -> OptimizeResult:
        func = self.wrapper

        upper_bound = float(np.log10(o.rotated_phenotype.var()))
        lower_bound = float(np.log10(self.minimum_variance) - upper_bound)
        logger.debug(
            f"FaSTLMM will optimize between {lower_bound} to {upper_bound} "
            f"in steps of size {self.step}"
        )
        xa = np.arange(lower_bound, upper_bound, step=self.step, dtype=np.float64)
        logger.debug(f"FaSTLMM will optimize in {xa.size} steps")

        fmin: float = np.inf
        best_x: float | None = None
        with np.errstate(all="ignore"):
            for bounds in zip(xa, xa + self.step, strict=True):
                a, c = bounds
                try:
                    x, fval, _, _ = scipy.optimize.brent(
                        func, args=(o,), brack=(a, c), tol=1e-15, full_output=True
                    )
                    if fval < fmin:
                        fmin = fval
                        best_x = x
                except (FloatingPointError, ValueError) as exception:
                    logger.debug(f"FaSTLMM failed at {bounds}", exc_info=exception)
                    pass

        if best_x is None:
            logger.warning("Could not find FaSTLMM minimum")
            return OptimizeResult(
                x=np.full((2,), np.nan),
                fun=np.nan,
            )

        log_variance_ratio = best_x
        variance_ratio = np.power(10, log_variance_ratio)

        # Scale by genetic variance
        terms = jnp.asarray([variance_ratio, 1])
        se = self.get_standard_errors(terms, o)
        genetic_variance = float(np.square(se.scaled_residuals).mean())
        terms *= genetic_variance

        logger.debug(f"Found FaSTLMM minimum at {fmin}")
        return OptimizeResult(
            x=np.asarray(terms),
            fun=float(fmin),
        )


@dataclass(frozen=True, eq=True)
class PenalizedFaSTLMM(FaSTLMM):
    def __post_init__(self) -> None:
        logger.warning("PenalizedFaSTLMM doesn't work")

    def minus_two_log_likelihood(
        self, terms: Float[Array, " 2"], o: OptimizeInput
    ) -> Float[Array, ""]:
        t = self.get_minus_two_log_likelihood_terms(terms, o)

        variance_ratio = terms[0]
        genetic_variance = t.deviation / (t.sample_count - 4)
        error_variance = variance_ratio * genetic_variance

        penalty = -2 * jnp.log(variance_ratio)

        minus_two_log_likelihood = (
            (t.sample_count - 4)
            + t.logarithmic_determinant
            + t.sample_count * jnp.log(genetic_variance)
            + penalty
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
