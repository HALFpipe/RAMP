from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, Self, TypeAlias, TypeVar

import jax
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float, Integer
from numpy import typing as npt

from ..log import logger
from .base import NullModelResult

sample_count = TypeVar("sample_count")
covariate_count = TypeVar("covariate_count")
terms_count = TypeVar("terms_count")
grid_search_size = TypeVar("grid_search_size")

OptimizeInput: TypeAlias = tuple[
    Float[Array, " sample_count"],  # eigenvalues
    Float[Array, " sample_count covariate_count"],  # rotated_covariates
    Float[Array, " sample_count 1"],  # rotated_phenotype
]


class OptimizeResult(NamedTuple):
    x: npt.NDArray[np.float64]
    fun: float


class RegressionWeights(NamedTuple):
    regression_weights: Float[Array, " covariate_count 1"]
    halfway_scaled_residuals: Float[Array, " sample_count 1"]
    variance: Float[Array, " sample_count 1"]
    inverse_variance: Float[Array, " sample_count 1"]
    scaled_covariates: Float[Array, " sample_count covariate_count"]
    scaled_phenotype: Float[Array, " sample_count 1"]


class MinusTwoLogLikelihoodTerms(NamedTuple):
    sample_count: Integer[Array, ""]
    genetic_variance: Float[Array, ""]
    logarithmic_determinant: Float[Array, ""]
    deviation: Float[Array, ""]
    r: RegressionWeights


class StandardErrors(NamedTuple):
    regression_weights: Float[Array, " covariate_count 1"]
    standard_errors: Float[Array, " covariate_count 1"]
    halfway_scaled_residuals: Float[Array, " sample_count 1"]
    variance: Float[Array, " sample_count 1"]


@dataclass(frozen=True, eq=True, kw_only=True)
class MaximumLikelihoodBase:
    minimum_variance: float = 1e-4
    maximum_variance_multiplier: float = 2.0

    grid_search_size: int = 100

    enable_softplus_penalty: bool = True
    softplus_beta: float = 10000.0

    @classmethod
    def create(cls, **kwargs: Any) -> Self:
        return cls(**kwargs)

    @staticmethod
    def terms_to_tensor(
        numpy_terms: list[float] | npt.NDArray[np.float64],
    ) -> Float[Array, " terms_count"]:
        terms = jnp.asarray(numpy_terms)
        terms = jnp.where(jnp.isfinite(terms), terms, 0.0)
        return terms

    @staticmethod
    def get_initial_terms(o: OptimizeInput) -> list[float]:
        _, _, rotated_phenotype = o
        variance: float = rotated_phenotype.var().item()
        return [variance / 2] * 2

    @abstractmethod
    def minus_two_log_likelihood(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> Float[Array, ""]: ...

    @abstractmethod
    def optimize(
        self,
        o: OptimizeInput,
        method: str = "L-BFGS-B",
        enable_hessian: bool = False,
        disp: bool = False,
    ) -> OptimizeResult: ...

    def get_null_model_result(self, o: OptimizeInput) -> NullModelResult:
        try:
            optimize_result = self.optimize(o)
            if optimize_result.x.dtype != np.float64:
                raise RuntimeError("Dtype needs to be float64")
            terms = jnp.asarray(optimize_result.x)
            se: StandardErrors = self.get_standard_errors(terms, o)
            minus_two_log_likelihood = float(optimize_result.fun)
            heritability, genetic_variance, error_variance = self.get_heritability(terms)
            null_model_result = NullModelResult(
                log_likelihood=-0.5 * minus_two_log_likelihood,
                heritability=heritability,
                genetic_variance=genetic_variance,
                error_variance=error_variance,
                regression_weights=np.asarray(se.regression_weights),
                standard_errors=np.asarray(se.standard_errors),
                halfway_scaled_residuals=np.asarray(se.halfway_scaled_residuals),
                variance=np.asarray(se.variance),
            )
        except Exception as e:
            logger.error("Failed to fit null model", exc_info=e)
            null_model_result = NullModelResult.null()
        return null_model_result

    @staticmethod
    def get_heritability(
        terms: Float[Array, " terms_count"] | npt.NDArray[np.float64],
    ) -> tuple[float, float, float]:
        genetic_variance = float(terms[1])
        error_variance = float(terms[0])
        heritability = float(genetic_variance / (genetic_variance + error_variance))
        return heritability, genetic_variance, error_variance

    @staticmethod
    @jax.jit
    def get_regression_weights(
        terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> RegressionWeights:
        eigenvalues, rotated_covariates, rotated_phenotype = o

        genetic_variance = terms[1]
        error_variance = terms[0]
        variance = (genetic_variance * eigenvalues + error_variance)[:, jnp.newaxis]
        inverse_variance = jnp.pow(variance, -0.5)

        scaled_covariates = rotated_covariates * inverse_variance
        scaled_phenotype = rotated_phenotype * inverse_variance

        regression_weights, _, _, _ = jnp.linalg.lstsq(
            scaled_covariates, scaled_phenotype, rcond=None
        )
        halfway_scaled_residuals = (
            scaled_phenotype - scaled_covariates @ regression_weights
        )

        return RegressionWeights(
            regression_weights=regression_weights,
            halfway_scaled_residuals=halfway_scaled_residuals,
            variance=variance,
            inverse_variance=inverse_variance,
            scaled_covariates=scaled_covariates,
            scaled_phenotype=scaled_phenotype,
        )

    @classmethod
    @partial(jax.jit, static_argnums=0)
    def get_standard_errors(
        cls, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> StandardErrors:
        r: RegressionWeights = cls.get_regression_weights(terms, o)

        degrees_of_freedom = r.scaled_covariates.shape[0] - r.scaled_covariates.shape[1]
        deviation = jnp.square(r.halfway_scaled_residuals).sum()
        residual_variance = deviation / degrees_of_freedom

        inverse_covariance = jnp.linalg.inv(
            r.scaled_covariates.transpose() @ r.scaled_covariates
        )
        standard_errors = residual_variance * jnp.sqrt(jnp.diagonal(inverse_covariance))
        standard_errors = jnp.reshape(standard_errors, (-1, 1))

        return StandardErrors(
            regression_weights=r.regression_weights,
            standard_errors=standard_errors,
            halfway_scaled_residuals=r.halfway_scaled_residuals,
            variance=r.variance,
        )
