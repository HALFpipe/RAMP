from dataclasses import dataclass

import numpy as np
from jax import hessian
from jax import numpy as jnp
from jaxtyping import Array, Float
from numpy import typing as npt

from .pml import (
    OptimizeInput,
    ProfileMaximumLikelihood,
    RegressionWeights,
    StandardErrors,
)
from .pml import terms_count as terms_count


@dataclass(frozen=True, eq=True)
class MaximumLikelihood(ProfileMaximumLikelihood):
    def get_initial_terms(self, o: OptimizeInput) -> list[float]:
        terms = super().get_initial_terms(o)
        r = super().get_regression_weights(
            self.terms_to_tensor(terms),
            o,
        )
        regression_weights = list(np.asarray(r.regression_weights).ravel())
        return terms + regression_weights

    def grid_search(self, o: OptimizeInput) -> npt.NDArray[np.float64]:
        pml = ProfileMaximumLikelihood(**vars(self))
        terms = pml.grid_search(o)
        r = pml.get_regression_weights(
            self.terms_to_tensor(terms),
            o,
        )
        regression_weights = np.asarray(r.regression_weights).ravel()
        return np.hstack([terms, regression_weights])

    def bounds(self, o: OptimizeInput) -> list[tuple[float, float]]:
        return super().bounds(o) + [(-np.inf, np.inf)] * self.covariate_count

    @staticmethod
    def get_regression_weights(
        terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> RegressionWeights:
        terms = jnp.where(
            jnp.isfinite(terms),
            terms,
            0,
        )

        variance = terms[1] * o.eigenvalues + terms[0]
        inverse_variance = jnp.pow(variance, -0.5)[:, jnp.newaxis]

        scaled_covariates = o.rotated_covariates * inverse_variance
        scaled_phenotype = o.rotated_phenotype * inverse_variance

        regression_weights = terms[2:]
        regression_weights = jnp.reshape(regression_weights, (-1, 1))
        scaled_residuals = scaled_phenotype - scaled_covariates @ regression_weights
        return RegressionWeights(
            regression_weights=regression_weights,
            scaled_residuals=scaled_residuals,
            variance=variance,
            inverse_variance=inverse_variance,
            scaled_covariates=scaled_covariates,
            scaled_phenotype=scaled_phenotype,
        )

    @classmethod
    def get_standard_errors(
        cls, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> StandardErrors:
        r = cls.get_regression_weights(terms, o)

        covariance = hessian(cls.minus_two_log_likelihood)(terms, o)
        inverse_covariance = jnp.linalg.inv(covariance)
        standard_errors = jnp.sqrt(jnp.diagonal(inverse_covariance))
        standard_errors = standard_errors[2:]
        standard_errors = jnp.reshape(standard_errors, (-1, 1))

        return StandardErrors(
            regression_weights=r.regression_weights,
            standard_errors=standard_errors,
            scaled_residuals=r.scaled_residuals,
            variance=r.variance,
        )
