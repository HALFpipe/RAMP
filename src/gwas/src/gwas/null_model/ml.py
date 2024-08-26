from dataclasses import dataclass, field
from typing import Self, override

import numpy as np
from chex import register_dataclass_type_with_jax_tree_util
from jax import hessian
from jax import numpy as jnp
from jaxtyping import Array, Float
from numpy import typing as npt

from .mlb import (
    OptimizeInput,
    RegressionWeights,
    StandardErrors,
)
from .mlb import terms_count as terms_count
from .pml import ProfileMaximumLikelihood


@dataclass(frozen=True, eq=True)
class MaximumLikelihood(ProfileMaximumLikelihood):
    pml: ProfileMaximumLikelihood = field(kw_only=True)

    @override
    @classmethod
    def create(cls, sample_count: int, covariate_count: int, **kwargs) -> Self:
        pml = ProfileMaximumLikelihood.create(sample_count, covariate_count, **kwargs)
        return super().create(sample_count, covariate_count, pml=pml, **kwargs)

    @override
    @classmethod
    def get_initial_terms(cls, o: OptimizeInput) -> list[float]:
        terms = super().get_initial_terms(o)
        r = super().get_regression_weights(cls.terms_to_tensor(terms), o)
        regression_weights = list(np.asarray(r.regression_weights).ravel())
        return terms + regression_weights

    @override
    def grid_search(self, o: OptimizeInput) -> npt.NDArray[np.float64]:
        pml = self.pml

        terms = pml.grid_search(o)
        r = pml.get_regression_weights(self.terms_to_tensor(terms), o)
        regression_weights = np.asarray(r.regression_weights).ravel()
        return np.hstack([terms, regression_weights])

    @override
    def bounds(self, o: OptimizeInput) -> list[tuple[float, float]]:
        _, rotated_covariates, _ = o
        _, covariate_count = rotated_covariates.shape
        return self.pml.bounds(o) + [(-np.inf, np.inf)] * covariate_count

    @override
    @staticmethod
    def get_regression_weights(
        terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> RegressionWeights:
        eigenvalues, rotated_covariates, rotated_phenotype = o
        terms = jnp.where(jnp.isfinite(terms), terms, 0)

        variance = terms[1] * eigenvalues + terms[0]
        inverse_variance = jnp.pow(variance, -0.5)[:, jnp.newaxis]

        scaled_covariates = rotated_covariates * inverse_variance
        scaled_phenotype = rotated_phenotype * inverse_variance

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

    @override
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


register_dataclass_type_with_jax_tree_util(MaximumLikelihood)
