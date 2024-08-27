from dataclasses import dataclass, field
from functools import partial
from typing import Any, Self, override

import jax
import numpy as np
from chex import register_dataclass_type_with_jax_tree_util
from jax import numpy as jnp
from jaxtyping import Array, Float

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
    def create(cls, **kwargs: Any) -> Self:
        pml = ProfileMaximumLikelihood(**kwargs)
        return super().create(pml=pml, **kwargs)

    @override
    @classmethod
    def get_initial_terms(cls, o: OptimizeInput) -> list[float]:
        terms = super().get_initial_terms(o)
        r: RegressionWeights = super().get_regression_weights(
            cls.terms_to_tensor(terms), o
        )
        regression_weights = list(np.asarray(r.regression_weights).ravel())
        return terms + regression_weights

    @override
    @partial(jax.jit, static_argnums=0)
    def grid_search(self, o: OptimizeInput) -> Float[Array, "..."]:
        pml = self.pml
        terms = pml.grid_search(o)
        r: RegressionWeights = pml.get_regression_weights(terms, o)
        return jnp.hstack([terms, r.regression_weights.ravel()])

    @override
    def bounds(self, o: OptimizeInput) -> list[tuple[float, float]]:
        _, rotated_covariates, _ = o
        _, covariate_count = rotated_covariates.shape
        return self.pml.bounds(o) + [(-np.inf, np.inf)] * covariate_count

    @override
    @staticmethod
    @jax.jit
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

    @override
    @partial(jax.jit, static_argnums=0)
    def get_standard_errors(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> StandardErrors:
        r: RegressionWeights = self.get_regression_weights(terms, o)

        covariance = self.hessian(terms, o)
        inverse_covariance = jnp.linalg.inv(covariance)
        standard_errors = jnp.sqrt(jnp.diagonal(inverse_covariance))
        standard_errors = standard_errors[2:]
        standard_errors = jnp.reshape(standard_errors, (-1, 1))

        return StandardErrors(
            regression_weights=r.regression_weights,
            standard_errors=standard_errors,
            halfway_scaled_residuals=r.halfway_scaled_residuals,
            variance=r.variance,
        )


register_dataclass_type_with_jax_tree_util(MaximumLikelihood)
