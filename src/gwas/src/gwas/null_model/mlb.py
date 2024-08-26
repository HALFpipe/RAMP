from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Self, TypeAlias, TypeVar

import jax
import numpy as np
from chex import set_n_cpu_devices
from jax import numpy as jnp
from jaxtyping import Array, Float, Integer
from numpy import typing as npt

from ..eig.base import Eigendecomposition
from ..log import logger
from ..pheno import VariableCollection
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


def setup_jax() -> None:
    set_n_cpu_devices(1)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_traceback_filtering", "off")


class OptimizeResult(NamedTuple):
    x: npt.NDArray[np.float64]
    fun: float


class RegressionWeights(NamedTuple):
    regression_weights: Float[Array, " covariate_count 1"]
    scaled_residuals: Float[Array, " sample_count 1"]
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
    scaled_residuals: Float[Array, " sample_count 1"]
    variance: Float[Array, " sample_count 1"]


@dataclass(frozen=True, eq=True, kw_only=True)
class MaximumLikelihoodBase:
    func: (
        Callable[[Float[Array, " terms_count"], OptimizeInput], Float[Array, ""]] | None
    ) = None
    vec_func: (
        Callable[
            [Float[Array, " grid_search_size terms_count"], OptimizeInput],
            Float[Array, " grid_search_size"],
        ]
        | None
    ) = None
    func_with_grad: (
        Callable[
            [Float[Array, " terms_count"], OptimizeInput],
            tuple[Float[Array, ""], Float[Array, " terms_count"]],
        ]
        | None
    ) = None
    hessian: (
        Callable[
            [Float[Array, " terms_count"], OptimizeInput],
            tuple[Float[Array, " terms_count terms_count"]],
        ]
        | None
    ) = None

    minimum_variance: float = 1e-4
    maximum_variance_multiplier: float = 2.0

    grid_search_size: int = 100

    enable_softplus_penalty: bool = True
    softplus_beta: float = 10000.0

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

    @classmethod
    def create(cls, sample_count: int, covariate_count: int, **kwargs: Any) -> Self:
        base = cls(**kwargs)

        o: OptimizeInput = (
            jnp.zeros((sample_count,)),
            jnp.zeros((sample_count, covariate_count)),
            jnp.zeros((sample_count, 1)),
        )
        terms = base.terms_to_tensor(base.get_initial_terms(o))
        vec_terms = jnp.zeros((base.grid_search_size**2, *terms.shape))

        logger.debug(f"Compiling for {sample_count=} {covariate_count=} {terms.shape=}")

        minus_two_log_likelihood = base.minus_two_log_likelihood.__func__  # type: ignore

        jit_func = jax.jit(minus_two_log_likelihood, static_argnums=0)
        func = jit_func.lower(base, terms, o).compile()

        vmap_func = jax.vmap(minus_two_log_likelihood, in_axes=[None, 0, None])
        jit_vec_func = jax.jit(vmap_func, static_argnums=0)
        vec_func = jit_vec_func.lower(base, vec_terms, o).compile()

        value_and_grad = jax.value_and_grad(minus_two_log_likelihood, argnums=1)
        jit_func_with_grad = jax.jit(value_and_grad, static_argnums=0)
        func_with_grad = jit_func_with_grad.lower(base, terms, o).compile()

        jit_hessian = jax.jit(
            jax.hessian(minus_two_log_likelihood, argnums=1), static_argnums=0
        )
        hessian = jit_hessian.lower(base, terms, o).compile()

        kwargs.update(
            func=func, vec_func=vec_func, func_with_grad=func_with_grad, hessian=hessian
        )
        return cls(**kwargs)

    @abstractmethod
    def optimize(
        self,
        o: OptimizeInput,
        method: str = "L-BFGS-B",
        enable_hessian: bool = False,
        disp: bool = False,
    ) -> OptimizeResult: ...

    def get_null_model_result(
        self, vc: VariableCollection, phenotype_index: int, eig: Eigendecomposition
    ) -> NullModelResult:
        eigenvectors = eig.eigenvectors
        covariates = vc.covariates.copy()
        phenotype = vc.phenotypes[:, phenotype_index, np.newaxis]

        # Subtract column mean from covariates (except intercept).
        covariates[:, 1:] -= covariates[:, 1:].mean(axis=0)

        # Rotate covariates and phenotypes.
        eigenvalues = jnp.asarray(eig.eigenvalues)
        rotated_covariates = jnp.asarray(eigenvectors.transpose() @ covariates)
        rotated_phenotype = jnp.asarray(eigenvectors.transpose() @ phenotype)

        o: OptimizeInput = (eigenvalues, rotated_covariates, rotated_phenotype)

        try:
            optimize_result = self.optimize(o)
            if optimize_result.x.dtype != np.float64:
                raise RuntimeError("Dtype needs to be float64")
            terms = jnp.asarray(optimize_result.x)
            se = self.get_standard_errors(terms, o)
            minus_two_log_likelihood = float(optimize_result.fun)
            null_model_result = NullModelResult(
                -0.5 * minus_two_log_likelihood,
                *self.get_heritability(terms),
                np.asarray(se.regression_weights),
                np.asarray(se.standard_errors),
                np.asarray(se.scaled_residuals),
                np.asarray(se.variance),
            )
        except Exception as e:
            logger.error(
                "Failed to fit null model for phenotype at index " f"{phenotype_index}",
                exc_info=e,
            )
            null_model_result = NullModelResult(
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            )
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

        degrees_of_freedom = r.scaled_covariates.shape[0] - r.scaled_covariates.shape[1]
        residual_variance = jnp.square(r.scaled_residuals).sum() / degrees_of_freedom

        inverse_covariance = jnp.linalg.inv(
            r.scaled_covariates.transpose() @ r.scaled_covariates
        )
        standard_errors = residual_variance * jnp.sqrt(jnp.diagonal(inverse_covariance))
        standard_errors = jnp.reshape(standard_errors, (-1, 1))

        return StandardErrors(
            regression_weights=r.regression_weights,
            standard_errors=standard_errors,
            scaled_residuals=r.scaled_residuals,
            variance=r.variance,
        )
