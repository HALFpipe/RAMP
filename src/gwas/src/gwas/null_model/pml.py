from functools import cached_property
from typing import Any, Callable, NamedTuple, Self, TypeAlias, TypeVar

import jax
import numpy as np
import scipy
from chex import dataclass
from jax import numpy as jnp
from jax.export import deserialize, export
from jax.lax import select
from jaxtyping import Array, Float, Integer
from numpy import typing as npt

from ..eig.base import Eigendecomposition
from ..log import logger
from ..pheno import VariableCollection
from .base import NullModelResult

sample_count = TypeVar("sample_count")
covariate_count = TypeVar("covariate_count")
terms_count = TypeVar("terms_count")

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


@dataclass(frozen=True, eq=True)
class ProfileMaximumLikelihood:
    serialized_func: bytearray | None
    serialized_vec_func: bytearray | None
    serialized_func_with_grad: bytearray | None
    serialized_hessian: bytearray | None

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
    def create(cls, sample_count: int, covariate_count: int, **kwargs) -> Self:
        base = cls(
            serialized_func=None,
            serialized_vec_func=None,
            serialized_func_with_grad=None,
            serialized_hessian=None,
            **kwargs,
        )

        o: OptimizeInput = (
            jnp.zeros((sample_count,)),
            jnp.zeros((sample_count, covariate_count)),
            jnp.zeros((sample_count, 1)),
        )
        terms = base.terms_to_tensor(base.get_initial_terms(o))
        vec_terms = jnp.zeros((base.grid_search_size**2, *terms.shape))

        logger.debug(f"Compiling for {sample_count=} {covariate_count=} {terms.shape=}")

        minus_two_log_likelihood = base.minus_two_log_likelihood.__func__  # type: ignore

        func = jax.jit(minus_two_log_likelihood, static_argnums=0)
        exported_func = export(func)(base, terms, o)

        vmap_func = jax.vmap(minus_two_log_likelihood, in_axes=[None, 0, None])
        vec_func = jax.jit(vmap_func, static_argnums=0)
        exported_vec_func = export(vec_func)(base, vec_terms, o)

        value_and_grad = jax.value_and_grad(minus_two_log_likelihood, argnums=1)
        func_with_grad = jax.jit(value_and_grad, static_argnums=0)
        exported_func_with_grad = export(func_with_grad)(base, terms, o)

        hessian = jax.jit(
            jax.hessian(minus_two_log_likelihood, argnums=1), static_argnums=0
        )
        exported_hessian = export(hessian)(base, terms, o)

        return cls(
            serialized_func=exported_func.serialize(),
            serialized_vec_func=exported_vec_func.serialize(),
            serialized_func_with_grad=exported_func_with_grad.serialize(),
            serialized_hessian=exported_hessian.serialize(),
            **kwargs,
        )

    @cached_property
    def func(
        self,
    ) -> Callable[[Float[Array, " terms_count"], OptimizeInput], Float[Array, ""]]:
        if self.serialized_func is None:
            raise RuntimeError("Model not compiled")
        return deserialize(self.serialized_func).call

    @cached_property
    def vec_func(
        self,
    ) -> Callable[
        [Float[Array, " grid_search_size terms_count"], OptimizeInput],
        Float[Array, " grid_search_size"],
    ]:
        if self.serialized_vec_func is None:
            raise RuntimeError("Model not compiled")
        return deserialize(self.serialized_vec_func).call

    @cached_property
    def func_with_grad(
        self,
    ) -> Callable[
        [Float[Array, " terms_count"], OptimizeInput],
        tuple[Float[Array, ""], Float[Array, " terms_count"]],
    ]:
        if self.serialized_func_with_grad is None:
            raise RuntimeError("Model not compiled")
        return deserialize(self.serialized_func_with_grad).call

    @cached_property
    def hessian(
        self,
    ) -> Callable[
        [Float[Array, " terms_count"], OptimizeInput],
        tuple[Float[Array, " terms_count terms_count"]],
    ]:
        if self.serialized_hessian is None:
            raise RuntimeError("Model not compiled")
        return deserialize(self.serialized_hessian).call

    def grid_search(self, o: OptimizeInput) -> npt.NDArray[np.float64]:
        _, _, rotated_phenotype = o
        variance: Float[Array, ""] = rotated_phenotype.var()

        variance_ratios = jnp.linspace(0.01, 0.99, self.grid_search_size)
        variances = np.linspace(
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

        minus_two_log_likelihoods = self.vec_func(vec_terms, o)
        i = jnp.argmin(minus_two_log_likelihoods)

        return np.asarray(vec_terms[i, :])

    def bounds(self, o: OptimizeInput) -> list[tuple[float, float]]:
        _, _, rotated_phenotype = o
        variance: float = rotated_phenotype.var().item()
        return [
            (self.minimum_variance, variance * self.maximum_variance_multiplier),
            (0, variance * self.maximum_variance_multiplier),
        ]

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

    def wrapper_with_grad(
        self, numpy_terms: npt.NDArray[np.float64], o: OptimizeInput
    ) -> tuple[float, npt.NDArray[np.float64]]:
        try:
            terms = self.terms_to_tensor(numpy_terms)
            value, grad = self.func_with_grad(terms, o)
            return value.item(), np.asarray(grad)
        except RuntimeError:
            return np.nan, np.full_like(numpy_terms, np.nan)

    def hessian_wrapper(
        self, numpy_terms: npt.NDArray[np.float64], o: OptimizeInput
    ) -> npt.NDArray[np.float64]:
        terms = self.terms_to_tensor(numpy_terms)
        hess = self.hessian(terms, o)
        return np.asarray(hess)

    @staticmethod
    def get_heritability(
        terms: Float[Array, " terms_count"] | npt.NDArray[np.float64],
    ) -> tuple[float, float, float]:
        genetic_variance = float(terms[1])
        error_variance = float(terms[0])
        heritability = float(genetic_variance / (genetic_variance + error_variance))
        return heritability, genetic_variance, error_variance

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
        optimize_result = scipy.optimize.basinhopping(
            self.wrapper_with_grad,
            init,
            minimizer_kwargs=minimizer_kwargs,
            stepsize=float(init.mean()) / 8,
            niter=2**10,
            niter_success=2**4,
            disp=disp,
        )

        return OptimizeResult(x=optimize_result.x, fun=optimize_result.fun)

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


def logdet(a: Float[Array, " n n"]) -> Float[Array, ""]:
    """A re-implementation of torch.logdet that returns infinity instead of NaN, which
    prevents an error in autodiff.

    Args:
        a (Float[Array, "..."]): _description_

    Returns:
        _type_: _description_
    """
    sign, logabsdet = jnp.linalg.slogdet(a)
    logdet = jnp.where(sign == -1.0, jnp.inf, logabsdet)
    return logdet
