# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from functools import cache, cached_property, partial
from multiprocessing import cpu_count
from typing import Any, Callable, NamedTuple, Self, TypeVar

import numpy as np
import scipy
from jax import hessian, jit, value_and_grad, vmap
from jax import numpy as jnp
from jax.lax import select
from jax.tree_util import Partial
from jaxtyping import Array, Float, Integer
from numpy import typing as npt
from tqdm.auto import tqdm

from ..eig.base import Eigendecomposition
from ..log import logger
from ..pheno import VariableCollection
from ..utils import IterationOrder, make_pool_or_null_context
from .base import NullModelCollection, NullModelResult

terms_count = TypeVar("terms_count")


class OptimizeInput(NamedTuple):
    eigenvalues: Float[Array, " sample_count"]
    rotated_covariates: Float[Array, " sample_count covariate_count"]
    rotated_phenotype: Float[Array, " sample_count 1"]


@dataclass
class OptimizeJob:
    indices: tuple[int, int]

    eig: Eigendecomposition
    vc: VariableCollection


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
    sample_count: int
    covariate_count: int

    minimum_variance: float = 1e-4
    maximum_variance_multiplier: float = 2.0

    grid_search_size: int = 100

    enable_softplus_penalty: bool = True
    softplus_beta: Float[Array, ""] = field(default_factory=partial(jnp.asarray, 10000))

    @classmethod
    @cache
    def create(cls, sample_count: int, covariate_count: int) -> Self:
        return cls(sample_count, covariate_count)

    def get_initial_terms(self, o: OptimizeInput) -> list[float]:
        variance: float = o.rotated_phenotype.var().item()
        return [variance / 2] * 2

    def grid_search(self, o: OptimizeInput) -> npt.NDArray[np.float64]:
        variance: Float[Array, ""] = o.rotated_phenotype.var()

        variance_ratios = jnp.linspace(0.01, 0.99, self.grid_search_size)
        variances = np.linspace(
            self.minimum_variance,
            variance * self.maximum_variance_multiplier,
            self.grid_search_size,
        )
        grid = jnp.meshgrid(variance_ratios, variances)

        combinations = jnp.vstack([m.ravel() for m in grid]).transpose()
        genetic_variance = (1 - combinations[:, 0]) * combinations[:, 1]
        error_variance = combinations[:, 0] * combinations[:, 1]

        terms_grid = jnp.vstack([error_variance, genetic_variance]).transpose()
        wrapper = vmap(Partial(self.minus_two_log_likelihood, o=o))

        minus_two_log_likelihoods = wrapper(jnp.asarray(terms_grid))
        i = jnp.argmin(minus_two_log_likelihoods)

        return np.asarray(terms_grid[i, :])

    def bounds(self, o: OptimizeInput) -> list[tuple[float, float]]:
        variance: float = o.rotated_phenotype.var().item()
        return [
            (self.minimum_variance, variance * self.maximum_variance_multiplier),
            (0, variance * self.maximum_variance_multiplier),
        ]

    @cached_property
    def func_with_grad(
        self,
    ) -> Callable[
        [Float[Array, " terms_count"], OptimizeInput],
        tuple[Float[Array, ""], Float[Array, " terms_count"]],
    ]:
        return jit(value_and_grad(self.minus_two_log_likelihood))

    @cached_property
    def hessian(
        self,
    ) -> Callable[
        [Float[Array, " terms_count"], OptimizeInput],
        tuple[Float[Array, " terms_count terms_count"]],
    ]:
        func = hessian(self.minus_two_log_likelihood)
        return jit(func)

    @staticmethod
    def get_regression_weights(
        terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> RegressionWeights:
        genetic_variance = terms[1]
        error_variance = terms[0]
        variance = (genetic_variance * o.eigenvalues + error_variance)[:, jnp.newaxis]
        inverse_variance = jnp.pow(variance, -0.5)

        scaled_covariates = o.rotated_covariates * inverse_variance
        scaled_phenotype = o.rotated_phenotype * inverse_variance

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

    @classmethod
    def terms_to_tensor(
        cls, numpy_terms: list[float] | npt.NDArray[np.float64]
    ) -> Float[Array, " terms_count"]:
        terms = jnp.asarray(numpy_terms)
        terms = jnp.where(jnp.isfinite(terms), terms, 0.0)
        return terms

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

    @classmethod
    def apply(cls, optimize_job: OptimizeJob) -> tuple[tuple[int, int], NullModelResult]:
        (_, phenotype_index) = optimize_job.indices
        eig = optimize_job.eig
        vc = optimize_job.vc

        eigenvectors = eig.eigenvectors
        covariates = vc.covariates.to_numpy().copy()
        phenotype = vc.phenotypes.to_numpy()[:, phenotype_index, np.newaxis]

        # Subtract column mean from covariates (except intercept).
        covariates[:, 1:] -= covariates[:, 1:].mean(axis=0)

        # Rotate covariates and phenotypes.
        eigenvalues = jnp.asarray(eig.eigenvalues)
        rotated_covariates = jnp.asarray(eigenvectors.transpose() @ covariates)
        rotated_phenotype = jnp.asarray(eigenvectors.transpose() @ phenotype)

        o = OptimizeInput(
            eigenvalues,
            rotated_covariates,
            rotated_phenotype,
        )

        ml = cls.create(vc.sample_count, vc.covariate_count)

        try:
            optimize_result = ml.optimize(o)
            if optimize_result.x.dtype != np.float64:
                raise RuntimeError("Dtype needs to be float64")
            terms = jnp.asarray(optimize_result.x)
            se = ml.get_standard_errors(terms, o)
            minus_two_log_likelihood = float(optimize_result.fun)
            null_model_result = NullModelResult(
                -0.5 * minus_two_log_likelihood,
                *ml.get_heritability(terms),
                np.asarray(se.regression_weights),
                np.asarray(se.standard_errors),
                np.asarray(se.scaled_residuals),
                np.asarray(se.variance),
            )
            return optimize_job.indices, null_model_result
        except Exception as e:
            logger.error(
                "Failed to fit null model for phenotype at index "
                f"{optimize_job.indices}",
                exc_info=e,
            )
            return optimize_job.indices, NullModelResult(
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            )

    @classmethod
    def fit(
        cls,
        eigendecompositions: list[Eigendecomposition],
        variable_collections: list[VariableCollection],
        null_model_collections: list[NullModelCollection],
        num_threads: int = cpu_count(),
    ) -> None:
        # Fit null model for each phenotype
        optimize_jobs = [
            OptimizeJob((collection_index, phenotype_index), eig, vc)
            for collection_index, (eig, vc) in enumerate(
                zip(
                    eigendecompositions,
                    variable_collections,
                    strict=True,
                )
            )
            for phenotype_index in range(vc.phenotype_count)
        ]

        pool, iterator = make_pool_or_null_context(
            optimize_jobs,
            cls.apply,
            num_threads=num_threads,
            iteration_order=IterationOrder.UNORDERED,
        )
        with pool:
            for indices, r in tqdm(
                iterator,
                desc="fitting null models",
                unit="phenotypes",
                total=len(optimize_jobs),
            ):
                collection_index, phenotype_index = indices
                nm = null_model_collections[collection_index]
                nm.put(phenotype_index, r)

    def softplus_penalty(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> Float[Array, "..."]:
        maximum_variance = o.rotated_phenotype.var() * jnp.asarray(
            self.maximum_variance_multiplier
        )
        upper_penalty = softplus(
            terms[:2] - maximum_variance,
            beta=self.softplus_beta,
        )
        lower_penalty = softplus(
            -terms[:2],
            beta=self.softplus_beta,
        )
        penalty = jnp.asarray(self.softplus_beta) * (
            lower_penalty.sum() + upper_penalty.sum()
        )
        return penalty

    def get_minus_two_log_likelihood_terms(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> MinusTwoLogLikelihoodTerms:
        sample_count = jnp.asarray(self.sample_count)
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


def softplus(x: Float[Array, ""], beta: Float[Array, ""]) -> Float[Array, ""]:
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
    assert isinstance(sign, jnp.ndarray)
    assert isinstance(logabsdet, jnp.ndarray)
    logdet = jnp.where(sign == -1.0, jnp.inf, logabsdet)
    return logdet
