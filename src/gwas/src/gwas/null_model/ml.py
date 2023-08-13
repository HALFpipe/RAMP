# -*- coding: utf-8 -*-
from dataclasses import dataclass
from functools import cached_property, partial
from multiprocessing import cpu_count
from typing import Any, ClassVar, NamedTuple

import numpy as np
import scipy
import torch
from functorch import grad_and_value, hessian
from functorch.compile import memory_efficient_fusion
from numpy import typing as npt
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from ..eig import Eigendecomposition
from ..log import logger
from ..pheno import VariableCollection
from ..utils import Pool
from .base import NullModelCollection, NullModelResult

minimum_variance: float = 1e-4
maximum_variance_multiplier: float = 4.0


class OptimizeInput(NamedTuple):
    eigenvalues: torch.Tensor
    rotated_covariates: torch.Tensor
    rotated_phenotype: torch.Tensor


class OptimizeJob(NamedTuple):
    phenotype_index: int
    num_nested_threads: int
    optimize_input: OptimizeInput


class OptimizeResult(NamedTuple):
    x: npt.NDArray
    fun: float


class RegressionWeights(NamedTuple):
    regression_weights: torch.Tensor
    scaled_residuals: torch.Tensor
    variance: torch.Tensor
    inverse_variance: torch.Tensor
    scaled_covariates: torch.Tensor
    scaled_phenotype: torch.Tensor


class MinusTwoLogLikelihoodTerms(NamedTuple):
    sample_count: torch.Tensor
    genetic_variance: torch.Tensor
    logarithmic_determinant: torch.Tensor
    r: RegressionWeights


@dataclass
class ProfileMaximumLikelihood:
    sample_count: int
    covariate_count: int

    enable_softplus_penalty: bool = True

    requires_grad: ClassVar[bool] = True

    def get_initial_terms(self, o: OptimizeInput):
        variance: float = o.rotated_phenotype.var().item()
        return [variance / 2] * 2

    def bounds(self, o: OptimizeInput) -> list[tuple[float, float]]:
        variance: float = o.rotated_phenotype.var().item()
        return [
            (minimum_variance, variance * maximum_variance_multiplier),
            (0, variance * maximum_variance_multiplier),
        ]

    @cached_property
    def func(self):
        o = OptimizeInput(
            torch.rand(self.sample_count),
            torch.rand(self.sample_count, self.covariate_count),
            torch.rand(self.sample_count, 1),
        )
        terms = self.terms_to_tensor(self.get_initial_terms(o))

        func = self.minus_two_log_likelihood
        if self.requires_grad:
            func = grad_and_value(func)

        func = memory_efficient_fusion(func)

        # perform warm-up iterations
        for _ in range(3):
            func(terms, o)

        return func

    @cached_property
    def hessian(self):
        o = OptimizeInput(
            torch.rand(self.sample_count),
            torch.rand(self.sample_count, self.covariate_count),
            torch.rand(self.sample_count, 1),
        )
        terms = self.terms_to_tensor(self.get_initial_terms(o))

        func = hessian(self.minus_two_log_likelihood)
        func = memory_efficient_fusion(func)

        # perform warm-up iterations
        for _ in range(3):
            func(terms, o)

        return func

    @staticmethod
    def get_regression_weights(
        terms: torch.Tensor, o: OptimizeInput
    ) -> RegressionWeights:
        (eigenvalues, rotated_covariates, rotated_phenotype) = o

        genetic_variance = terms[1]
        error_variance = terms[0]
        variance = (genetic_variance * eigenvalues + error_variance)[:, np.newaxis]
        inverse_variance = torch.pow(variance, -0.5)

        scaled_covariates = rotated_covariates * inverse_variance
        scaled_phenotype = rotated_phenotype * inverse_variance

        regression_weights = torch.linalg.lstsq(
            scaled_covariates, scaled_phenotype, rcond=None
        ).solution

        scaled_residuals = torch.ravel(
            scaled_phenotype - scaled_covariates @ regression_weights
        )

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
        cls, terms: torch.Tensor, o: OptimizeInput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r = cls.get_regression_weights(terms, o)

        degrees_of_freedom = r.scaled_covariates.shape[0] - r.scaled_covariates.shape[1]
        residual_variance = torch.square(r.scaled_residuals).sum() / degrees_of_freedom

        inverse_covariance = torch.linalg.inv(
            r.scaled_covariates.t() @ r.scaled_covariates
        )
        standard_errors = residual_variance * torch.sqrt(
            torch.diagonal(inverse_covariance)
        )

        return r.regression_weights, standard_errors, r.scaled_residuals, r.variance

    @classmethod
    def terms_to_tensor(cls, numpy_terms):
        terms = torch.tensor(
            numpy_terms,
            dtype=torch.float64,
            requires_grad=cls.requires_grad,
        )
        terms = torch.where(torch.isfinite(terms), terms, 0.0)
        return terms

    def wrapper(self, numpy_terms: npt.NDArray, o: OptimizeInput):
        try:
            terms = self.terms_to_tensor(numpy_terms)
            grad, value = self.func(terms, o)
            return value.item(), grad.detach().numpy()
        except RuntimeError:
            return np.nan, np.full_like(numpy_terms, np.nan)

    def hessian_wrapper(self, numpy_terms: npt.NDArray, o: OptimizeInput):
        terms = self.terms_to_tensor(numpy_terms)
        hess = self.hessian(terms, o)
        return hess.detach().numpy()

    @staticmethod
    def get_heritability(terms) -> tuple[float, float, float]:
        genetic_variance = float(terms[1])
        error_variance = float(terms[0])
        heritability = float(genetic_variance / (genetic_variance + error_variance))
        return heritability, genetic_variance, error_variance

    def optimize(
        self,
        o: OptimizeInput,
        method: str = "TNC",
        enable_hessian: bool = False,
        disp: bool = False,
        **kwargs,
    ) -> OptimizeResult:
        init = self.get_initial_terms(o)
        bounds = self.bounds(o)

        minimizer_kwargs: dict[str, Any] = dict(
            method=method,
            jac=True,
            bounds=bounds,
            args=(o,),
            options=dict(disp=disp),
        )
        if enable_hessian:
            minimizer_kwargs.update(dict(hess=self.hessian_wrapper))
        optimize_result = scipy.optimize.basinhopping(  # type: ignore
            self.wrapper,
            init,
            minimizer_kwargs=minimizer_kwargs,
            stepsize=init[0] / 10,
            niter=2**10,
            niter_success=2**4,
            disp=disp,
            **kwargs,
        )

        return optimize_result

    def apply(self, optimize_job: OptimizeJob, **kwargs) -> tuple[int, NullModelResult]:
        (phenotype_index, num_nested_threads, o) = optimize_job
        with threadpool_limits(limits=num_nested_threads):
            optimize_result = self.optimize(o, **kwargs)
            terms = torch.tensor(optimize_result.x)
            weights, errors, residuals, variance = self.get_standard_errors(
                terms,
                o,
            )
            minus_two_log_likelihood = float(optimize_result.fun)
            return phenotype_index, NullModelResult(
                -0.5 * minus_two_log_likelihood,
                *self.get_heritability(terms),
                weights.detach().numpy(),
                errors.detach().numpy(),
                residuals.detach().numpy(),
                variance.detach().numpy(),
            )

    @classmethod
    def fit(
        cls,
        eig: Eigendecomposition,
        vc: VariableCollection,
        nm: NullModelCollection,
        num_threads: int = cpu_count(),
        **kwargs,
    ) -> None:
        eigenvectors = eig.eigenvectors
        covariates = vc.covariates.to_numpy().copy()
        phenotypes = vc.phenotypes.to_numpy()

        # Subtract column mean from covariates (except intercept).
        covariates[:, 1:] -= covariates[:, 1:].mean(axis=0)

        # Rotate covariates and phenotypes.
        eigenvalues = torch.tensor(eig.eigenvalues)
        rotated_covariates = torch.tensor(eigenvectors.transpose() @ covariates)
        rotated_phenotypes = torch.tensor(eigenvectors.transpose() @ phenotypes)

        ml = cls(vc.sample_count, vc.covariate_count)

        # Fit null model for each phenotype.
        num_processes = min(num_threads, vc.phenotype_count)
        num_nested_threads = num_threads // num_processes
        optimize_jobs = (
            OptimizeJob(
                phenotype_index,
                num_nested_threads,
                OptimizeInput(
                    eigenvalues,
                    rotated_covariates,
                    rotated_phenotypes[:, phenotype_index, np.newaxis],
                ),
            )
            for phenotype_index in range(vc.phenotype_count)
        )
        apply = partial(ml.apply, **kwargs)

        with Pool(processes=num_processes) as pool:
            for i, r in tqdm(
                pool.imap_unordered(apply, optimize_jobs),
                desc="fitting null models",
                unit="phenotypes",
                total=vc.phenotype_count,
            ):
                nm.put(i, r)

    def softplus_penalty(self, terms: torch.Tensor, o: OptimizeInput) -> torch.Tensor:
        beta = 1e4
        maximum_variance = o.rotated_phenotype.var() * torch.tensor(
            maximum_variance_multiplier
        )
        upper_penalty = torch.nn.functional.softplus(
            terms[:2] - maximum_variance,
            beta=beta,
        )
        lower_penalty = torch.nn.functional.softplus(
            -terms[:2],
            beta=beta,
        )
        penalty = torch.tensor(beta) * (lower_penalty.sum() + upper_penalty.sum())
        return penalty

    def get_minus_two_log_likelihood_terms(
        self, terms: torch.Tensor, o: OptimizeInput
    ) -> MinusTwoLogLikelihoodTerms:
        sample_count = torch.tensor(self.sample_count, dtype=torch.float64)
        genetic_variance = terms[1]
        r = self.get_regression_weights(terms, o)

        logarithmic_determinant = torch.log(r.variance).sum()

        return MinusTwoLogLikelihoodTerms(
            sample_count=sample_count,
            genetic_variance=genetic_variance,
            logarithmic_determinant=logarithmic_determinant,
            r=r,
        )

    def minus_two_log_likelihood(
        self, terms: torch.Tensor, o: OptimizeInput
    ) -> torch.Tensor:
        t = self.get_minus_two_log_likelihood_terms(terms, o)
        deviation = torch.square(t.r.scaled_residuals).sum()
        minus_two_log_likelihood = t.logarithmic_determinant + deviation

        if self.enable_softplus_penalty:
            minus_two_log_likelihood += self.softplus_penalty(terms, o)

        return torch.where(
            torch.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            torch.inf,
        )


def logdet(a: torch.Tensor) -> torch.Tensor:
    """A re-implementation of torch.logdet that returns infinity instead of NaN, which
    prevents an error in autodiff.

    Args:
        a (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    sign, logabsdet = torch.slogdet(a)
    return torch.where(
        sign == -1.0,
        torch.inf,
        logabsdet,
    )


@dataclass
class RestrictedMaximumLikelihood(ProfileMaximumLikelihood):
    def minus_two_log_likelihood(
        self, terms: torch.Tensor, o: OptimizeInput
    ) -> torch.Tensor:
        t = self.get_minus_two_log_likelihood_terms(terms, o)
        deviation = torch.square(t.r.scaled_residuals).sum()
        penalty = logdet(t.r.scaled_covariates.t() @ t.r.scaled_covariates)
        minus_two_log_likelihood = t.logarithmic_determinant + deviation + penalty

        if self.enable_softplus_penalty:
            minus_two_log_likelihood += self.softplus_penalty(terms, o)

        return torch.where(
            torch.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            torch.inf,
        )


@dataclass
class MaximumPenalizedLikelihood(ProfileMaximumLikelihood):
    """
    - Chung, Y., Rabe-Hesketh, S., Dorie, V., Gelman, A., & Liu, J. (2013).
      A nondegenerate penalized likelihood estimator for variance parameters in
      multilevel models.
      Psychometrika, 78, 685-709.
    - Chung, Y., Rabe-Hesketh, S., & Choi, I. H. (2013).
      Avoiding zero between-study variance estimates in random-effects meta-analysis.
      Statistics in medicine, 32(23), 4071-4089.
    - Chung, Y., Rabe-Hesketh, S., Gelman, A., Liu, J., & Dorie, V. (2012).
      Avoiding boundary estimates in linear mixed models through weakly informative
      priors.
    """

    def minus_two_log_likelihood(
        self, terms: torch.Tensor, o: OptimizeInput
    ) -> torch.Tensor:
        penalty = -torch.log(terms).sum()
        return super().minus_two_log_likelihood(terms, o) + penalty


@dataclass
class MaximumLikelihood(ProfileMaximumLikelihood):
    def get_initial_terms(self, o: OptimizeInput):
        terms = torch.tensor([1, 0], dtype=torch.float64)
        r = super().get_regression_weights(
            terms,
            o,
        )
        regression_weights = list(r.regression_weights.detach().numpy().ravel())
        return super().get_initial_terms(o) + regression_weights

    def bounds(self, o: OptimizeInput) -> list[tuple[float, float]]:
        return super().bounds(o) + [(-np.inf, np.inf)] * self.covariate_count

    @staticmethod
    def get_regression_weights(terms: torch.Tensor, o: OptimizeInput):
        terms = torch.where(
            torch.isfinite(terms),
            terms,
            0,
        )

        (eigenvalues, rotated_covariates, rotated_phenotype) = o

        variance = terms[1] * eigenvalues + terms[0]
        inverse_variance = torch.pow(variance, -0.5)[:, np.newaxis]

        scaled_covariates = rotated_covariates * inverse_variance
        scaled_phenotype = rotated_phenotype * inverse_variance

        regression_weights = terms[2:].unsqueeze(1)
        scaled_residuals = scaled_phenotype - scaled_covariates @ regression_weights
        return (
            regression_weights,
            scaled_residuals,
            variance,
            scaled_covariates,
            scaled_phenotype,
        )

    @classmethod
    def get_standard_errors(cls, terms: torch.Tensor, o: OptimizeInput):
        weights, residuals, variance, _, _ = cls.get_regression_weights(terms, o)

        covariance = hessian(cls.minus_two_log_likelihood)(terms, o)
        inverse_covariance = torch.linalg.inv(covariance)
        standard_errors = torch.sqrt(torch.diagonal(inverse_covariance))
        standard_errors = standard_errors[2:].unsqueeze(1)

        return weights, standard_errors, residuals, variance


@dataclass
class FaST_LMM(ProfileMaximumLikelihood):
    step: float = 0.2

    requires_grad: ClassVar[bool] = False

    def minus_two_log_likelihood(
        self, terms: torch.Tensor, o: OptimizeInput
    ) -> torch.Tensor:
        t = self.get_minus_two_log_likelihood_terms(terms, o)
        minus_two_log_likelihood = (
            t.sample_count
            + t.logarithmic_determinant
            + t.sample_count * torch.log(torch.square(t.r.scaled_residuals).mean())
        )

        if self.enable_softplus_penalty:
            minus_two_log_likelihood += self.softplus_penalty(terms, o)

        return torch.where(
            torch.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            torch.inf,
        )

    def wrapper(self, log_variance_ratio: npt.NDArray, o: OptimizeInput):
        variance_ratio = np.power(10, log_variance_ratio)
        terms = torch.tensor([variance_ratio, 1], dtype=torch.float64)
        return self.func(terms, o).item()

    def optimize(
        self,
        o: OptimizeInput,
        method: str = "TNC",
        enable_hessian: bool = False,
        disp: bool = False,
        **kwargs,
    ) -> OptimizeResult:
        func = self.wrapper

        upper_bound = float(np.log10(o.rotated_phenotype.var()))
        lower_bound = float(np.log10(minimum_variance) - upper_bound)
        xa = np.arange(lower_bound, upper_bound, step=self.step)
        logger.debug(
            f"FaST_LMM will optimize between {lower_bound} to {upper_bound} "
            f"in {xa.size} steps of size {self.step}"
        )

        fmin = np.inf
        best_optimize_result: OptimizeResult | None = None
        with np.errstate(all="ignore"):
            for bounds in zip(xa, xa + self.step):
                try:
                    optimize_result = scipy.optimize.minimize_scalar(  # type: ignore
                        func,
                        args=(o,),
                        bounds=bounds,
                    )
                    if optimize_result.fun < fmin:
                        fmin = optimize_result.fun
                        best_optimize_result = optimize_result
                except FloatingPointError:
                    pass

        if best_optimize_result is None:
            raise RuntimeError

        log_variance_ratio = best_optimize_result.x
        variance_ratio = np.power(10, log_variance_ratio)

        # Scale by genetic variance
        terms = torch.Tensor([variance_ratio, 1])
        _, _, residuals, _ = self.get_standard_errors(terms, o)
        genetic_variance = float(np.square(residuals).mean())
        terms *= genetic_variance

        return OptimizeResult(
            x=terms.numpy(),
            fun=fmin,
        )
