# -*- coding: utf-8 -*-
from dataclasses import dataclass
from functools import cached_property
from multiprocessing import cpu_count
from typing import ClassVar, NamedTuple

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


@dataclass
class ProfileMaximumLikelihood:
    sample_count: int
    covariate_count: int

    needs_grad: ClassVar[bool] = True

    def get_initial_terms(self, o: OptimizeInput):
        var: float = o.rotated_phenotype.var().item()
        return [var / 2] * 2

    def bounds(self, o: OptimizeInput) -> list[tuple[float, float]]:
        variance = o.rotated_phenotype.var().item()
        return [(minimum_variance, variance), (0, variance)]

    @cached_property
    def func(self):
        o = OptimizeInput(
            torch.rand(self.sample_count),
            torch.rand(self.sample_count, self.covariate_count),
            torch.rand(self.sample_count, 1),
        )
        terms = self.terms_to_tensor(self.get_initial_terms(o))

        func = self.minus_two_log_likelihood

        if self.needs_grad:
            func = grad_and_value(func)
        func = memory_efficient_fusion(func)

        # perform warm-up iterations
        for _ in range(3):
            func(terms, o)

        return func

    @staticmethod
    def get_regression_weights(
        terms: torch.Tensor, o: OptimizeInput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        regression_weights = torch.linalg.lstsq(
            scaled_covariates,
            scaled_phenotype,
            rcond=None,
        ).solution

        scaled_residuals = scaled_phenotype - scaled_covariates @ regression_weights

        return (
            regression_weights,
            scaled_residuals,
            variance,
            scaled_covariates,
            scaled_phenotype,
        )

    @classmethod
    def get_standard_errors(
        cls, terms: torch.Tensor, o: OptimizeInput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        weights, residuals, variance, covariates, _ = cls.get_regression_weights(
            terms, o
        )

        degrees_of_freedom = covariates.shape[0] - covariates.shape[1]
        residual_variance = torch.square(residuals).sum() / degrees_of_freedom

        inverse_covariance = torch.linalg.inv(covariates.t() @ covariates)
        standard_errors = residual_variance * torch.sqrt(
            torch.diagonal(inverse_covariance)
        )

        return weights, standard_errors, residuals, variance

    def minus_two_log_likelihood(self, terms: torch.Tensor, o: OptimizeInput):
        _, scaled_residuals, variance, _, _ = self.get_regression_weights(terms, o)
        constant = torch.log(variance).sum()
        minus_two_log_likelihood = (
            constant
            + self.sample_count
            + self.sample_count * torch.log(torch.square(scaled_residuals).mean())
        )
        return torch.where(
            torch.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            torch.inf,
        )

    @staticmethod
    def terms_to_tensor(numpy_terms):
        return torch.tensor(
            numpy_terms,
            dtype=torch.float64,
            requires_grad=True,
        )

    def wrapper(self, numpy_terms: npt.NDArray, o: OptimizeInput):
        terms = self.terms_to_tensor(numpy_terms)
        grad, value = self.func(terms, o)
        return value.item(), grad.detach().numpy()

    @staticmethod
    def get_heritability(terms) -> tuple[float, float, float]:
        genetic_variance = float(terms[1])
        error_variance = float(terms[0])
        heritability = float(genetic_variance / (genetic_variance + error_variance))
        return heritability, genetic_variance, error_variance

    def optimize(self, o: OptimizeInput, scale: bool = False) -> OptimizeResult:
        if scale:
            scaling_factor: float = o.rotated_phenotype.std().item()
            o = OptimizeInput(
                o.eigenvalues,
                o.rotated_covariates,
                o.rotated_phenotype / scaling_factor,
            )
        else:
            scaling_factor = 1

        func = self.wrapper
        optimize_result = scipy.optimize.basinhopping(  # type: ignore
            func,
            self.get_initial_terms(o),
            minimizer_kwargs=dict(
                method="L-BFGS-B", jac=True, bounds=self.bounds(o), args=(o,)
            ),
            niter=2**10,
        )

        optimize_result.x[:2] *= np.square(scaling_factor)
        optimize_result.x[2:] *= scaling_factor

        return optimize_result

    def apply(self, optimize_job: OptimizeJob) -> tuple[int, NullModelResult]:
        (phenotype_index, num_nested_threads, o) = optimize_job
        with threadpool_limits(limits=num_nested_threads):
            optimize_result = self.optimize(o)
            terms = torch.tensor(optimize_result.x)
            weights, errors, residuals, variance = self.get_standard_errors(
                terms,
                o,
            )
            return phenotype_index, NullModelResult(
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

        with Pool(processes=num_processes) as pool:
            for i, r in tqdm(
                pool.imap_unordered(ml.apply, optimize_jobs),
                desc="fitting null models",
                unit="phenotypes",
                total=vc.phenotype_count,
            ):
                nm.put(i, r)


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
    @classmethod
    def minus_two_log_likelihood(cls, variance_terms: torch.Tensor, o: OptimizeInput):
        (
            _,
            scaled_residuals,
            variance,
            scaled_covariates,
            scaled_phenotype,
        ) = cls.get_regression_weights(variance_terms, o)

        minus_two_log_likelihood = (
            torch.log(variance).sum()
            + logdet(scaled_covariates.t() @ scaled_covariates)
            + (scaled_phenotype * scaled_residuals).sum()
        )

        return torch.where(
            torch.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            torch.inf,
        )


@dataclass
class MaximumLikelihood(ProfileMaximumLikelihood):
    def get_initial_terms(self, o: OptimizeInput):
        terms = torch.tensor([1, 0], dtype=torch.float64)
        regression_weights, _, _, _, _ = super().get_regression_weights(
            terms,
            o,
        )
        regression_weights = list(regression_weights.detach().numpy().ravel())
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

    needs_grad: ClassVar[bool] = False

    @staticmethod
    def terms_to_tensor(numpy_terms):
        return torch.tensor(
            numpy_terms,
            dtype=torch.float64,
        )

    def wrapper(self, log_variance_ratio: npt.NDArray, o: OptimizeInput):
        variance_ratio = np.power(10, log_variance_ratio)
        terms = torch.tensor([variance_ratio, 1], dtype=torch.float64)
        return self.func(terms, o).item()

    def optimize(self, o: OptimizeInput, scale: bool = False) -> OptimizeResult:
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
