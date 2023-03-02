# -*- coding: utf-8 -*-
from dataclasses import dataclass

import numpy as np
import scipy
import torch
from functorch import grad
from numpy import typing as npt
from tqdm.auto import tqdm

from gwas.eig import Eigendecomposition
from gwas.log import logger
from gwas.pheno import VariableCollection
from gwas.var import RegressionResult, VarianceRatio

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)


@dataclass
class ProfileMaximumLikelihood:
    sample_count: int

    eigenvalues: torch.Tensor

    rotated_covariates: torch.Tensor
    rotated_phenotype: torch.Tensor

    def get_regression_weights(self, variance_terms: torch.Tensor):
        variance = variance_terms[1] * self.eigenvalues + variance_terms[0]

        inverse_variance = torch.where(
            torch.isclose(variance, torch.tensor(float(0))),
            0,
            torch.reciprocal(variance),
        )

        scaled_covariates = self.rotated_covariates.t() * inverse_variance

        a = scaled_covariates @ self.rotated_covariates
        b = scaled_covariates @ self.rotated_phenotype
        regression_weights = torch.linalg.lstsq(
            a,
            b,
            rcond=None,
            driver="gelsd",
        ).solution

        residuals = (
            self.rotated_phenotype - self.rotated_covariates @ regression_weights
        )

        return regression_weights, residuals, inverse_variance, a

    def minus_two_log_likelihood(self, variance_terms: torch.Tensor):
        _, residuals, inverse_variance, _ = self.get_regression_weights(variance_terms)

        c = (torch.square(residuals) * inverse_variance).sum()

        minus_two_log_likelihood = (
            c - self.sample_count * torch.log(inverse_variance).sum()
        )

        return torch.where(
            torch.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            torch.inf,
        )

    def wrapper(self, numpy_terms: npt.NDArray):
        terms = torch.tensor(
            numpy_terms,
            dtype=torch.float64,
            requires_grad=True,
        )

        return (
            self.minus_two_log_likelihood(terms).item(),
            grad(self.minus_two_log_likelihood)(terms).detach().numpy(),
        )

    def make_result(
        self,
        terms,
        regression_weights: torch.Tensor,
        residuals: torch.Tensor,
        inverse_variance: torch.Tensor,
    ):
        heritability = terms[1] / terms[:2].sum()
        logger.info(
            f"Optimizer result is {terms} "
            f"which corresponds to a heritability of {heritability}"
        )
        return RegressionResult(
            regression_weights.detach().numpy(),
            (residuals.ravel() * inverse_variance).detach().numpy(),
            inverse_variance.detach().numpy(),
            self.sample_count,
        )

    def optimize(self):
        var: float = self.rotated_phenotype.var().item()
        optimize_result = scipy.optimize.basinhopping(
            self.wrapper,
            [var / 2] * 2,
            minimizer_kwargs=dict(
                method="L-BFGS-B", jac=True, bounds=[(0, np.inf), (0, np.inf)]
            ),
            stepsize=var / 10,
        )

        variance_terms = optimize_result.x
        weights, residuals, inverse_variance, _ = self.get_regression_weights(
            variance_terms
        )
        return self.make_result(variance_terms, weights, residuals, inverse_variance)

    @classmethod
    def fit(
        cls,
        eig: Eigendecomposition,
        vc: VariableCollection,
        vr: VarianceRatio,
    ):
        eigenvectors = eig.eigenvectors
        covariates = vc.covariates.to_numpy()
        phenotypes = vc.phenotypes.to_numpy()

        eigenvalues = torch.tensor(eig.eigenvalues)
        rotated_covariates = torch.tensor(eigenvectors.transpose() @ covariates)
        rotated_phenotypes = torch.tensor(eigenvectors.transpose() @ phenotypes)

        for i in tqdm(range(vc.phenotype_count)):
            ml = cls(
                vc.sample_count,
                eigenvalues,
                rotated_covariates,
                rotated_phenotypes[:, i, np.newaxis],
            )
            r = ml.optimize()
            vr.put(i, r)


@dataclass
class RestrictedMaximumLikelihood(ProfileMaximumLikelihood):
    def minus_two_log_likelihood(self, variance_terms: torch.Tensor):
        _, residuals, inverse_variance, a = self.get_regression_weights(variance_terms)

        c = (self.rotated_phenotype * inverse_variance * residuals).sum()

        minus_two_log_likelihood = (
            torch.logdet(a) + c - torch.log(inverse_variance).sum()
        )

        return torch.where(
            torch.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            torch.inf,
        )


@dataclass
class MaximumLikelihood(ProfileMaximumLikelihood):
    def get_residuals(self, terms: torch.Tensor):
        variance = terms[1] * self.eigenvalues + terms[0]
        regression_weights = terms[2:].unsqueeze(1)

        inverse_variance = torch.where(
            torch.isclose(variance, torch.tensor(float(0))),
            0,
            torch.reciprocal(variance),
        )

        residuals = (
            self.rotated_phenotype - self.rotated_covariates @ regression_weights
        )

        return residuals, inverse_variance

    def minus_two_log_likelihood(self, terms: torch.Tensor):
        residuals, inverse_variance = self.get_residuals(terms)
        c = (torch.square(residuals) * inverse_variance).sum()
        minus_two_log_likelihood = (
            c - self.sample_count * torch.log(inverse_variance).sum()
        )
        return torch.where(
            torch.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            torch.inf,
        )

    def optimize(self):
        init_regression_weights_tensor, _, _, _ = self.get_regression_weights(
            torch.tensor([1, 0], dtype=torch.float64)
        )
        init_regression_weights = list(
            init_regression_weights_tensor.detach().numpy().ravel()
        )
        covariate_count = len(init_regression_weights)

        var: float = self.rotated_phenotype.var().item()

        bounds = [(0.0, np.inf), (0.0, np.inf)] + [(-np.inf, np.inf)] * covariate_count
        optimize_result = scipy.optimize.basinhopping(
            self.wrapper,
            [var / 2] * 2 + init_regression_weights,
            minimizer_kwargs=dict(
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            ),
            stepsize=var / 10,
        )

        terms = optimize_result.x
        regression_weights = torch.tensor(terms[2:])
        residuals, inverse_variance = self.get_residuals(torch.tensor(terms))
        return self.make_result(terms, regression_weights, residuals, inverse_variance)
