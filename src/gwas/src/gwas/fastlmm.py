# -*- coding: utf-8 -*-
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import scipy
from numpy import typing as npt
from scipy.optimize import brent
from tqdm.auto import tqdm

from gwas.eig import Eigendecomposition
from gwas.pheno import VariableCollection
from gwas.var import RegressionResult, VarianceRatio


@dataclass
class FastLMM:
    eig: Eigendecomposition
    vc: VariableCollection
    vr: VarianceRatio

    step: float = 0.2
    epsilon: float = 1e-8
    lapack_driver: str = "gelsy"

    def get_sqrt_covariance(
        self,
        logarithmic_variance_ratio: float,
    ) -> npt.NDArray[np.float64]:
        # ensure we don't have overflow/underflow in the calculation
        c = np.hypot(
            self.eig.sqrt_eigenvalues, np.power(10, logarithmic_variance_ratio / 2)
        )
        return c

    def get_regression_weights(
        self,
        logarithmic_variance_ratio: float,
        phenotype: npt.NDArray[np.float64],
    ):
        c = self.get_sqrt_covariance(logarithmic_variance_ratio)
        nonzero = c > np.sqrt(self.epsilon)
        rank = int(np.count_nonzero(nonzero))

        # invert covariance
        inv_sqrt_covariance = np.reciprocal(c[nonzero])

        # partition eigenvectors according to equation 5
        # of 10.1534/genetics.107.080101
        eigenvectors = self.eig.eigenvectors[:, nonzero]
        covariates = self.vc.covariates.to_numpy()
        rotated_covariates = eigenvectors.transpose() @ covariates
        rotated_phenotype = eigenvectors.transpose() @ phenotype

        # calculate regression
        scaled_covariates = rotated_covariates.transpose() * inv_sqrt_covariance
        scaled_phenotype = rotated_phenotype * inv_sqrt_covariance
        a = scaled_covariates @ scaled_covariates.transpose()
        b = scaled_covariates @ scaled_phenotype
        regression_weights, _, _, _ = scipy.linalg.lstsq(  # min || m x m - m x 1 ||
            a,
            b,
            lapack_driver=self.lapack_driver,
            overwrite_a=True,
            overwrite_b=True,
        )
        rotated_residuals = (  # n x 1 - n x m . m x 1
            rotated_phenotype - rotated_covariates @ regression_weights
        )
        genetic_variance = float(
            np.square(rotated_residuals * inv_sqrt_covariance).mean()
        )

        # we need the scaled values for the next steps
        inv_covariance = np.square(inv_sqrt_covariance)
        scaled_inv_covariance = inv_covariance / genetic_variance
        return (
            RegressionResult(
                regression_weights,
                rotated_residuals * scaled_inv_covariance,
                scaled_inv_covariance,
                rank,
            ),
            genetic_variance,
        )

    def minus_two_log_likelihood(
        self,
        logarithmic_variance_ratio: float,
        phenotype: npt.NDArray[np.float64],
    ) -> float:
        r, genetic_variance = self.get_regression_weights(
            logarithmic_variance_ratio,
            phenotype,
        )

        # calculate log( eigenvalues + variance_ratio )
        # but with increased numerical precision as per
        # https://stackoverflow.com/a/65233446
        sqrt_eigenvalues = self.eig.sqrt_eigenvalues
        log_eigenvalues = 2 * np.log(sqrt_eigenvalues)
        logarithmic_determinant = (
            log_eigenvalues
            + np.log1p(
                np.power(10, logarithmic_variance_ratio)
                * np.power(sqrt_eigenvalues, -2)
            )
        ).sum()

        # -2 log-likelihood
        # as per equation A14 of 10.1534/genetics.107.080101
        # and supplement 1.4 of 10.1038/nmeth.1681
        return (
            r.rank * np.log(2 * np.pi)
            + logarithmic_determinant
            + r.rank
            + r.rank * np.log(genetic_variance)
        )

    def optimize(
        self,
        phenotype: npt.NDArray[np.float64],
    ):
        logarithmic_variance_ratio: float | None = None

        fmin = np.inf

        xa = np.arange(-10, 10, step=self.step)

        with np.errstate(all="ignore"):
            for brack in zip(xa, xa + self.step):
                try:
                    x, fval, _, _ = brent(
                        self.minus_two_log_likelihood,
                        args=(phenotype,),
                        brack=brack,
                        full_output=True,
                    )
                    if fval < fmin:
                        fmin = fval
                        logarithmic_variance_ratio = x
                except FloatingPointError:
                    pass

        if logarithmic_variance_ratio is None:
            raise ValueError

        return logarithmic_variance_ratio

    def apply(self, phenotype_index: int):
        a = self.vc.phenotypes.to_numpy()
        phenotype = a[:, phenotype_index]

        return_value = 1
        try:
            logarithmic_variance_ratio = self.optimize(phenotype)
            r, _ = self.get_regression_weights(logarithmic_variance_ratio, phenotype)
        except ValueError:
            r = RegressionResult.null()
            return_value = 0
        self.vr.put(phenotype_index, r)

        return return_value

    @classmethod
    def fit(
        cls,
        eig: Eigendecomposition,
        vc: VariableCollection,
        vr: VarianceRatio,
        **kwargs,
    ):
        fastlmm = cls(eig, vc, vr, **kwargs)
        phenotype_count = fastlmm.vc.phenotype_count

        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(fastlmm.apply, range(phenotype_count))):
                pass
