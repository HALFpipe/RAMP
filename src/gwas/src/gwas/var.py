# -*- coding: utf-8 -*-
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import scipy
from numpy import typing as npt
from scipy.optimize import brent
from tqdm.auto import tqdm

from gwas.eig import Eigendecomposition
from gwas.mem.arr import SharedArray
from gwas.pheno import VariableCollection


@dataclass
class RegressionResult:
    regression_weights: float | npt.NDArray[np.float64]
    rotated_residuals: float | npt.NDArray[np.float64]
    inv_covariance: float | npt.NDArray[np.float64]
    rank: int


@dataclass
class VarianceRatio:
    inv_covariance: SharedArray
    regression_weights: SharedArray
    rotated_residuals: SharedArray

    def put(self, phenotype_index: int, r: RegressionResult):
        rr = self.rotated_residuals.to_numpy()
        rr[: r.rank, phenotype_index] = r.rotated_residuals
        rr[r.rank :, phenotype_index] = 0

        c = self.inv_covariance.to_numpy()
        c[: r.rank, phenotype_index] = r.inv_covariance
        c[r.rank :, phenotype_index] = 0

        rw = self.regression_weights.to_numpy()
        rw[phenotype_index, :] = r.regression_weights

    @classmethod
    def from_eig(cls, eig: Eigendecomposition, vc: VariableCollection, **kwargs):
        sw = eig.sw
        name = SharedArray.get_name(sw, "covariance")
        covariance = sw.alloc(name, *vc.phenotypes.shape)
        name = SharedArray.get_name(sw, "rotated-residuals")
        rotated_residuals = sw.alloc(name, *vc.phenotypes.shape)
        name = SharedArray.get_name(sw, "regression-weights")
        regression_weights = sw.alloc(name, vc.phenotype_count, vc.covariate_count)

        vr = cls(
            covariance,
            regression_weights,
            rotated_residuals,
        )

        fastlmm = FastLMM(eig, vc, vr, **kwargs)
        fastlmm.fit()

        return vr


def scale_rows(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return (a.transpose() * b.ravel()).transpose()


@dataclass
class FastLMM:
    eig: Eigendecomposition
    vc: VariableCollection
    vr: VarianceRatio

    step: float = 0.2
    epsilon: float = 1e-8
    lapack_driver: str = "gelsy"

    def rotate(self, a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return self.eig.eigenvectors.transpose() @ a

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
    ) -> RegressionResult:
        c = self.get_sqrt_covariance(logarithmic_variance_ratio)
        nonzero = c > np.sqrt(self.epsilon)
        rank = int(np.count_nonzero(nonzero))

        # partition eigenvectors according to equation 5
        # of 10.1534/genetics.107.080101
        eigenvectors = self.eig.eigenvectors[:, nonzero]
        covariates = self.vc.covariates.to_numpy()
        rotated_covariates = eigenvectors.transpose() @ covariates
        rotated_phenotype = eigenvectors.transpose() @ phenotype

        # invert covariance
        inv_sqrt_covariance = np.reciprocal(c[nonzero])
        inv_covariance = np.square(inv_sqrt_covariance)

        # calculate regression
        scaled_covariates = rotated_covariates.transpose() * inv_covariance
        a = scaled_covariates @ rotated_covariates
        b = scaled_covariates @ rotated_phenotype
        regression_weights, _, _, _ = scipy.linalg.lstsq(  # min || n x m - n x 1 ||
            a,
            b,
            lapack_driver=self.lapack_driver,
            overwrite_a=True,
            overwrite_b=True,
        )
        rotated_residuals = (  # n x 1 - n x m . m x 1
            rotated_phenotype - rotated_covariates @ regression_weights
        )

        # we need the squared covariance for the next steps
        return RegressionResult(
            regression_weights,
            rotated_residuals,
            inv_covariance,
            rank,
        )

    def minus_two_log_likelihood(
        self,
        logarithmic_variance_ratio: float,
        phenotype: npt.NDArray[np.float64],
    ) -> float:
        r = self.get_regression_weights(
            logarithmic_variance_ratio,
            phenotype,
        )

        genetic_variance = float(
            np.square(r.rotated_residuals * r.inv_covariance).mean()
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
        rotated_phenotype = a[:, phenotype_index]

        return_value = 1
        try:
            logarithmic_variance_ratio = self.optimize(rotated_phenotype)
            r = self.get_regression_weights(
                logarithmic_variance_ratio, rotated_phenotype
            )
        except ValueError:
            r = RegressionResult(np.nan, np.nan, np.nan, -1)
            return_value = 0
        self.vr.put(phenotype_index, r)

        return return_value

    def fit(self):
        phenotype_count = self.vc.phenotype_count

        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(self.apply, range(phenotype_count))):
                pass
