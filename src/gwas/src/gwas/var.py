# -*- coding: utf-8 -*-
from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from gwas.eig import Eigendecomposition
from gwas.mem.arr import SharedArray
from gwas.pheno import VariableCollection


@dataclass
class RegressionResult:
    regression_weights: float | npt.NDArray[np.float64]
    scaled_residuals: float | npt.NDArray[np.float64]
    variance: float | npt.NDArray[np.float64]
    rank: int

    @classmethod
    def null(cls):
        return cls(np.nan, np.nan, np.nan, 0)


@dataclass
class VarianceRatio:
    variance: SharedArray
    regression_weights: SharedArray
    scaled_residuals: SharedArray

    def put(self, phenotype_index: int, r: RegressionResult):
        residuals = self.scaled_residuals.to_numpy()
        residuals[: r.rank, phenotype_index] = np.ravel(r.scaled_residuals)
        residuals[r.rank :, phenotype_index] = 0

        variance = self.variance.to_numpy()
        variance[: r.rank, phenotype_index] = np.ravel(r.variance)
        variance[r.rank :, phenotype_index] = 0

        weights = self.regression_weights.to_numpy()
        weights[phenotype_index, :] = np.ravel(r.regression_weights)

    @classmethod
    def from_eig(
        cls,
        eig: Eigendecomposition,
        vc: VariableCollection,
        method: str | None = "pml",
        **kwargs,
    ):
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

        if method == "fastlmm":
            from .fastlmm import FastLMM

            FastLMM.fit(eig, vc, vr, **kwargs)
        elif method == "ml":
            from .ml import MaximumLikelihood

            MaximumLikelihood.fit(eig, vc, vr)
        elif method == "pml":
            from .ml import ProfileMaximumLikelihood

            ProfileMaximumLikelihood.fit(eig, vc, vr)
        elif method == "reml":
            from .ml import RestrictedMaximumLikelihood

            RestrictedMaximumLikelihood.fit(eig, vc, vr)

        return vr
