# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy import typing as npt

from gwas.eig import Eigendecomposition
from gwas.mem.arr import SharedArray
from gwas.pheno import VariableCollection


@dataclass
class NullModelResult:
    heritability: float
    genetic_variance: float
    error_variance: float

    regression_weights: float | npt.NDArray[np.float64]
    standard_errors: float | npt.NDArray[np.float64]

    scaled_residuals: float | npt.NDArray[np.float64]
    variance: float | npt.NDArray[np.float64]

    @classmethod
    def null(cls) -> Self:
        return cls(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


@dataclass
class NullModelCollection:
    method: str | None

    heritability: npt.NDArray[np.float64]
    genetic_variance: npt.NDArray[np.float64]
    error_variance: npt.NDArray[np.float64]

    regression_weights: SharedArray
    standard_errors: SharedArray
    scaled_residuals: SharedArray
    variance: SharedArray

    methods: ClassVar[list[str]] = ["fastlmm", "ml", "pml", "rml"]

    @property
    def phenotype_count(self) -> int:
        return self.regression_weights.shape[0]

    @property
    def sample_count(self) -> int:
        return self.scaled_residuals.shape[1]

    def put(self, phenotype_index: int, r: NullModelResult) -> None:
        self.heritability[phenotype_index] = r.heritability
        self.genetic_variance[phenotype_index] = r.genetic_variance
        self.error_variance[phenotype_index] = r.error_variance

        weights = self.regression_weights.to_numpy()
        errors = self.standard_errors.to_numpy()
        residuals = self.scaled_residuals.to_numpy()
        variance = self.variance.to_numpy()

        weights[phenotype_index, :] = np.ravel(r.regression_weights)
        errors[phenotype_index, :] = np.ravel(r.standard_errors)

        residuals[:, phenotype_index] = np.ravel(r.scaled_residuals)
        variance[:, phenotype_index] = np.ravel(r.variance)

    def free(self) -> None:
        self.regression_weights.free()
        self.standard_errors.free()
        self.scaled_residuals.free()
        self.variance.free()

    @classmethod
    def from_eig(
        cls,
        eig: Eigendecomposition,
        vc: VariableCollection,
        method: str | None = "ml",
        **kwargs,
    ) -> Self:
        from .ml import (
            FaST_LMM,
            MaximumLikelihood,
            ProfileMaximumLikelihood,
            RestrictedMaximumLikelihood,
        )

        if eig.samples != vc.samples:
            raise ValueError("Arguments `eig` and `vc` must have the same samples.")

        sw = eig.sw
        name = SharedArray.get_name(sw, "regression-weights")
        regression_weights = sw.alloc(name, vc.phenotype_count, vc.covariate_count)
        name = SharedArray.get_name(sw, "standard-errors")
        standard_errors = sw.alloc(name, vc.phenotype_count, vc.covariate_count)
        name = SharedArray.get_name(sw, "variance")
        shape = list(vc.phenotypes.shape)
        variance = sw.alloc(name, *shape)
        name = SharedArray.get_name(sw, "scaled-residuals")
        scaled_residuals = sw.alloc(name, *shape)

        nm = cls(
            method,
            np.full((vc.phenotype_count,), np.nan),
            np.full((vc.phenotype_count,), np.nan),
            np.full((vc.phenotype_count,), np.nan),
            regression_weights,
            standard_errors,
            scaled_residuals,
            variance,
        )

        if method == "fastlmm":
            FaST_LMM.fit(eig, vc, nm, **kwargs)
        elif method == "ml":
            MaximumLikelihood.fit(eig, vc, nm, **kwargs)
        elif method == "pml":
            ProfileMaximumLikelihood.fit(eig, vc, nm, **kwargs)
        elif method == "reml":
            RestrictedMaximumLikelihood.fit(eig, vc, nm, **kwargs)

        return nm
