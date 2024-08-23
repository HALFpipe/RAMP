from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy import typing as npt

from ..eig.base import Eigendecomposition
from ..mem.arr import SharedArray
from ..mem.wkspace import SharedWorkspace
from ..pheno import VariableCollection
from ..utils.multiprocessing import get_global_lock


@dataclass
class NullModelResult:
    log_likelihood: float
    heritability: float
    genetic_variance: float
    error_variance: float

    regression_weights: float | npt.NDArray[np.float64]
    standard_errors: float | npt.NDArray[np.float64]

    half_scaled_residuals: float | npt.NDArray[np.float64]
    variance: float | npt.NDArray[np.float64]

    @classmethod
    def null(cls) -> Self:
        return cls(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


@dataclass
class NullModelCollection:
    method: str | None

    log_likelihood: npt.NDArray[np.float64]
    heritability: npt.NDArray[np.float64]
    genetic_variance: npt.NDArray[np.float64]
    error_variance: npt.NDArray[np.float64]

    regression_weights: SharedArray
    standard_errors: SharedArray
    half_scaled_residuals: SharedArray
    variance: SharedArray

    methods: ClassVar[list[str]] = ["fastlmm", "pfastlmm", "pml", "mpl", "reml", "ml"]

    @property
    def phenotype_count(self) -> int:
        return self.regression_weights.shape[0]

    @property
    def sample_count(self) -> int:
        return self.half_scaled_residuals.shape[1]

    @property
    def sw(self) -> SharedWorkspace:
        return self.regression_weights.sw

    def put(self, phenotype_index: int, r: NullModelResult) -> None:
        self.log_likelihood[phenotype_index] = r.log_likelihood
        self.heritability[phenotype_index] = r.heritability
        self.genetic_variance[phenotype_index] = r.genetic_variance
        self.error_variance[phenotype_index] = r.error_variance

        weights = self.regression_weights.to_numpy()
        errors = self.standard_errors.to_numpy()
        residuals = self.half_scaled_residuals.to_numpy()
        variance = self.variance.to_numpy()

        weights[phenotype_index, :] = np.ravel(r.regression_weights)
        errors[phenotype_index, :] = np.ravel(r.standard_errors)

        residuals[:, phenotype_index] = np.ravel(r.half_scaled_residuals)
        variance[:, phenotype_index] = np.ravel(r.variance)

    def get_arrays_for_score_calc(self) -> tuple[SharedArray, SharedArray]:
        variance = self.variance.to_numpy()
        (sample_count, phenotype_count) = variance.shape
        half_scaled_residuals = self.half_scaled_residuals.to_numpy()

        with get_global_lock():
            inverse_variance_array = self.sw.alloc(
                SharedArray.get_name(self.sw, prefix="inverse-variance"),
                sample_count,
                phenotype_count,
            )
            scaled_residuals_array = self.sw.alloc(
                SharedArray.get_name(self.sw, prefix="scaled-residuals"),
                sample_count,
                phenotype_count,
            )
        inverse_variance_matrix = inverse_variance_array.to_numpy()
        scaled_residuals_matrix = scaled_residuals_array.to_numpy()

        # Pre-compute the inverse variance.
        np.reciprocal(variance, out=inverse_variance_matrix)
        # Pre-compute the inverse variance scaled residuals
        np.true_divide(
            half_scaled_residuals,
            np.sqrt(variance),
            out=scaled_residuals_matrix,
        )
        return inverse_variance_array, scaled_residuals_array

    def free(self) -> None:
        self.regression_weights.free()
        self.standard_errors.free()
        self.half_scaled_residuals.free()
        self.variance.free()

    @classmethod
    def empty(
        cls, eig: Eigendecomposition, vc: VariableCollection, method: str | None = None
    ) -> Self:
        if eig.samples != vc.samples:
            raise ValueError("Arguments `eig` and `vc` must have the same samples.")

        sw = eig.sw

        with get_global_lock():
            name = SharedArray.get_name(sw, prefix="regression-weights")
            regression_weights = sw.alloc(name, vc.phenotype_count, vc.covariate_count)
            name = SharedArray.get_name(sw, prefix="standard-errors")
            standard_errors = sw.alloc(name, vc.phenotype_count, vc.covariate_count)
            name = SharedArray.get_name(sw, prefix="variance")
            shape = list(vc.phenotypes.shape)
            variance = sw.alloc(name, *shape)
            name = SharedArray.get_name(sw, prefix="half-scaled-residuals")
            scaled_residuals = sw.alloc(name, *shape)

        nm = cls(
            method,
            np.full((vc.phenotype_count,), np.nan),
            np.full((vc.phenotype_count,), np.nan),
            np.full((vc.phenotype_count,), np.nan),
            np.full((vc.phenotype_count,), np.nan),
            regression_weights,
            standard_errors,
            scaled_residuals,
            variance,
        )

        return nm
