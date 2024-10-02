from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from numpy import typing as npt

from ..eig.base import Eigendecomposition
from ..mem.arr import SharedArray
from ..pheno import VariableCollection
from ..utils.multiprocessing import get_global_lock


@dataclass(kw_only=True)
class NullModelResult:
    log_likelihood: float
    heritability: float
    genetic_variance: float
    error_variance: float

    regression_weights: float | npt.NDArray[np.float64]
    standard_errors: float | npt.NDArray[np.float64]

    halfway_scaled_residuals: float | npt.NDArray[np.float64]
    variance: float | npt.NDArray[np.float64]

    @classmethod
    def null(cls) -> Self:
        return cls(
            log_likelihood=np.nan,
            heritability=np.nan,
            genetic_variance=np.nan,
            error_variance=np.nan,
            regression_weights=np.nan,
            standard_errors=np.nan,
            halfway_scaled_residuals=np.nan,
            variance=np.nan,
        )


@dataclass
class NullModelCollection(SharedArray[np.float64]):
    method: str | None

    phenotype_count: int
    covariate_count: int
    sample_count: int

    methods: ClassVar[list[str]] = ["fastlmm", "pml", "mpl", "reml", "ml"]

    def sub_array(self, start: int, shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
        size = np.prod(shape)
        array = self.to_numpy()[start : start + size]
        return array.reshape(shape)

    @property
    def per_phenotype_values(self) -> npt.NDArray[np.float64]:
        start = 0
        shape = (self.phenotype_count, 4)
        return self.sub_array(start, shape)

    @property
    def per_covariate_values(self) -> npt.NDArray[np.float64]:
        start = self.phenotype_count * 4
        shape = (self.phenotype_count, self.covariate_count, 2)
        return self.sub_array(start, shape)

    @property
    def per_sample_values(self) -> npt.NDArray[np.float64]:
        start = self.phenotype_count * (4 + 2 * self.covariate_count)
        shape = (self.sample_count, self.phenotype_count, 2)
        return self.sub_array(start, shape)

    @property
    def log_likelihood(self) -> npt.NDArray[np.float64]:
        return self.per_phenotype_values[:, 0]

    @property
    def heritability(self) -> npt.NDArray[np.float64]:
        return self.per_phenotype_values[:, 1]

    @property
    def genetic_variance(self) -> npt.NDArray[np.float64]:
        return self.per_phenotype_values[:, 2]

    @property
    def error_variance(self) -> npt.NDArray[np.float64]:
        return self.per_phenotype_values[:, 3]

    @property
    def regression_weights(self) -> npt.NDArray[np.float64]:
        return self.per_covariate_values[:, :, 0]

    @property
    def standard_errors(self) -> npt.NDArray[np.float64]:
        return self.per_covariate_values[:, :, 1]

    @property
    def halfway_scaled_residuals(self) -> npt.NDArray[np.float64]:
        return self.per_sample_values[:, :, 0]

    @property
    def variance(self) -> npt.NDArray[np.float64]:
        return self.per_sample_values[:, :, 1]

    def put(self, phenotype_index: int, r: NullModelResult) -> None:
        self.log_likelihood[phenotype_index] = r.log_likelihood
        self.heritability[phenotype_index] = r.heritability
        self.genetic_variance[phenotype_index] = r.genetic_variance
        self.error_variance[phenotype_index] = r.error_variance

        self.regression_weights[phenotype_index, :] = np.ravel(r.regression_weights)
        self.standard_errors[phenotype_index, :] = np.ravel(r.standard_errors)

        self.halfway_scaled_residuals[:, phenotype_index] = np.ravel(
            r.halfway_scaled_residuals
        )
        self.variance[:, phenotype_index] = np.ravel(r.variance)

    def get_arrays_for_score_calc(self) -> tuple[SharedArray, SharedArray]:
        sample_count = self.sample_count
        phenotype_count = self.phenotype_count

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
        np.reciprocal(self.variance, out=inverse_variance_matrix)
        # Pre-compute the inverse variance scaled residuals
        np.true_divide(
            self.halfway_scaled_residuals,
            np.sqrt(self.variance),
            out=scaled_residuals_matrix,
        )
        return inverse_variance_array, scaled_residuals_array

    @classmethod
    def empty(
        cls, eig: Eigendecomposition, vc: VariableCollection, method: str | None = None
    ) -> Self:
        if eig.samples != vc.samples:
            raise ValueError("Arguments `eig` and `vc` must have the same samples.")

        sw = eig.sw

        size = vc.phenotype_count * (4 + 2 * vc.covariate_count + 2 * vc.sample_count)

        with get_global_lock():
            name = SharedArray.get_name(sw, prefix="null-model-collection")
            sw.alloc(name, size, dtype=np.float64)
        nm = cls(
            name,
            sw,
            method=method,
            phenotype_count=vc.phenotype_count,
            covariate_count=vc.covariate_count,
            sample_count=vc.sample_count,
        )

        return nm
