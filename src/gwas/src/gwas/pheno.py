from dataclasses import dataclass
from functools import reduce
from typing import Self

import numpy as np
import pandas as pd
from numpy import typing as npt
from upath import UPath

from .compression.arr.base import (
    CompressionMethod,
    FileArray,
)
from .log import logger
from .mem.arr import SharedArray
from .mem.wkspace import SharedWorkspace


def combine(data_frames: list[pd.DataFrame]) -> pd.DataFrame:
    return reduce(lambda a, b: a.combine_first(b), data_frames)


def read_and_combine(
    paths: list[UPath],
) -> tuple[list[str], list[str], npt.NDArray[np.float64]]:
    import pandas as pd

    data_frames: list[pd.DataFrame] = list()
    for path in paths:
        data_frame = pd.read_table(path, sep="\t", header=0, index_col=False, dtype=str)
        data_frame = data_frame.set_index(data_frame.columns[0])
        data_frames.append(data_frame)

    data_frame = combine(data_frames)

    return (
        list(data_frame.index),
        list(data_frame.columns),
        data_frame.to_numpy(dtype=np.float64),
    )


@dataclass
class VariableCollection:
    samples: list[str]

    phenotype_names: list[str]
    phenotypes: SharedArray

    covariate_names: list[str]
    covariates: SharedArray

    name: str | None = None

    @property
    def sample_count(self) -> int:
        sample_count = self.covariates.shape[0]
        if sample_count != self.phenotypes.shape[0]:
            raise ValueError
        if sample_count != len(self.samples):
            raise ValueError
        return sample_count

    @property
    def covariate_count(self) -> int:
        return self.covariates.shape[1]

    @property
    def phenotype_count(self) -> int:
        return self.phenotypes.shape[1]

    @property
    def sw(self) -> SharedWorkspace:
        return self.phenotypes.sw

    @property
    def covariate_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=self.covariates.to_numpy(),
            copy=False,
            columns=self.covariate_names,
            index=self.samples,
        )

    @property
    def phenotype_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=self.phenotypes.to_numpy(),
            copy=False,
            columns=self.phenotype_names,
            index=self.samples,
        )

    @property
    def data_frame(self) -> pd.DataFrame:
        return combine([self.covariate_frame, self.phenotype_frame])

    @property
    def is_finite(self) -> bool:
        return bool(
            np.isfinite(self.phenotypes.to_numpy()).all()
            and np.isfinite(self.covariates.to_numpy()).all()
        )

    def __post_init__(self) -> None:
        self.remove_zero_variance_covariates()

    def copy(
        self, phenotype_names: list[str] | None = None, samples: list[str] | None = None
    ) -> Self:
        phenotypes = self.phenotypes.to_numpy()
        covariates = self.covariates.to_numpy()

        if phenotype_names is not None:
            phenotypes = self.subset_phenotypes_array(
                phenotypes, self.phenotype_names, phenotype_names
            )
        else:
            phenotype_names = self.phenotype_names.copy()

        if samples is not None:
            phenotypes, covariates = self.subset_samples_array(
                self.samples, samples, phenotypes, covariates
            )
        else:
            samples = self.samples.copy()

        vc = self.from_arrays(
            samples,
            phenotype_names,
            phenotypes,
            self.covariate_names.copy(),
            covariates,
            self.sw,
            missing_value_strategy="listwise_deletion",  # No need to remove any samples
        )
        vc.name = self.name
        return vc

    def free(self) -> None:
        self.phenotypes.free()
        self.covariates.free()

    @staticmethod
    def subset_phenotypes_array(
        phenotypes: npt.NDArray[np.float64],
        phenotype_names: list[str],
        subset_phenotype_names: list[str],
    ) -> npt.NDArray[np.float64]:
        phenotype_indices = [
            phenotype_names.index(name) for name in subset_phenotype_names
        ]
        return phenotypes[:, phenotype_indices]

    def subset_phenotypes(self, subset_phenotype_names: list[str]) -> None:
        if subset_phenotype_names == self.phenotype_names:
            # Nothing to do.
            return
        self.phenotype_names = subset_phenotype_names
        new_phenotypes = self.subset_phenotypes_array(
            self.phenotypes.to_numpy(), self.phenotype_names, subset_phenotype_names
        )
        self.phenotypes.free()
        self.phenotypes = SharedArray.from_numpy(
            new_phenotypes, self.phenotypes.sw, prefix="phenotypes"
        )
        logger.debug(
            f"Subsetting variable collection to have {self.phenotype_count} phenotypes"
        )

    def remove_zero_variance_covariates(self) -> None:
        sw = self.covariates.sw

        old_covariates = self.covariates

        new_covariates = old_covariates.to_numpy()
        zero_variance: npt.NDArray[np.bool_] = np.isclose(new_covariates.var(axis=0), 0)
        zero_variance[0] = False  # Do not remove intercept

        if np.any(zero_variance):
            removed_covariates = [
                name
                for name, zero in zip(self.covariate_names, zero_variance, strict=True)
                if zero
            ]
            logger.debug(
                f"Removing covariates {removed_covariates} because they have zero "
                "variance"
            )
            self.covariate_names = [
                name
                for name, zero in zip(self.covariate_names, zero_variance, strict=True)
                if not zero
            ]
            new_covariates = new_covariates[:, ~zero_variance]

            self.covariates = SharedArray.from_numpy(
                new_covariates, sw, prefix="covariates"
            )
            old_covariates.free()

    @staticmethod
    def subset_samples_array(
        samples: list[str],
        subset_samples: list[str],
        phenotypes: npt.NDArray[np.float64],
        covariates: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        sample_indices = [samples.index(sample) for sample in subset_samples]
        new_phenotypes = phenotypes[sample_indices, :]
        new_covariates = covariates[sample_indices, :]
        return new_phenotypes, new_covariates

    def subset_samples(self, samples: list[str]) -> None:
        if samples == self.samples:
            # Nothing to do.
            return

        sw = self.phenotypes.sw

        old_phenotypes = self.phenotypes
        old_covariates = self.covariates

        new_phenotypes, new_covariates = self.subset_samples_array(
            self.samples, samples, old_phenotypes.to_numpy(), old_covariates.to_numpy()
        )

        self.samples = samples

        self.phenotypes = SharedArray.from_numpy(new_phenotypes, sw, prefix="phenotypes")
        old_phenotypes.free()

        self.covariates = SharedArray.from_numpy(new_covariates, sw, prefix="covariates")
        old_covariates.free()
        self.remove_zero_variance_covariates()

        logger.info(
            f"Subsetting variable collection to have {len(samples)} samples, "
            f"{self.covariate_count} covariates, and {self.phenotype_count} phenotypes."
        )

    @classmethod
    def from_arrays(
        cls,
        samples: list[str],
        phenotype_names: list[str],
        phenotypes: npt.NDArray[np.float64],
        covariate_names: list[str],
        covariates: npt.NDArray[np.float64],
        sw: SharedWorkspace,
        missing_value_strategy: str = "complete_samples",
        **kwargs,
    ) -> Self:
        # Add intercept if not present.
        first_column = covariates[:, 0, np.newaxis]
        if not np.allclose(first_column, 1):
            covariates = np.hstack([np.ones_like(first_column), covariates])
            covariate_names = ["intercept"] + covariate_names

        # Apply missing value strategy.
        if missing_value_strategy == "complete_samples":
            # Remove samples with any missing values.
            criterion = np.isfinite(phenotypes).all(axis=1) & np.isfinite(
                covariates
            ).all(axis=1)
        elif missing_value_strategy == "listwise_deletion":
            # Only remove samples with all-missing values.
            criterion = np.isfinite(phenotypes).any(axis=1) | np.isfinite(
                covariates
            ).any(axis=1)
        else:
            raise ValueError(f"Unknown missing value strategy: {missing_value_strategy}")

        samples = [sample for sample, c in zip(samples, criterion, strict=True) if c]
        phenotypes = phenotypes[criterion, :]
        covariates = covariates[criterion, :]

        logger.debug(
            f"Creating variable collection with {len(samples)} samples, "
            f"{covariates.shape[1]} covariates, and {phenotypes.shape[1]} phenotypes."
        )

        return cls(
            samples,
            phenotype_names,
            SharedArray.from_numpy(phenotypes, sw, prefix="phenotypes"),
            covariate_names,
            SharedArray.from_numpy(covariates, sw, prefix="covariates"),
            **kwargs,
        )

    @classmethod
    def from_txt(
        cls,
        phenotype_paths: list[UPath],
        covariate_paths: list[UPath],
        sw: SharedWorkspace,
        samples: list[str] | None = None,
        **kwargs: str,
    ) -> Self:
        logger.debug("Reading phenotypes")
        phenotype_samples, phenotype_names, phenotype_array = read_and_combine(
            phenotype_paths
        )
        if samples is None:
            samples = phenotype_samples
        if samples is None:
            raise RuntimeError("No samples found in phenotype files")

        logger.debug("Reading covariates")
        covariate_samples, covariate_names, covariate_array = read_and_combine(
            covariate_paths
        )

        # Use only samples that are present in all files
        sample_indices = [
            phenotype_samples.index(sample)
            for sample in samples
            if sample in phenotype_samples and sample in covariate_samples
        ]
        phenotype_samples = [phenotype_samples[i] for i in sample_indices]
        phenotype_array = phenotype_array[sample_indices, :]

        # Ensure that covariates are in the same order as phenotypes
        sample_indices = [
            covariate_samples.index(sample)
            for sample in phenotype_samples
            if sample in samples and sample in covariate_samples
        ]
        covariate_samples = [covariate_samples[i] for i in sample_indices]
        covariate_array = covariate_array[sample_indices, :]

        samples = [sample for sample in samples if sample in phenotype_samples]

        if samples is None:
            raise RuntimeError

        return cls.from_arrays(
            samples,
            phenotype_names,
            phenotype_array,
            covariate_names,
            covariate_array,
            sw,
            **kwargs,
        )

    def covariance_to_txt(
        self, path: UPath, compression_method: CompressionMethod, num_threads: int
    ) -> None:
        path = path.with_suffix(compression_method.suffix)
        if path.is_file():
            logger.debug("Skip writing covariance matrix because it already exists")
            return

        phenotype_array = self.phenotypes.to_numpy()
        covariate_array = self.covariates.to_numpy()

        array = np.hstack((phenotype_array, covariate_array))
        names = [*self.phenotype_names, *self.covariate_names]

        data_frame = pd.DataFrame(array, index=self.samples, columns=names)

        logger.debug("Calculating covariance matrix")
        covariance = data_frame.cov().to_numpy(dtype=np.float64)

        file_array = FileArray.create(
            path,
            covariance.shape,
            covariance.dtype,
            compression_method,
            num_threads=num_threads,
        )
        data_frame = pd.DataFrame(dict(variable=names))
        with file_array:
            file_array.set_axis_metadata(0, data_frame)
            file_array.set_axis_metadata(1, names)
            file_array[:, :] = covariance


@dataclass
class VariableSummary:
    minimum: float
    lower_quartile: float
    median: float
    upper_quartile: float
    maximum: float

    mean: float
    variance: float

    @property
    def values(self) -> npt.NDArray[np.float64]:
        return np.array(
            [
                self.minimum,
                self.lower_quartile,
                self.median,
                self.upper_quartile,
                self.maximum,
                self.mean,
                self.variance,
            ]
        )

    def is_close(self, other: Self) -> bool:
        return np.allclose(self.values, other.values)

    @classmethod
    def from_array(cls, value: npt.NDArray[np.float64]) -> Self:
        if value.size == 0:
            raise ValueError("Cannot compute summary from an empty array")
        quantiles = np.quantile(
            value,
            np.array([0, 0.25, 0.5, 0.75, 1]),
        )
        (minimum, lower_quartile, median, upper_quartile, maximum) = quantiles.ravel()
        return cls(
            float(minimum),
            float(lower_quartile),
            float(median),
            float(upper_quartile),
            float(maximum),
            float(value.mean()),
            float(value.var(ddof=1)),
        )
