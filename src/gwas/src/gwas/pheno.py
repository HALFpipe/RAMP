# -*- coding: utf-8 -*-

from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from numpy import typing as npt

from .log import logger
from .mem.arr import SharedArray
from .mem.wkspace import SharedWorkspace


def combine(data_frames: list[pd.DataFrame]) -> pd.DataFrame:
    return reduce(lambda a, b: a.combine_first(b), data_frames)


def read_and_combine(paths: list[Path]) -> tuple[list[str], list[str], npt.NDArray]:
    import pandas as pd

    data_frames: list[pd.DataFrame] = list()
    for path in paths:
        data_frame = pd.read_table(path, sep="\t", header=0, index_col=False, dtype=str)
        data_frame.set_index(data_frame.columns[0], inplace=True)
        data_frames.append(data_frame)

    data_frame = combine(data_frames)

    return (
        list(data_frame.index),
        list(data_frame.columns),
        data_frame.values.astype(np.float64),
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
    def is_finite(self) -> bool:
        return bool(
            np.isfinite(self.phenotypes.to_numpy()).all()
            and np.isfinite(self.covariates.to_numpy()).all()
        )

    def copy(self) -> Self:
        phenotypes = self.phenotypes.to_numpy()
        covariates = self.covariates.to_numpy()
        vc = self.from_arrays(
            self.samples.copy(),
            self.phenotype_names.copy(),
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

    def subset_phenotypes(self, phenotype_names: list[str]) -> None:
        if phenotype_names == self.phenotype_names:
            # Nothing to do.
            return

        phenotype_indices = [
            self.phenotype_names.index(name) for name in phenotype_names
        ]
        self.phenotype_names = phenotype_names

        new_phenotypes = self.phenotypes.to_numpy()[:, phenotype_indices]
        self.phenotypes.free()
        self.phenotypes = SharedArray.from_numpy(
            new_phenotypes, self.phenotypes.sw, prefix="phenotypes"
        )
        logger.debug(
            f"Subsetting variable collection to have {self.phenotype_count} phenotypes"
        )

    def subset_samples(self, samples: list[str]) -> None:
        if samples == self.samples:
            # Nothing to do.
            return

        sw = self.phenotypes.sw

        sample_indices = [self.samples.index(sample) for sample in samples]
        self.samples = samples

        new_phenotypes = self.phenotypes.to_numpy()[sample_indices, :]
        self.phenotypes.free()
        self.phenotypes = SharedArray.from_numpy(
            new_phenotypes, sw, prefix="phenotypes"
        )

        new_covariates = self.covariates.to_numpy()[sample_indices, :]
        zero_variance: npt.NDArray[np.bool_] = np.isclose(
            new_covariates[:, 1:].var(axis=0), 0
        )
        if np.any(zero_variance):
            removed_covariates = [
                name
                for name, zero in zip(self.covariate_names[1:], zero_variance)
                if zero
            ]
            logger.debug(
                f"Removing covariates {removed_covariates} because they have zero "
                "variance after subsetting samples"
            )
            self.covariate_names = [
                name
                for name, zero in zip(self.covariate_names, zero_variance)
                if not zero
            ]
            new_covariates = new_covariates[:, ~zero_variance]

        self.covariates.free()
        self.covariates = SharedArray.from_numpy(
            new_covariates, sw, prefix="covariates"
        )

        logger.debug(
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
            raise ValueError(
                f"Unknown missing value strategy: {missing_value_strategy}"
            )

        samples = [sample for sample, c in zip(samples, criterion) if c]
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
        )

    @classmethod
    def from_txt(
        cls,
        phenotype_paths: list[Path],
        covariate_paths: list[Path],
        sw: SharedWorkspace,
        samples: list[str] | None = None,
        **kwargs,
    ) -> Self:
        phenotype_samples, phenotype_names, phenotype_array = read_and_combine(
            phenotype_paths
        )
        if samples is None:
            samples = phenotype_samples
        if samples is None:
            raise RuntimeError("No samples found in phenotype files")

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

    def covariance_to_txt(self, path: Path) -> None:
        phenotype_array = self.phenotypes.to_numpy()
        covariate_array = self.covariates.to_numpy()

        data_frame = pd.DataFrame(
            np.hstack((phenotype_array, covariate_array)),
            index=self.samples,
            columns=[
                *self.phenotype_names,
                *self.covariate_names,
            ],
        )
        data_frame.cov().to_csv(path, sep="\t")


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
    def from_array(cls, value: npt.NDArray[np.float64], **kwargs) -> Self:
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
            **kwargs,
        )
