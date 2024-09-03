from dataclasses import dataclass
from functools import reduce
from typing import Self, override

import numpy as np
import pandas as pd
from numpy import typing as npt
from tqdm.auto import tqdm
from upath import UPath

from .log import logger
from .mem.arr import SharedArray
from .mem.wkspace import SharedWorkspace


def combine(data_frames: list[pd.DataFrame]) -> pd.DataFrame:
    return reduce(lambda a, b: a.combine_first(b), data_frames)


def read_and_combine(
    paths: list[UPath], noun: str = "files"
) -> tuple[list[str], list[str], npt.NDArray[np.float64]]:
    import pandas as pd

    data_frames: list[pd.DataFrame] = list()
    for path in tqdm(paths, desc=f"reading {noun}", unit="files"):
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
class VariableCollection(SharedArray[np.float64]):
    samples: list[str]

    phenotype_names: list[str]
    covariate_names: list[str]

    @override
    @staticmethod
    def get_prefix(**kwargs: str | int | None) -> str:
        return "vc"

    @property
    def covariate_count(self) -> int:
        return len(self.covariate_names)

    @property
    def phenotype_count(self) -> int:
        return len(self.phenotype_names)

    @property
    def covariates(self) -> npt.NDArray[np.float64]:
        a = self.to_numpy()
        return a[:, : self.covariate_count]

    @property
    def phenotypes(self) -> npt.NDArray[np.float64]:
        a = self.to_numpy()
        return a[:, self.covariate_count :]

    @property
    def sample_count(self) -> int:
        sample_count = self.shape[0]
        if sample_count != len(self.samples):
            raise ValueError
        return sample_count

    @property
    def covariate_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=self.covariates,
            copy=False,
            columns=self.covariate_names,
            index=self.samples,
        )

    @property
    def phenotype_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=self.phenotypes,
            copy=False,
            columns=self.phenotype_names,
            index=self.samples,
        )

    @property
    def data_frame(self) -> pd.DataFrame:
        return combine([self.covariate_frame, self.phenotype_frame])

    @property
    def is_finite(self) -> bool:
        return bool(np.isfinite(self.to_numpy()).all())

    @property
    def names(self) -> list[str]:
        return self.covariate_names + self.phenotype_names

    def __post_init__(self) -> None:
        if self.shape[1] != self.covariate_count + self.phenotype_count:
            raise ValueError(
                f"Expected {self.covariate_count} covariates and "
                f"{self.phenotype_count} phenotypes, but got {self.shape[1]} columns"
            )
        self.remove_zero_variance_covariates()

    def copy(
        self,
        phenotype_names: list[str] | None = None,
        samples: list[str] | None = None,
        **kwargs,
    ) -> Self:
        column_indices: list[int] = list(range(self.covariate_count))

        if phenotype_names is None:
            phenotype_names = self.phenotype_names.copy()
        for name in phenotype_names:
            column_indices.append(
                self.covariate_count + self.phenotype_names.index(name)
            )

        if samples is None:
            samples = self.samples.copy()
        row_indices = [self.samples.index(sample) for sample in samples]

        array = self.to_numpy()
        vc = self.from_numpy(
            array[np.ix_(row_indices, column_indices)],
            self.sw,
            samples=samples,
            phenotype_names=phenotype_names,
            covariate_names=self.covariate_names,
            **kwargs,
        )
        return vc

    def remove_zero_variance_covariates(self) -> None:
        mask: npt.NDArray[np.bool_] = np.isclose(self.covariates.var(axis=0), 0)
        mask[0] = False  # Do not remove intercept

        if not np.any(mask):
            return

        (covariate_indices,) = np.where(np.logical_not(mask))

        removed_covariates = [
            name
            for i, name in enumerate(self.covariate_names)
            if i not in covariate_indices
        ]
        logger.debug(
            f"Removing covariates {removed_covariates} because they have zero variance"
        )

        self.covariate_names = [self.covariate_names[i] for i in covariate_indices]

        phenotype_indices = np.arange(self.covariate_count, self.shape[1])
        keep = np.concatenate([covariate_indices, phenotype_indices], axis=0)
        self.compress(keep, axis=1)

    def subset_samples(self, samples: list[str]) -> None:
        if samples == self.samples:
            # Nothing to do.
            return

        sample_indices = [self.samples.index(sample) for sample in samples]
        self.compress(np.asarray(sample_indices), axis=0)
        self.samples = samples

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

        array = np.concatenate([covariates, phenotypes], axis=1)

        logger.debug(
            f"Creating variable collection with {len(samples)} samples, "
            f"{covariates.shape[1]} covariates, and {phenotypes.shape[1]} phenotypes."
        )

        return cls.from_numpy(
            array,
            sw,
            samples=samples,
            phenotype_names=phenotype_names,
            covariate_names=covariate_names,
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
            phenotype_paths, noun="phenotypes"
        )
        if samples is None:
            samples = phenotype_samples
        if not samples:
            raise RuntimeError("No samples found in phenotype files")

        logger.debug("Reading covariates")
        covariate_samples, covariate_names, covariate_array = read_and_combine(
            covariate_paths, noun="covariates"
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

        return cls.from_arrays(
            samples=samples,
            phenotype_names=phenotype_names,
            phenotypes=phenotype_array,
            covariate_names=covariate_names,
            covariates=covariate_array,
            sw=sw,
            **kwargs,
        )


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
