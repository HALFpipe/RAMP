# -*- coding: utf-8 -*-

from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Self

import numpy as np
from numpy import typing as npt

from gwas.log import logger
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace


def read_and_combine(paths: list[Path]) -> tuple[list[str], list[str], npt.NDArray]:
    import pandas as pd

    data_frames: list[pd.DataFrame] = list()
    for path in paths:
        data_frame = pd.read_table(path, sep="\t", header=0, index_col=False, dtype=str)
        data_frame.set_index(data_frame.columns[0], inplace=True)
        data_frames.append(data_frame)

    data_frame = reduce(lambda a, b: a.combine_first(b), data_frames)

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
    def is_finite(self) -> bool:
        return bool(
            np.isfinite(self.phenotypes.to_numpy()).all()
            and np.isfinite(self.covariates.to_numpy()).all()
        )

    def free(self):
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
        zero_variance = np.isclose(np.var(new_covariates, axis=0), 0)
        if np.any(zero_variance):
            removed_covariates = [
                name for name, zero in zip(self.covariate_names, zero_variance) if zero
            ]
            logger.warning(
                f"Removing covariates with zero variance {removed_covariates} "
                "while subsetting samples"
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

        logger.info(
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
            raise RuntimeError("No samples found in phenotype files.")

        covariate_samples, covariate_names, covariate_array = read_and_combine(
            covariate_paths
        )

        # Use only samples that are present in all files.
        sample_indices = [
            i
            for i, sample in enumerate(phenotype_samples)
            if sample in samples and sample in covariate_samples
        ]
        phenotype_samples = [phenotype_samples[i] for i in sample_indices]
        phenotype_array = phenotype_array[sample_indices, :]

        # Ensure that covariates are in the same order as phenotypes.
        sample_indices = [
            covariate_samples.index(sample)
            for sample in phenotype_samples
            if sample in samples and sample in covariate_samples
        ]
        covariate_samples = [covariate_samples[i] for i in sample_indices]
        covariate_array = covariate_array[sample_indices, :]

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
