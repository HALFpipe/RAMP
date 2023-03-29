# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import typing as npt

from gwas.mem.wkspace import SharedWorkspace

from .mem.arr import SharedArray


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

    def free(self):
        self.phenotypes.free()
        self.covariates.free()

    @classmethod
    def from_arrays(
        cls,
        samples: list[str],
        phenotype_names: list[str],
        phenotypes: npt.NDArray[np.float64],
        covariate_names: list[str],
        covariates: npt.NDArray[np.float64],
        sw: SharedWorkspace,
    ):
        # Add intercept if not present.
        first_column = covariates[:, 0, np.newaxis]
        if not np.allclose(first_column, 1):
            covariates = np.hstack([np.ones_like(first_column), covariates])
            covariate_names = ["intercept"] + covariate_names

        return cls(
            samples,
            phenotype_names,
            SharedArray.from_array(phenotypes, sw, prefix="phenotypes"),
            covariate_names,
            SharedArray.from_array(covariates, sw, prefix="covariates"),
        )

    @classmethod
    def from_txt(
        cls,
        phenotype_path: Path,
        covariate_path: Path,
        sw: SharedWorkspace,
        samples: list[str] | None = None,
    ):
        phenotypes_array = np.loadtxt(phenotype_path, dtype=object)
        phenotype_names = phenotypes_array[0, 1:]
        phenotype_samples = phenotypes_array[1:, 0].tolist()
        phenotypes = phenotypes_array[1:, 1:].astype(np.float64)

        if samples is None:
            samples = phenotype_samples.tolist()
        if samples is None:
            raise RuntimeError

        sample_indices = [phenotype_samples.index(sample) for sample in samples]
        phenotypes = phenotypes[sample_indices, :]

        covariates_array = np.loadtxt(covariate_path, dtype=object)
        covariate_names = covariates_array[0, 1:]
        covariate_samples = covariates_array[1:, 0].tolist()
        covariates = covariates_array[1:, 1:].astype(np.float64)

        sample_indices = [covariate_samples.index(sample) for sample in samples]
        covariates = covariates[sample_indices, :]

        # Remove samples with missing values.
        non_missing = np.isfinite(phenotypes).all(axis=1)
        non_missing &= np.isfinite(covariates).all(axis=1)
        samples = [sample for sample, n in zip(samples, non_missing) if n]
        phenotypes = phenotypes[non_missing, :]
        covariates = covariates[non_missing, :]

        if samples is None:
            raise RuntimeError

        return cls.from_arrays(
            samples,
            phenotype_names.tolist(),
            phenotypes,
            covariate_names.tolist(),
            covariates,
            sw,
        )
