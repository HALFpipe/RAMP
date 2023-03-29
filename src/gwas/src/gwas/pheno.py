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
        return cls(
            samples,
            phenotype_names,
            SharedArray.from_array(phenotypes, sw, prefix="phenotypes"),
            covariate_names,
            SharedArray.from_array(covariates, sw, prefix="covariates"),
        )

    @classmethod
    def from_txt(cls, phenotype_path: Path, covariate_path: Path, sw: SharedWorkspace):
        phenotypes_array = np.loadtxt(phenotype_path, dtype=object)
        phenotype_names = phenotypes_array[0, 1:].tolist()
        samples = phenotypes_array[1:, 0].tolist()
        phenotypes = phenotypes_array[1:, 1:].astype(np.float64)

        covariates_array = np.loadtxt(covariate_path, dtype=object)
        covariate_names = covariates_array[0, 1:].tolist()
        if samples != covariates_array[1:, 0].tolist():
            raise ValueError(
                "Samples do not match between phenotype and covariate files"
            )
        covariates = covariates_array[1:, 1:].astype(np.float64)

        return cls.from_arrays(
            samples,
            phenotype_names,
            phenotypes,
            covariate_names,
            covariates,
            sw,
        )
