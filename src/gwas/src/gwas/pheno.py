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

    phenotypes: SharedArray
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

    @classmethod
    def from_arrays(
        cls,
        samples: list[str],
        covariates: npt.NDArray[np.float64],
        phenotypes: npt.NDArray[np.float64],
        sw: SharedWorkspace,
    ):
        # add intercept if not present
        first_column = covariates[:, 0, np.newaxis]
        if not np.allclose(first_column, 1):
            covariates = np.hstack([np.ones_like(first_column), covariates])

        # subtract column mean from covariates
        covariates[:, 1:] -= covariates[:, 1:].mean(axis=0)

        vc = cls(
            samples,
            SharedArray.from_array(phenotypes, sw, prefix="phenotypes"),
            SharedArray.from_array(covariates, sw, prefix="covariates"),
        )

        return vc

    @classmethod
    def from_txt(cls, path: Path, sw: SharedWorkspace):
        data_array = np.loadtxt(path, dtype=str)
        columns = data_array[0, :]

        phenotypes = data_array[1:, np.char.isdigit(columns)].astype(float)
        covariates = data_array[1:, 1:][:, ~np.char.isdigit(columns)[1:]].astype(float)

        samples = list(data_array[1:, 0])

        return cls.from_arrays(samples, covariates, phenotypes, sw)
