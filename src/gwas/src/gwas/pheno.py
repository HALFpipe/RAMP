# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

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
        # add intercept if not present
        first_column = covariates[:, 0, np.newaxis]
        if not np.allclose(first_column, 1):
            covariates = np.hstack([np.ones_like(first_column), covariates])

        # subtract column mean from covariates
        covariates[:, 1:] -= covariates[:, 1:].mean(axis=0)

        vc = cls(
            samples,
            phenotype_names,
            SharedArray.from_array(phenotypes, sw, prefix="phenotypes"),
            covariate_names,
            SharedArray.from_array(covariates, sw, prefix="covariates"),
        )

        return vc
