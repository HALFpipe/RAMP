# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VariableCollection:
    samples: list[str]

    covariates: np.ndarray  # n x m
    phenotypes: np.ndarray  # n x k

    def __post_init__(self):
        # intercept
        first_column = self.covariates[:, 0, np.newaxis]
        if not np.allclose(first_column, 1):
            self.covariates = np.hstack(
                [np.ones_like(first_column), self.covariates]
            )

        # demean
        self.covariates[:, 1:] -= self.covariates[:, 1:].mean()

    @classmethod
    def from_txt(cls, path: Path) -> VariableCollection:
        data_array = np.loadtxt(path, dtype=str)

        columns = data_array[0, :]

        phenotypes = data_array[1:, np.char.isdigit(columns)].astype(float)
        covariates = data_array[1:, 1:][:, ~np.char.isdigit(columns)[1:]].astype(float)

        samples = list(data_array[1:, 0])

        return cls(samples, covariates, phenotypes)
