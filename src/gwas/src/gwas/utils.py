# -*- coding: utf-8 -*-
from dataclasses import dataclass
from shutil import which

import numpy as np


def chromosome_to_int(chromosome: int | str) -> int:
    if chromosome == "X":
        return 23
    elif isinstance(chromosome, int):
        return chromosome
    raise ValueError(f'Unknown chromsome "{chromosome}"')


def chromosomes_set() -> set[int | str]:
    return set(range(1, 22 + 1)) | {"X"}


@dataclass
class MinorAlleleFrequencyCutoff:
    minor_allele_frequency_cutoff: float = 0.05

    def __call__(self, row) -> bool:
        mean = float(row.mean())
        minor_allele_frequency = mean / 2
        return not (
            (minor_allele_frequency < self.minor_allele_frequency_cutoff)
            or ((1 - minor_allele_frequency) < self.minor_allele_frequency_cutoff)
            or np.isclose(row.var(), 0)  # additional safety check
        )


def unwrap_which(command: str) -> str:
    executable = which(command)
    if executable is None:
        raise ValueError
    return executable
