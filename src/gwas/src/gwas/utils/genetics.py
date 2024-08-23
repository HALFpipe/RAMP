import numpy as np
import pandas as pd
from numpy import typing as npt


def parse_chromosome(chromosome: str) -> int | str:
    if chromosome == "X":
        return chromosome
    elif chromosome.isdigit():
        return int(chromosome)
    else:
        raise ValueError(f'Unknown chromosome "{chromosome}"')


def chromosome_to_int(chromosome: int | str) -> int:
    if chromosome == "X":
        return 23
    elif isinstance(chromosome, str) and chromosome.isdigit():
        return int(chromosome)
    elif isinstance(chromosome, int):
        return chromosome
    raise ValueError(f'Unknown chromsome "{chromosome}"')


def chromosome_from_int(chromosome_int: int) -> int | str:
    if chromosome_int == 23:
        return "X"
    else:
        return chromosome_int


def chromosomes_list() -> list[int | str]:
    return [*range(1, 22 + 1), "X"]


def chromosomes_set() -> set[int | str]:
    return set(chromosomes_list())


def greater_or_close(series: pd.Series, cutoff: float) -> npt.NDArray[np.bool_]:
    value = np.asarray(series.values)
    result = np.logical_or(
        np.logical_or(value >= cutoff, np.isclose(value, cutoff)), np.isnan(value)
    )
    return result


def make_variant_mask(
    allele_frequencies: pd.Series | pd.DataFrame,
    r_squared: pd.Series,
    minor_allele_frequency_cutoff: float,
    r_squared_cutoff: float,
    aggregate_func: str = "max",
) -> npt.NDArray[np.bool_]:
    allele_frequencies = allele_frequencies.copy()
    allele_frequencies = allele_frequencies.where(
        allele_frequencies.to_numpy() <= 0.5, 1 - allele_frequencies
    )
    if isinstance(allele_frequencies, pd.DataFrame):
        allele_frequencies = allele_frequencies.aggregate(aggregate_func, axis="columns")
    variant_mask = greater_or_close(
        allele_frequencies,
        minor_allele_frequency_cutoff,
    ) & greater_or_close(r_squared, r_squared_cutoff)
    return variant_mask
