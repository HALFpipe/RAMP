# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
from numpy import typing as npt
from tqdm.auto import tqdm

from ..compression.arr._get import get_orthogonal_selection
from ..log import logger
from ..mem.arr import ScalarType, SharedArray
from ..mem.wkspace import SharedWorkspace
from ..utils import (
    IterationOrder,
    make_pool_or_null_context,
    make_variant_mask,
)
from .hg19 import offset
from .resolve import Phenotype, ScoreFile


@dataclass(frozen=True)
class LoadPValueJob:
    row_offset: int
    row_count: int
    column_indices: npt.NDArray[np.int64]
    urlpath: bytes
    data_array: SharedArray
    mask_array: SharedArray[np.bool_]
    num_threads: int = 1

    def subset_array(
        self, shared_array: SharedArray[ScalarType]
    ) -> npt.NDArray[ScalarType]:
        numpy_array = shared_array.to_numpy().transpose()
        numpy_array = numpy_array[
            self.row_offset : (self.row_offset + self.row_count), :
        ]
        column_count = self.column_indices.size
        numpy_array = numpy_array[:, :column_count]
        return numpy_array

    @property
    def data(self) -> npt.NDArray[np.float64]:
        return self.subset_array(self.data_array)

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        return self.subset_array(self.mask_array)


def calculate_chi_squared_p_value(
    u_stat: npt.NDArray[np.float64], v_stat: npt.NDArray[np.float64]
) -> None:
    from scipy.stats.distributions import chi2

    """
    Calculates the p-value from U-statistic and V-statistic using the chi-square test
    """
    logger.debug(f"Calculating chi-squared p-value for {u_stat.size} tests")

    # Find which inputs are invalid
    invalid_u_stat = np.logical_or(np.isnan(u_stat), np.isinf(u_stat))
    invalid_v_stat = np.logical_or(np.isnan(v_stat), np.isinf(v_stat))
    invalid_v_stat = np.logical_or(invalid_v_stat, np.isclose(v_stat, 0))
    invalid_stat = np.logical_or(invalid_u_stat, invalid_v_stat)
    if np.any(invalid_stat):
        logger.warning(
            f"Got {np.count_nonzero(invalid_stat)} invalid stats for u "
            f"{u_stat[invalid_stat]} and v {v_stat[invalid_stat]}"
        )

    # Calculate chi-squared statistic in place
    np.square(u_stat, out=u_stat, where=np.logical_not(invalid_stat))
    np.true_divide(u_stat, v_stat, out=v_stat, where=np.logical_not(invalid_stat))
    v_stat[invalid_stat] = 0

    # Calculate p-value and log-p-value
    u_stat[:] = chi2.sf(v_stat, df=1)
    u_stat[invalid_stat] = np.nan

    invalid_p_value = np.logical_or(np.isnan(u_stat), np.isinf(u_stat))
    invalid_p_value = np.logical_or(invalid_p_value, u_stat < 0)
    invalid_p_value = np.logical_and(np.logical_not(invalid_stat), invalid_p_value)
    if np.any(invalid_p_value):
        logger.warning(
            f"Got {np.count_nonzero(invalid_p_value)} invalid p-values "
            f"{u_stat[invalid_p_value]} for chi squared values {v_stat[invalid_p_value]}"
        )

    invalid_mask = np.logical_or(invalid_stat, invalid_p_value)
    np.log10(u_stat, out=v_stat, where=np.logical_not(invalid_mask))
    v_stat[invalid_mask] = np.nan

    invalid_log_p_value = np.logical_or(np.isnan(v_stat), np.isinf(v_stat))
    invalid_log_p_value = np.logical_or(invalid_log_p_value, v_stat > 1)
    invalid_log_p_value = np.logical_and(
        np.logical_not(invalid_mask), invalid_log_p_value
    )
    if np.any(invalid_log_p_value):
        logger.warning(
            f"Got {np.count_nonzero(invalid_log_p_value)} invalid log p-values "
            f"{v_stat[invalid_mask]} for p-values values {u_stat[invalid_mask]}"
        )


def load_score_file(job: LoadPValueJob) -> None:
    logger.debug(
        f"Loading {job.row_count} rows and {len(job.column_indices)} columns "
        f"from {job.urlpath.decode()} with b2nd_get_orthogonal_selection"
    )

    data = job.data
    row_indices = np.arange(job.row_count, dtype=np.int64)
    get_orthogonal_selection(
        urlpath=job.urlpath,
        row_indices=row_indices,
        column_indices=job.column_indices,
        array=data,
        num_threads=job.num_threads,
    )

    u_stat = data[:, ::2]
    v_stat = data[:, 1::2]

    # Update the mask in place
    mask = job.mask
    np.logical_and(~np.isclose(u_stat, 0), mask, out=mask)

    calculate_chi_squared_p_value(u_stat, v_stat)


@dataclass
class PlotJob:
    name: str
    phenotype_index: int


copies: int = 2  # We need to store u_stat and v_stat


@dataclass
class DataLoader:
    phenotypes: list[Phenotype]
    score_files: list[ScoreFile]
    variant_metadata: pd.DataFrame
    sw: SharedWorkspace

    r_squared_cutoff: float
    minor_allele_frequency_cutoff: float

    num_threads: int = 1

    chromosome_array: SharedArray[np.int64] = field(init=False)
    position_array: SharedArray[np.int64] = field(init=False)
    data_array: SharedArray = field(init=False)
    mask_array: SharedArray[np.bool_] = field(init=False)

    variant_count: int = field(init=False)
    phenotype_count: int = field(init=False)
    row_offsets: list[int] = field(init=False)

    @property
    def chromosome(self) -> npt.NDArray[np.int64]:
        return self.chromosome_array.to_numpy()

    def init_chromosome_array(self) -> None:
        self.chromosome_array = self.sw.alloc(
            "chromosome", self.variant_count, dtype=np.int64
        )
        self.chromosome[:] = self.variant_metadata.chromosome_int

    @property
    def position(self) -> npt.NDArray[np.int64]:
        return self.position_array.to_numpy()

    def init_position_array(self) -> None:
        self.position_array = self.sw.alloc(
            "position", self.variant_count, dtype=np.int64
        )
        self.position[:] = offset[self.chromosome - 1] + self.variant_metadata.position

    def init_data_array(self) -> None:
        row_index = 0
        self.row_offsets: list[int] = []
        for score_file in self.score_files:
            self.row_offsets.append(row_index)
            row_count = score_file.variant_count
            row_index += row_count

        item_size = np.float64().itemsize * copies + np.bool_().itemsize
        per_phenotype_size = item_size * self.variant_count
        max_phenotype_count = self.sw.unallocated_size // per_phenotype_size
        self.phenotype_count = min(len(self.phenotypes), max_phenotype_count)

        self.data_array = self.sw.alloc(
            "data", copies * self.phenotype_count, self.variant_count
        )
        self.mask_array = self.sw.alloc(
            "mask_array", self.phenotype_count, self.variant_count, dtype=np.bool_
        )

    def __post_init__(self) -> None:
        self.variant_count = len(self.variant_metadata.index)

        self.init_chromosome_array()
        self.init_position_array()
        self.init_data_array()

    def free(self) -> None:
        self.chromosome_array.free()
        self.position_array.free()
        self.data_array.free()
        self.mask_array.free()

    def generate_chunk(
        self,
        chunk: list[Phenotype],
    ) -> Iterator[PlotJob]:
        mask = self.mask_array.to_numpy().transpose()
        for i, phenotype in enumerate(chunk):
            mask[:, i] = make_variant_mask(
                self.variant_metadata[
                    f"{phenotype.variable_collection_name}_alternate_allele_frequency"
                ],
                self.variant_metadata.r_squared,
                self.minor_allele_frequency_cutoff,
                self.r_squared_cutoff,
            )

        column_indices = np.asarray(
            sum(
                (
                    [phenotype.u_stat_index, phenotype.v_stat_index]
                    for phenotype in chunk
                ),
                [],
            ),
            dtype=np.int64,
        )

        jobs: list[LoadPValueJob] = [
            LoadPValueJob(
                row_offset=row_offset,
                row_count=score_file.variant_count,
                column_indices=column_indices,
                urlpath=str(score_file.path).encode(),
                data_array=self.data_array,
                mask_array=self.mask_array,
                num_threads=self.num_threads,
            )
            for row_offset, score_file in zip(
                self.row_offsets, self.score_files, strict=True
            )
        ]
        pool, iterator = make_pool_or_null_context(
            jobs,
            load_score_file,
            num_threads=self.num_threads,
            iteration_order=IterationOrder.UNORDERED,
        )
        with pool:
            for _ in tqdm(
                iterator,
                total=len(jobs),
                unit="chromosomes",
                desc="loading summary statistics",
                leave=False,
            ):
                pass

        for i, phenotype in enumerate(chunk):
            yield PlotJob(
                name=phenotype.name,
                phenotype_index=i,
            )

    def run(self) -> Iterator[Iterator[PlotJob]]:
        phenotypes = self.phenotypes.copy()

        while phenotypes:
            chunk = phenotypes[: self.phenotype_count]
            phenotypes = phenotypes[self.phenotype_count :]

            yield self.generate_chunk(chunk)
