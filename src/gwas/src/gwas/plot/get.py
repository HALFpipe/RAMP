from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy import typing as npt
from tqdm.auto import tqdm

from gwas.mem.data_frame import SharedDataFrame

from ..compression.arr.base import FileArrayReader
from ..hg19 import offset
from ..log import logger
from ..mem.arr import ScalarType, SharedArray
from ..mem.wkspace import SharedWorkspace
from ..utils.genetics import make_variant_mask
from ..utils.multiprocessing import (
    IterationOrder,
    make_pool_or_null_context,
)
from .resolve import Phenotype, ScoreFile


@dataclass(frozen=True)
class LoadPValueJob:
    row_offset: int
    row_count: int
    column_indices: npt.NDArray[np.uint32]
    reader: FileArrayReader[np.float64]
    data_array: SharedArray
    mask_array: SharedArray[np.bool_]
    num_threads: int = 1

    def subset_array(
        self, shared_array: SharedArray[ScalarType], column_count: int
    ) -> npt.NDArray[ScalarType]:
        numpy_array = shared_array.to_numpy().transpose()
        numpy_array = numpy_array[
            self.row_offset : (self.row_offset + self.row_count), :
        ]
        numpy_array = numpy_array[:, :column_count]
        return numpy_array

    @property
    def data(self) -> npt.NDArray[np.float64]:
        column_count = self.column_indices.size
        return self.subset_array(self.data_array, column_count)

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        column_count = self.column_indices.size // 2
        return self.subset_array(self.mask_array, column_count)


def calculate_chi_squared_p_value(
    u_stat: npt.NDArray[np.float64], v_stat: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    import scipy.stats

    """
    Calculates the p-value from U-statistic and V-statistic using the chi-square test
    """

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
    u_stat[:] = scipy.stats.distributions.chi2.sf(v_stat, df=1)
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

    return u_stat, v_stat


def load_score_file(job: LoadPValueJob) -> None:
    try:
        logger.debug(
            f"Loading {job.row_count} rows and {len(job.column_indices)} columns "
            f"from {job.reader.file_path}"
        )

        data = job.data
        row_indices = np.arange(job.row_count, dtype=np.uint32)
        with job.reader:
            job.reader.read_indices(
                row_indices=row_indices, column_indices=job.column_indices, array=data
            )

        u_stat = data[:, ::2]
        v_stat = data[:, 1::2]

        # Update the mask in place
        mask = job.mask
        np.logical_and(~np.isclose(u_stat, 0), mask, out=mask)

        calculate_chi_squared_p_value(u_stat, v_stat)
    except Exception as e:
        logger.error(f'Error while loading "{job.reader.file_path}"', exc_info=e)
        raise e


@dataclass
class PlotJob:
    name: str
    phenotype_index: int


copies: int = 2  # We need to store u_stat and v_stat


@dataclass
class DataLoader:
    phenotypes: list[Phenotype]
    score_files: list[ScoreFile]
    variant_metadata: SharedDataFrame | None
    sw: SharedWorkspace

    r_squared_cutoff: float
    minor_allele_frequency_cutoff: float

    num_threads: int = 1

    chromosome_array: SharedArray[np.uint8] = field(init=False)
    position_array: SharedArray[np.int64] = field(init=False)
    data_array: SharedArray[np.float64] = field(init=False)
    mask_array: SharedArray[np.bool_] = field(init=False)

    variant_count: int = field(init=False)
    phenotype_count: int = field(init=False)
    row_offsets: list[int] = field(init=False)

    @property
    def chromosome(self) -> npt.NDArray[np.uint8]:
        return self.chromosome_array.to_numpy()

    @property
    def position(self) -> npt.NDArray[np.int64]:
        return self.position_array.to_numpy()

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
            "data", copies * self.phenotype_count, self.variant_count, dtype=np.float64
        )
        self.mask_array = self.sw.alloc(
            "mask_array", self.phenotype_count, self.variant_count, dtype=np.bool_
        )

    def __post_init__(self) -> None:
        if self.variant_metadata is None:
            raise ValueError("Variant metadata is required to initialize data loader")

        self.chromosome_array = self.variant_metadata["chromosome_int"].values  # noqa: PD011

        self.variant_count, _ = self.variant_metadata.shape
        variant_metadata = self.variant_metadata.to_pandas()

        sw = self.sw
        self.position_array = sw.alloc("position", self.variant_count, dtype=np.int64)
        self.position[:] = offset[self.chromosome - 1] + variant_metadata.position
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
        if self.variant_metadata is None:
            raise ValueError("Variant metadata is required to generate chunks")

        mask = self.mask_array.to_numpy().transpose()
        for i, phenotype in enumerate(chunk):
            mask[:, i] = make_variant_mask(
                allele_frequencies=self.variant_metadata[
                    f"{phenotype.variable_collection_name}_alternate_allele_frequency"
                ].to_pandas(),
                r_squared=self.variant_metadata["r_squared"].to_pandas(),
                minor_allele_frequency_cutoff=self.minor_allele_frequency_cutoff,
                r_squared_cutoff=self.r_squared_cutoff,
            )

        column_indices = np.asarray(
            sum(
                (
                    [phenotype.u_stat_index, phenotype.v_stat_index]
                    for phenotype in chunk
                ),
                [],
            ),
            dtype=np.uint32,
        )

        jobs: list[LoadPValueJob] = [
            LoadPValueJob(
                row_offset=row_offset,
                row_count=score_file.variant_count,
                column_indices=column_indices,
                reader=score_file.reader,
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

            try:
                # need to yield the generator so that we only continue
                # after all the array data has been processed, so we
                # don't overwrite data that is still being read
                yield self.generate_chunk(chunk)
            except Exception as e:
                logger.error("Error while processing chunk", exc_info=e)
