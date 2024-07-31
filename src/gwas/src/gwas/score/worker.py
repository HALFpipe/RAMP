from dataclasses import dataclass, field
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event

import numpy as np
import scipy
from numpy import typing as npt

from ..compression.arr.base import FileArrayWriter
from ..eig.collection import EigendecompositionCollection
from ..log import logger, multiprocessing_context
from ..mem.arr import SharedArray
from ..utils import Action, Process, SharedState
from ..vcf.base import VCFFile
from .calc import calc_u_stat, calc_v_stat


@dataclass
class TaskProgress:
    variant_count: int


@dataclass(kw_only=True)
class TaskSyncCollection(SharedState):
    job_count: int
    # Genotypes array can be overwritten by fresh data
    can_read: Event = field(default_factory=multiprocessing_context.Event)
    # Passes the number of variants that were read to the calculation process
    read_count_queue: Queue[int] = field(default_factory=multiprocessing_context.Queue)
    # Indicates that writing has finished and we can calculate
    can_calc: list[Event] = field(init=False)
    # Passes the number of variants that have finished calculating to the writer
    # process
    calc_count_queue: Queue[int] = field(default_factory=multiprocessing_context.Queue)
    # Indicates that calculation has finished and we can write out the results
    can_write: list[Event] = field(init=False)

    # Passes the current progress to the main process
    progress_queue: Queue[TaskProgress] = field(
        default_factory=multiprocessing_context.Queue
    )

    def __post_init__(self) -> None:
        self.can_calc = [multiprocessing_context.Event() for _ in range(self.job_count)]
        self.can_write = [multiprocessing_context.Event() for _ in range(self.job_count)]


class Worker(Process):
    def __init__(self, t: TaskSyncCollection, num_threads: int | None) -> None:
        self.t = t
        super().__init__(t.exception_queue, num_threads)


class GenotypeReader(Worker):
    def __init__(
        self,
        t: TaskSyncCollection,
        vcf_file: VCFFile,
        genotypes_array: SharedArray,
    ) -> None:
        if genotypes_array.shape[0] != vcf_file.sample_count:
            raise ValueError(
                "The row count of the genotype array does not match the VCF file."
            )

        self.vcf_file = vcf_file
        self.genotypes_array = genotypes_array

        super().__init__(t, num_threads=None)

    def func(self) -> None:
        variant_indices = self.vcf_file.variant_indices.copy()
        variant_count = self.genotypes_array.shape[1]

        sample_count = self.vcf_file.sample_count

        with self.vcf_file:
            while len(variant_indices) > 0:
                # Make sure the genotypes array is not in use
                action = self.t.wait(self.t.can_read)
                if action is Action.EXIT:
                    break
                self.t.can_read.clear()
                # Read the genotypes
                self.vcf_file.variant_indices = variant_indices[:variant_count]
                genotypes: npt.NDArray[np.float64] = self.genotypes_array.to_numpy(
                    shape=(sample_count, self.vcf_file.variant_count)
                )
                logger.debug("Reading genotypes")
                self.vcf_file.read(genotypes.transpose())
                # Pass how many variants were read to the calculation process
                self.t.read_count_queue.put_nowait(
                    int(self.vcf_file.variant_indices.size)
                )
                # Remove already read variant indices
                variant_indices = variant_indices[variant_count:]
                if variant_indices.size == 0:
                    # Signal that we are done
                    self.t.read_count_queue.put_nowait(0)
                    # Exit the process
                    logger.debug("Genotype reader has finished and will exit")
                    break


class Calc(Worker):
    def __init__(
        self,
        t: TaskSyncCollection,
        genotypes_array: SharedArray,
        ec: EigendecompositionCollection,
        rotated_genotypes_array: SharedArray,
        inverse_variance_arrays: list[SharedArray],
        scaled_residuals_arrays: list[SharedArray],
        stat_array: SharedArray,
        num_threads: int,
    ) -> None:
        self.genotypes_array = genotypes_array
        self.rotated_genotypes_array = rotated_genotypes_array
        self.ec = ec
        self.inverse_variance_arrays = inverse_variance_arrays
        self.scaled_residuals_arrays = scaled_residuals_arrays
        self.stat_array = stat_array

        super().__init__(t, num_threads)

    def func(self) -> None:
        eigenvector_matrices = [
            eigenvector_array.to_numpy()
            for eigenvector_array in self.ec.eigenvector_arrays
        ]
        inverse_variance_matrices = [
            inverse_variance_array.to_numpy()
            for inverse_variance_array in self.inverse_variance_arrays
        ]
        scaled_residuals_matrices = [
            scaled_residuals_array.to_numpy()
            for scaled_residuals_array in self.scaled_residuals_arrays
        ]

        phenotype_counts = np.fromiter(
            (
                inverse_variance_matrix.shape[1]
                for inverse_variance_matrix in inverse_variance_matrices
            ),
            dtype=int,
        )
        phenotype_indices = np.cumsum(phenotype_counts)
        phenotype_slices = [
            slice(start, end)
            for start, end in zip(
                np.concatenate(([0], phenotype_indices[:-1])),
                phenotype_indices,
                strict=True,
            )
        ]

        job_count = len(self.ec.eigenvector_arrays)

        while True:
            logger.debug("Waiting for the reader to finish reading")
            value = self.t.get(self.t.read_count_queue)
            if value is Action.EXIT:
                break
            elif not isinstance(value, int):
                raise ValueError("Expected an integer.")
            variant_count = value
            self.t.calc_count_queue.put_nowait(variant_count)
            if value == 0:
                # Exit the process
                return

            sample_count = self.genotypes_array.shape[0]
            genotypes = self.genotypes_array.to_numpy(
                shape=(sample_count, variant_count)
            )

            (_, phenotype_count, _) = self.stat_array.shape
            stat = self.stat_array.to_numpy(shape=(2, phenotype_count, variant_count))

            for i in range(job_count):
                can_calc = self.t.can_calc[i]
                eigenvector_matrix = eigenvector_matrices[i]
                inverse_variance_matrix = inverse_variance_matrices[i]
                scaled_residuals_matrix = scaled_residuals_matrices[i]
                phenotype_slice = phenotype_slices[i]
                can_write = self.t.can_write[i]

                (sample_count, _) = scaled_residuals_matrix.shape
                rotated_genotypes = self.rotated_genotypes_array.to_numpy(
                    shape=(sample_count, variant_count)
                )
                logger.debug("Multiplying the genotypes with the eigenvectors")
                scipy.linalg.blas.dgemm(
                    alpha=1,
                    a=eigenvector_matrix,
                    b=genotypes,
                    beta=0,
                    c=rotated_genotypes,
                    trans_a=True,
                    trans_b=False,
                    overwrite_c=True,
                )
                logger.debug("Subtracting the rotated mean from the rotated genotypes")
                sample_boolean_vector = self.ec.sample_boolean_vectors[i]
                mean = genotypes.mean(
                    axis=0,
                    where=sample_boolean_vector[:, np.newaxis],
                )
                scipy.linalg.blas.dger(
                    alpha=-1,
                    x=eigenvector_matrix.sum(axis=0),
                    y=mean,
                    a=rotated_genotypes,
                    overwrite_a=True,
                    overwrite_y=False,
                )
                if i == job_count - 1:
                    logger.debug("Allow the reader to read the next batch of variants")
                    self.t.can_read.set()

                logger.debug("Waiting for the writer to finish writing")
                action = self.t.wait(can_calc)
                if action is Action.EXIT:
                    break
                can_calc.clear()

                # Prepare arrays
                u_stat = stat[0, phenotype_slice, :].transpose()
                v_stat = stat[1, phenotype_slice, :].transpose()

                # Calculate the score statistics
                logger.debug(f"Calculating U statistic for phenotypes {phenotype_slice}")
                calc_u_stat(scaled_residuals_matrix, rotated_genotypes, u_stat)

                logger.debug("Squaring genotypes")
                np.square(rotated_genotypes, out=rotated_genotypes)

                logger.debug("Calculating V statistic")
                calc_v_stat(inverse_variance_matrix, rotated_genotypes, u_stat, v_stat)

                # Signal that calculation has finished
                can_write.set()


class ScoreWriter(Worker):
    def __init__(
        self,
        t: TaskSyncCollection,
        stat_array: SharedArray,
        stat_file_array: FileArrayWriter[np.float64],
        phenotype_offset: int,
        variant_offset: int,
    ) -> None:
        self.stat_array = stat_array
        self.stat_file_array = stat_file_array

        self.phenotype_offset = phenotype_offset
        self.variant_offset = variant_offset

        super().__init__(t, num_threads=None)

    def func(self) -> None:
        job_count = len(self.t.can_write)
        (_, phenotype_count, _) = self.stat_array.shape
        phenotype_slice = slice(
            2 * self.phenotype_offset, 2 * (self.phenotype_offset + phenotype_count)
        )
        variant_index = self.variant_offset

        with self.stat_file_array:
            while True:
                logger.debug("Wait for the calculation to start")
                value = self.t.get(self.t.calc_count_queue)
                if value is Action.EXIT:
                    break
                elif not isinstance(value, int):
                    raise ValueError("Expected an integer.")
                variant_count = value
                if variant_count == 0:
                    # Exit the process
                    break

                logger.debug("Wait for the calculation to finish")
                for i in range(job_count):
                    can_write = self.t.can_write[i]
                    action = self.t.wait(can_write)
                    if action is Action.EXIT:
                        break
                    can_write.clear()

                # Write the data
                variant_slice = slice(variant_index, variant_index + variant_count)
                stat = self.stat_array.to_numpy(
                    shape=(2, phenotype_count, variant_count)
                )
                logger.debug(
                    f"Writing variants {variant_slice} to {type(self.stat_file_array)} "
                    f"with shape {stat.transpose().shape}"
                )
                two_dimensional_stat = stat.transpose().reshape(
                    (variant_count, 2 * phenotype_count)
                )
                self.stat_file_array[variant_slice, phenotype_slice] = (
                    two_dimensional_stat
                )

                # Allow the calculation to continue
                for i in range(job_count):
                    can_calc = self.t.can_calc[i]
                    can_calc.set()

                variant_index += variant_count
                self.t.progress_queue.put_nowait(TaskProgress(variant_count))
