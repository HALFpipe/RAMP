# -*- coding: utf-8 -*-

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from queue import Empty

import numpy as np
from numpy import typing as npt

from ..compression.arr import ArrayProxy
from ..eig import Eigendecomposition, EigendecompositionCollection
from ..log import logger
from ..mem.arr import SharedArray
from ..utils import Action, Process, SharedState
from ..vcf.base import VCFFile


def calc_u_stat(
    inverse_variance_scaled_residuals: npt.NDArray,
    rotated_genotypes: npt.NDArray,
    u_stat: npt.NDArray,
) -> None:
    logger.debug("Calculating numerator")
    u_stat[:] = rotated_genotypes.transpose() @ inverse_variance_scaled_residuals


def calc_v_stat(
    inverse_variance: npt.NDArray,
    squared_genotypes: npt.NDArray,
    u_stat: npt.NDArray,
    v_stat: npt.NDArray,
) -> npt.NDArray[np.bool_]:
    logger.debug("Calculating denominator")
    v_stat[:] = squared_genotypes.transpose() @ inverse_variance
    logger.debug("Zeroing invalid values")
    invalid = np.isclose(v_stat, 0)
    u_stat[invalid] = 0
    v_stat[invalid] = 1
    return invalid


@dataclass(kw_only=True)
class TaskSyncCollection(SharedState):
    job_count: int
    # Genotypes array can be overwritten by fresh data.
    can_read: Event = field(default_factory=mp.Event)
    # Passes the number of variants that were read to the calculation process.
    read_count_queue: Queue[int] = field(default_factory=mp.Queue)
    # Indicates that writing has finished and we can calculate.
    can_calc: list[Event] = field(init=False)
    # Passes the number of variants that have finished calculating to the writer
    # process.
    calc_count_queue: Queue[int] = field(default_factory=mp.Queue)
    # Indicates that calculation has finished and we can write out the results.
    can_write: list[Event] = field(init=False)

    def __post_init__(self) -> None:
        self.can_calc = [mp.Event() for _ in range(self.job_count)]
        self.can_write = [mp.Event() for _ in range(self.job_count)]


class Worker(Process):
    def __init__(
        self,
        t: TaskSyncCollection,
        *args,
        **kwargs,
    ) -> None:
        self.t = t
        super().__init__(t.exception_queue, *args, **kwargs)


class GenotypeReader(Worker):
    def __init__(
        self,
        t: TaskSyncCollection,
        vcf_file: VCFFile,
        genotypes_array: SharedArray,
        *args,
        **kwargs,
    ) -> None:
        if genotypes_array.shape[0] != vcf_file.sample_count:
            raise ValueError(
                "The row count of the genotype array does not match the VCF file."
            )

        self.vcf_file = vcf_file
        self.genotypes_array = genotypes_array

        super().__init__(t, *args, **kwargs)

    def func(self) -> None:
        variant_indices = self.vcf_file.variant_indices.copy()
        variant_count = self.genotypes_array.shape[1]

        with self.vcf_file:
            while len(variant_indices) > 0:
                # Make sure the genotypes array is not in use.
                action = self.t.wait(self.t.can_read)
                if action is Action.EXIT:
                    break
                self.t.can_read.clear()
                # Read the genotypes.
                self.vcf_file.variant_indices = variant_indices[:variant_count]
                genotypes = self.genotypes_array.to_numpy(
                    shape=(self.vcf_file.sample_count, self.vcf_file.variant_count)
                )
                logger.debug("Reading genotypes")
                self.vcf_file.read(genotypes.transpose())
                # Pass how many variants were read to the calculation process.
                self.t.read_count_queue.put_nowait(
                    int(self.vcf_file.variant_indices.size)
                )
                # Remove already read variant indices.
                variant_indices = variant_indices[variant_count:]
                if variant_indices.size == 0:
                    # Signal that we are done.
                    self.t.read_count_queue.put_nowait(0)
                    # Exit the process.
                    break


class Calc(Worker):
    def __init__(
        self,
        t: TaskSyncCollection,
        genotypes_array: SharedArray,
        eigenvector_arrays: list[SharedArray],
        rotated_genotypes_array: SharedArray,
        iv_arrays: list[SharedArray],
        ivsr_arrays: list[SharedArray],
        stat_array: SharedArray,
        *args,
        **kwargs,
    ) -> None:
        self.genotypes_array = genotypes_array
        self.rotated_genotypes_array = rotated_genotypes_array
        self.eigenvector_arrays = eigenvector_arrays
        self.iv_arrays = iv_arrays
        self.ivsr_arrays = ivsr_arrays
        self.stat_array = stat_array

        super().__init__(t, *args, **kwargs)

    def func(self) -> None:
        eigenvector_arrays = [eig.to_numpy() for eig in self.eigenvector_arrays]
        iv_arrays = [iv.to_numpy() for iv in self.iv_arrays]
        ivsr_arrays = [ivsr.to_numpy() for ivsr in self.ivsr_arrays]

        phenotype_counts = np.fromiter(
            (iv.shape[1] for iv in iv_arrays),
            dtype=int,
        )
        phenotype_indices = np.cumsum(phenotype_counts)
        phenotype_slices = [
            slice(start, end)
            for start, end in zip(
                np.concatenate(([0], phenotype_indices[:-1])),
                phenotype_indices,
            )
        ]

        job_count = len(self.eigenvector_arrays)

        while True:
            logger.debug(
                "Waiting for the reader to finish reading the a batch of variants"
            )
            value = self.t.get(self.t.read_count_queue)
            if value is Action.EXIT:
                break
            elif not isinstance(value, int):
                raise ValueError("Expected an integer.")
            variant_count = value
            self.t.calc_count_queue.put_nowait(variant_count)
            if value == 0:
                # Exit the process.
                return

            sample_count = self.genotypes_array.shape[0]
            genotypes = self.genotypes_array.to_numpy(
                shape=(sample_count, variant_count)
            )

            (_, phenotype_count, _) = self.stat_array.shape
            stat = self.stat_array.to_numpy(shape=(2, phenotype_count, variant_count))

            for i in range(job_count):
                can_calc = self.t.can_calc[i]
                eigenvectors = eigenvector_arrays[i]
                iv = iv_arrays[i]
                ivsr = ivsr_arrays[i]
                phenotype_slice = phenotype_slices[i]
                can_write = self.t.can_write[i]

                logger.debug("Multiply the genotypes with the eigenvectors")
                (sample_count, _) = ivsr.shape
                rotated_genotypes = self.rotated_genotypes_array.to_numpy(
                    shape=(sample_count, variant_count)
                )
                rotated_genotypes[:] = eigenvectors.transpose() @ genotypes

                if i == job_count - 1:
                    logger.debug("Allow the reader to read the next batch of variants")
                    self.t.can_read.set()

                logger.debug(
                    "Waiting for the writer to finish writing the previous batch of "
                    "variants"
                )
                action = self.t.wait(can_calc)
                if action is Action.EXIT:
                    break
                can_calc.clear()

                # Calculate the score statistics.
                u_stat = stat[0, phenotype_slice, :].transpose()
                v_stat = stat[1, phenotype_slice, :].transpose()
                logger.debug(
                    f"Calculating U statistic for phenotypes {phenotype_slice}"
                )
                calc_u_stat(ivsr, rotated_genotypes, u_stat)
                logger.debug("Squaring genotypes")
                rotated_genotypes[:] = np.square(rotated_genotypes)
                logger.debug("Calculating V statistic")
                calc_v_stat(iv, rotated_genotypes, u_stat, v_stat)
                # Signal that calculation has finished.
                can_write.set()


class ScoreWriter(Worker):
    def __init__(
        self,
        t: TaskSyncCollection,
        stat_array: SharedArray,
        array_proxy: ArrayProxy,
        phenotype_offset: int,
        variant_offset: int,
        *args,
        **kwargs,
    ) -> None:
        self.stat_array = stat_array
        self.array_proxy = array_proxy

        self.phenotype_offset = phenotype_offset
        self.variant_offset = variant_offset

        super().__init__(t, *args, **kwargs)

    def func(self) -> None:
        job_count = len(self.t.can_write)
        (_, phenotype_count, _) = self.stat_array.shape
        phenotype_slice = slice(
            self.phenotype_offset, self.phenotype_offset + phenotype_count
        )
        variant_index = self.variant_offset

        while True:
            # Wait for the calculation to finish.
            value = self.t.get(self.t.calc_count_queue)
            if value is Action.EXIT:
                break
            elif not isinstance(value, int):
                raise ValueError("Expected an integer.")
            variant_count = value
            if variant_count == 0:
                # Exit the process.
                break

            logger.debug("Wait for the calculation to finish")
            for i in range(job_count):
                can_write = self.t.can_write[i]
                action = self.t.wait(can_write)
                if action is Action.EXIT:
                    break
                can_write.clear()

            # Write the data.
            variant_slice = slice(variant_index, variant_index + variant_count)
            stat = self.stat_array.to_numpy(shape=(2, phenotype_count, variant_count))
            logger.debug(
                f"Writing variants {variant_slice} to {self.array_proxy} "
                f"with shape {stat.transpose().shape}"
            )
            self.array_proxy[variant_slice, phenotype_slice, :] = stat.transpose()

            # Allow the calculation to continue.
            for i in range(job_count):
                can_calc = self.t.can_calc[i]
                can_calc.set()

            variant_index += variant_count


def calc_score(
    vcf_file: VCFFile,
    eigendecompositions: list[Eigendecomposition],
    iv_arrays: list[SharedArray],
    ivsr_arrays: list[SharedArray],
    array_proxy: ArrayProxy,
    phenotype_offset: int = 0,
    variant_offset: int = 0,
) -> None:
    """Calculate the Chen and Abecasis (2007) score statistic for phenotypes with no
    missing data.

    Args:
        vcf_file (VCFFile): An object containing the VCF-file header information and
            which samples to read from it.
        vc (VariableCollection): An object containing the phenotype and covariate data.
        nm (NullModelCollection): An object containing the estimated variance
            components.
        eig (Eigendecomposition): An object containing the eigenvectors and eigenvalues
            for the leave-one-chromosome-out kinship matrices.
        sw (SharedWorkspace): The shared workspace from where we can allocate arrays in
            shared memory.
        output_directory (Path): The path to use for the output file.

    Raises:
        Exception: Any exception that is raised by the worker processes.
    """
    # Merge the eigenvector arrays so that we can use a single reader process.
    job_count = len(eigendecompositions)
    ec = EigendecompositionCollection.from_eigendecompositions(
        vcf_file,
        eigendecompositions,
    )
    vcf_file.set_samples(set(ec.samples))
    if vcf_file.samples != ec.samples:
        raise ValueError(
            "Sample order of eigendecompositions does not match the VCF file"
        )
    # Make sure that we can use all free memory.
    sw = ec.eigenvector_arrays[0].sw
    sw.squash()
    # We re-use sample x genotype matrix across all jobs, so we need to use
    # the total number of samples.
    sample_count = vcf_file.sample_count
    phenotype_count = sum(iv.shape[1] for iv in iv_arrays)
    per_variant_size = np.float64().itemsize * 2 * (phenotype_count + sample_count)
    variant_count = sw.unallocated_size // per_variant_size
    variant_count = min(variant_count, vcf_file.variant_count)
    logger.debug(
        f"Will calculate score statistics in blocks of {variant_count} variants "
        f"because we have {sw.unallocated_size} bytes of free memory and "
        f"need {per_variant_size} bytes per variant."
    )
    # Allocate the arrays in shared memory.
    name = SharedArray.get_name(sw, "genotypes")
    genotype_array = sw.alloc(name, sample_count, variant_count)
    name = SharedArray.get_name(sw, "rotated-genotypes")
    rotated_genotype_array = sw.alloc(name, sample_count, variant_count)
    name = SharedArray.get_name(sw, "stat")
    stat_array: SharedArray = sw.alloc(name, 2, phenotype_count, variant_count)
    # Create the worker processes.
    t = TaskSyncCollection(job_count=job_count)
    reader_proc = GenotypeReader(t, vcf_file, genotype_array)
    calc_proc = Calc(
        t,
        genotype_array,
        ec.eigenvector_arrays,
        rotated_genotype_array,
        iv_arrays,
        ivsr_arrays,
        stat_array,
    )
    writer_proc = ScoreWriter(
        t, stat_array, array_proxy, phenotype_offset, variant_offset
    )
    # Start the loop.
    procs = [reader_proc, calc_proc, writer_proc]
    try:
        for proc in procs:
            proc.start()
        # Allow use of genotype_array and stat_array.
        t.can_read.set()
        for can_calc in t.can_calc:
            can_calc.set()
        while True:
            try:
                raise t.exception_queue.get_nowait()
            except Empty:
                pass
            for proc in procs:
                proc.join(timeout=1)
            if all(not proc.is_alive() for proc in procs):
                break
    finally:
        t.should_exit.set()
        for proc in procs:
            proc.terminate()
            proc.join(timeout=1)
            if proc.is_alive():
                proc.kill()
            proc.join()
            proc.close()
        genotype_array.free()
        rotated_genotype_array.free()
        stat_array.free()
