# -*- coding: utf-8 -*-

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event

import numpy as np

from ..compression.arr.base import FileArray
from ..log import logger
from ..mem.arr import SharedArray
from ..utils import Action, Process, SharedState
from ..vcf.base import VCFFile
from .calc import calc_u_stat, calc_v_stat


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
        stat_file_array: FileArray,
        phenotype_offset: int,
        variant_offset: int,
        *args,
        **kwargs,
    ) -> None:
        self.stat_array = stat_array
        self.stat_file_array = stat_file_array

        self.phenotype_offset = phenotype_offset
        self.variant_offset = variant_offset

        super().__init__(t, *args, **kwargs)

    def func(self) -> None:
        job_count = len(self.t.can_write)
        (_, phenotype_count, _) = self.stat_array.shape
        phenotype_slice = slice(
            2 * self.phenotype_offset, 2 * (self.phenotype_offset + phenotype_count)
        )
        variant_index = self.variant_offset

        with self.stat_file_array:
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
                stat = self.stat_array.to_numpy(
                    shape=(2, phenotype_count, variant_count)
                )
                logger.debug(
                    f"Writing variants {variant_slice} to {self.stat_file_array} "
                    f"with shape {stat.transpose().shape}"
                )
                two_dimensional_stat = stat.transpose().reshape(
                    (variant_count, 2 * phenotype_count)
                )
                self.stat_file_array[
                    variant_slice, phenotype_slice
                ] = two_dimensional_stat

                # Allow the calculation to continue.
                for i in range(job_count):
                    can_calc = self.t.can_calc[i]
                    can_calc.set()

                variant_index += variant_count
