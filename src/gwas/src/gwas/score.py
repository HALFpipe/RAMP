# -*- coding: utf-8 -*-

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from queue import Empty

import numpy as np
from numpy import typing as npt

from .eig import Eigendecomposition
from .log import logger
from .mem.arr import SharedArray
from .mem.wkspace import SharedWorkspace
from .pheno import VariableCollection
from .rmw import AnnotatedVariant, CombinedScorefile
from .utils import Action, Process, SharedState
from .var import NullModelCollection
from .vcf import VCFFile
from .z import CompressedTextWriter


def calc_u_stat(
    nm: NullModelCollection,
    rotated_genotypes: npt.NDArray,
    u_stat: npt.NDArray,
) -> None:
    scaled_residuals = nm.scaled_residuals.to_numpy()
    variance = nm.variance.to_numpy()

    inverse_variance = np.power(variance, -0.5)
    logger.debug("Calculating numerator")
    u_stat[:] = rotated_genotypes.transpose() @ (inverse_variance * scaled_residuals)


def calc_v_stat(
    nm: NullModelCollection,
    squared_genotypes: npt.NDArray,
    u_stat: npt.NDArray,
    v_stat: npt.NDArray,
) -> npt.NDArray[np.bool_]:
    variance = nm.variance.to_numpy()

    inverse_variance = np.power(variance, -1)
    logger.debug("Calculating denominator")
    v_stat[:] = squared_genotypes.transpose() @ inverse_variance

    logger.debug("Zeroing invalid values")
    invalid = np.isclose(v_stat, 0)
    u_stat[invalid] = 0
    v_stat[invalid] = 1

    return invalid


@dataclass
class TaskSyncCollection(SharedState):
    # Indicates that the genotypes array is empty and we can read data into it.
    can_read: Event = field(default_factory=mp.Event)
    # Indicates that the genotypes array was read and we can start processing.
    can_calc: Event = field(default_factory=mp.Event)
    # Indicates that calculation has finished and we can write out the results.
    can_write: Event = field(default_factory=mp.Event)

    # Passes the number of variants that were read to the calculation process.
    variant_count_queue: Queue[int] = field(default_factory=mp.Queue)
    # Passes metadata of the read variants from the reader to the writer.
    annot_variant_queue: Queue[list[AnnotatedVariant]] = field(default_factory=mp.Queue)


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
        genotypes = self.genotypes_array.to_numpy()

        with self.vcf_file:
            while True:
                # Make sure the genotypes array is not in use.
                action = self.t.wait(self.t.can_read)
                if action is Action.EXIT:
                    break
                self.t.can_read.clear()

                variants = self.vcf_file.read(genotypes.transpose())

                variant_count = len(variants)
                if variant_count == 0:
                    # Signal that we are done.
                    self.t.variant_count_queue.put_nowait(variant_count)
                    self.t.annot_variant_queue.put_nowait(list())
                    # Exit the process.
                    break

                # Calculate metadata.
                alternate_allele_count = genotypes.sum(axis=0)
                mean = alternate_allele_count / self.vcf_file.sample_count
                alternate_allele_frequency = mean / 2
                call_rate = (
                    np.count_nonzero(genotypes, axis=0) / self.vcf_file.sample_count
                )

                # Mean-center the genotypes.
                logger.debug("Mean-centering genotypes")
                genotypes -= mean

                # Signal that processing can start.
                self.t.variant_count_queue.put_nowait(variant_count)

                # Send the metadata to the writer.
                annot_variants = [
                    AnnotatedVariant(
                        v.chromosome,
                        v.position,
                        v.reference_allele,
                        v.alternate_allele,
                        alternate_allele_count[i],
                        alternate_allele_frequency[i],
                        call_rate[i],
                    )
                    for i, v in enumerate(variants)
                ]
                self.t.annot_variant_queue.put_nowait(annot_variants)


class Calc(Worker):
    def __init__(
        self,
        t: TaskSyncCollection,
        genotypes_array: SharedArray,
        eig: Eigendecomposition,
        rotated_genotypes_array: SharedArray,
        nm: NullModelCollection,
        u_stat_array: SharedArray,
        v_stat_array: SharedArray,
        *args,
        **kwargs,
    ) -> None:
        self.genotypes_array = genotypes_array
        self.eig = eig
        self.rotated_genotypes_array = rotated_genotypes_array
        self.nm = nm
        self.u_stat_array = u_stat_array
        self.v_stat_array = v_stat_array

        super().__init__(t, *args, **kwargs)

    def func(self) -> None:
        sample_count = self.genotypes_array.shape[0]
        phenotype_count = self.nm.phenotype_count
        eigenvectors = self.eig.eigenvectors

        while True:
            # Wait for the reader to finish reading the a batch of variants.
            value = self.t.get(self.t.variant_count_queue)
            if value is Action.EXIT:
                break
            elif not isinstance(value, int):
                raise ValueError("Expected an integer.")
            variant_count = value

            if value == 0:
                break

            shape = (sample_count, variant_count)
            genotypes = self.genotypes_array.to_numpy(shape=shape)
            rotated_genotypes = self.rotated_genotypes_array.to_numpy(shape=shape)

            logger.debug("Rotating genotypes")
            rotated_genotypes[:] = eigenvectors.transpose() @ genotypes

            # Allow the reader to read the next batch of variants.
            self.t.can_read.set()

            # Wait for the writer to finish writing the previous batch of variants.
            action = self.t.wait(self.t.can_calc)
            if action is Action.EXIT:
                break
            self.t.can_calc.clear()

            shape = (phenotype_count, variant_count)
            v_stat = self.v_stat_array.to_numpy(shape=shape).transpose()
            u_stat = self.u_stat_array.to_numpy(shape=shape).transpose()

            calc_u_stat(self.nm, rotated_genotypes, u_stat)

            logger.debug("Squaring genotypes")
            rotated_genotypes[:] = np.square(rotated_genotypes)

            calc_v_stat(self.nm, rotated_genotypes, u_stat, v_stat)

            # Signal that calculation has finished.
            self.t.can_write.set()


class ScoreWriter(Worker):
    def __init__(
        self,
        t: TaskSyncCollection,
        eig: Eigendecomposition,
        vc: VariableCollection,
        nm: NullModelCollection,
        u_stat_array: SharedArray,
        v_stat_array: SharedArray,
        file_path: Path,
        *args,
        **kwargs,
    ) -> None:
        self.eig = eig
        self.vc = vc
        self.nm = nm
        self.u_stat_array = u_stat_array
        self.v_stat_array = v_stat_array
        self.file_path = file_path

        super().__init__(t, *args, **kwargs)

    def func(self) -> None:
        phenotype_count = self.vc.phenotype_count
        kwargs = dict(
            n_informative=self.vc.sample_count,
        )

        with CompressedTextWriter(self.file_path) as file_handle:
            header = CombinedScorefile.make_header(self.vc, self.nm)
            CombinedScorefile.write_header(file_handle, header)
            CombinedScorefile.write_names(file_handle, self.vc)
            while True:
                # Wait for the calculation to finish.
                value = self.t.get(self.t.annot_variant_queue)
                if value is Action.EXIT:
                    break
                elif not isinstance(value, list):
                    raise ValueError("Expected an integer.")
                variants = value

                if len(variants) == 0:
                    break

                action = self.t.wait(self.t.can_write)
                if action is Action.EXIT:
                    break
                self.t.can_write.clear()

                shape = (phenotype_count, len(variants))
                v_stat = self.v_stat_array.to_numpy(shape=shape).transpose()
                u_stat = self.u_stat_array.to_numpy(shape=shape).transpose()

                CombinedScorefile.write_scores(
                    file_handle,
                    variants,
                    u_stat,
                    v_stat,
                    **kwargs,
                )

                self.t.can_calc.set()


def calc_score(
    vcf_file: VCFFile,
    vc: VariableCollection,
    nm: NullModelCollection,
    eig: Eigendecomposition,
    sw: SharedWorkspace,
    score_path: Path,
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
        score_path (Path): The path to use for the output file.

    Raises:
        Exception: Any exception that is raised by the worker processes.
    """
    sw.squash()  # make sure that we can use all free memory

    variant_count = sw.unallocated_size // (
        np.float64().itemsize * 2 * (vc.phenotype_count + vc.sample_count)
    )
    variant_count = min(variant_count, vcf_file.variant_count)

    logger.info(
        f"Will calculate score statistics in chunks of {variant_count} variants"
    )

    name = SharedArray.get_name(sw, "genotypes")
    gen_array = sw.alloc(name, vc.sample_count, variant_count)
    name = SharedArray.get_name(sw, "rotated-genotypes")
    rotated_gen_array = sw.alloc(name, vc.sample_count, variant_count)
    name = SharedArray.get_name(sw, "u-stat")
    u_array = sw.alloc(name, vc.phenotype_count, variant_count)
    name = SharedArray.get_name(sw, "v-stat")
    v_array = sw.alloc(name, vc.phenotype_count, variant_count)

    t = TaskSyncCollection()
    reader_proc = GenotypeReader(t, vcf_file, gen_array)
    calc_proc = Calc(t, gen_array, eig, rotated_gen_array, nm, u_array, v_array)
    writer_proc = ScoreWriter(t, eig, vc, nm, u_array, v_array, score_path)

    procs = [reader_proc, calc_proc, writer_proc]
    try:
        for proc in procs:
            proc.start()

        # Allow use of the gen_array, u_array, and v_array.
        t.can_read.set()
        t.can_calc.set()

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

        gen_array.free()
        rotated_gen_array.free()
        u_array.free()
        v_array.free()
