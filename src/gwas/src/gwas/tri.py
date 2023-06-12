# -*- coding: utf-8 -*-
from __future__ import annotations

import multiprocessing as mp
from contextlib import nullcontext
from dataclasses import dataclass, field
from multiprocessing.synchronize import Event, Lock
from pathlib import Path
from queue import Empty
from typing import ContextManager, Mapping, MutableMapping, NamedTuple, Sequence

import numpy as np
from numpy import typing as npt

from .log import logger
from .mem.arr import SharedArray
from .mem.wkspace import SharedWorkspace
from .utils import Process, SharedState, invert_pivot
from .vcf.base import VCFFile


@dataclass
class Triangular(SharedArray):
    chromosome: int | str
    samples: list[str]
    variant_count: int

    minor_allele_frequency_cutoff: float
    r_squared_cutoff: float

    @property
    def sample_count(self) -> int:
        return self.shape[0]

    def to_file_name(self) -> str:
        return self.get_file_name(self.chromosome)

    def subset_samples(self, samples: list[str]) -> None:
        """Reduce the triangular matrix to a subset of samples.
        Golub and Van Loan (1996) section 12.5.2 implements this by first removing the
        specified columns from the matrix, and then applies Givens rotations to make
        it triangular again.
        Since we are just using them for SVD, we just remove the columns and leave
        the matrix non-triangular.

        Args:
            samples (list[str]): The samples to keep.
        """
        if samples == self.samples:
            # Nothing to do.
            return

        new_sample_count = len(samples)
        new_sample_indices = [self.samples.index(sample) for sample in samples]

        # Remove samples.
        array = self.to_numpy()
        array[:, :new_sample_count] = array[:, new_sample_indices]
        self.resize(self.sample_count, new_sample_count)

        self.samples = samples

    @staticmethod
    def get_file_name(chromosome: int | str) -> str:
        return f"chr{chromosome}.tri.txt.gz"

    @staticmethod
    def get_prefix(**kwargs) -> str:
        chromosome = kwargs.get("chromosome")
        if chromosome is not None:
            return f"chr{chromosome}-tri"
        else:
            return "tri"

    @classmethod
    def from_vcf(
        cls,
        vcf_file: VCFFile,
        sw: SharedWorkspace,
    ) -> Triangular | None:
        tsqr = TallSkinnyQR(
            vcf_file,
            sw,
        )
        return tsqr.map_reduce()


def scale(b: npt.NDArray):
    # calculate variant properties
    mean = b.mean(axis=1)
    minor_allele_frequency = mean / 2

    # apply scaling
    b -= mean[:, np.newaxis]
    standard_deviation = np.sqrt(
        2 * minor_allele_frequency * (1 - minor_allele_frequency)
    )
    b /= standard_deviation[:, np.newaxis]


@dataclass
class TallSkinnyQR:
    vcf_file: VCFFile
    sw: SharedWorkspace

    t: TaskSyncCollection | None = None
    variant_indices: npt.NDArray[np.uint32] | None = None

    def map(self) -> Triangular:
        """Triangularize as much of the VCF file as fits into the shared
        workspace. The result is an m-by-m lower triangular matrix with
        the given name.

        Parameters
        ----------
        vcf_file : VCFFile
            vcf_file
        sw : SharedWorkspace
            sw
        minor_allele_frequency_cutoff : float
            minor_allele_frequency_cutoff

        Returns
        -------
        SharedArray | None

        """

        sample_count = self.vcf_file.sample_count
        variant_count = self.vcf_file.variant_count

        name = Triangular.get_name(self.sw, chromosome=self.vcf_file.chromosome)
        shared_array = self.sw.alloc(name, sample_count, variant_count)

        if self.variant_indices is not None:
            if self.variant_indices.size == 0:
                if self.t is not None:
                    # We have enough space to start another task in parallel.
                    self.t.can_run.set()

        # Read dosages from the VCF file.
        array = shared_array.to_numpy()
        logger.debug(
            f"Mapping {array.shape[1]} variants from "
            f'"{self.vcf_file.file_path.name}" into "{shared_array.name}"'
        )
        self.vcf_file.read(array.transpose())

        # Transpose, reshape and scale the data.
        shared_array.resize(sample_count, variant_count)
        shared_array.transpose()
        array = shared_array.to_numpy()
        scale(array)

        multithreading_lock: ContextManager = nullcontext()
        if self.t is not None:
            multithreading_lock = self.t.multithreading_lock

        # Triangularize to upper triangle.
        with multithreading_lock:
            pivot = shared_array.triangularize_with_pivoting()

        # Transpose and reshape to lower triangle.
        shared_array.transpose()
        shared_array.resize(sample_count, sample_count)

        # Apply the inverse pivot to the rows.
        array = shared_array.to_numpy()
        array[:] = array[invert_pivot(pivot), :]

        return Triangular(
            name=name,
            sw=self.sw,
            chromosome=self.vcf_file.chromosome,
            samples=self.vcf_file.samples,
            variant_count=variant_count,
            minor_allele_frequency_cutoff=self.vcf_file.minor_allele_frequency_cutoff,
            r_squared_cutoff=self.vcf_file.r_squared_cutoff,
        )

    @staticmethod
    def reduce(*shared_arrays: Triangular) -> Triangular:
        if len(shared_arrays) == 0:
            raise ValueError

        if len(shared_arrays) == 1:
            (shared_array,) = shared_arrays
            return shared_array

        logger.debug(f"Reducing {len(shared_arrays)} chunks")

        names = [shared_array.name for shared_array in shared_arrays]

        sw = shared_arrays[0].sw
        reduce_array = sw.merge(*names)

        # Triangularize to upper triangle.
        reduce_array.transpose()
        pivot = reduce_array.triangularize_with_pivoting()

        # Transpose and reshape to lower triangle.
        reduce_array.transpose()
        sample_count = reduce_array.shape[0]
        reduce_array.resize(sample_count, sample_count)

        # Apply the inverse pivot to the rows.
        array = reduce_array.to_numpy()
        array[:] = array[invert_pivot(pivot), :]

        # Get metadata.
        chromosome_set = set(a.chromosome for a in shared_arrays)
        if len(chromosome_set) == 1:
            (chromosome,) = chromosome_set
        else:
            raise ValueError

        cutoffs = sorted(a.minor_allele_frequency_cutoff for a in shared_arrays)
        if np.isclose(min(cutoffs), max(cutoffs)):
            minor_allele_frequency_cutoff = cutoffs[0]
        else:
            raise ValueError

        cutoffs = sorted(a.r_squared_cutoff for a in shared_arrays)
        if np.isclose(min(cutoffs), max(cutoffs)):
            r_squared_cutoff = cutoffs[0]
        else:
            raise ValueError

        variant_count = sum(a.variant_count for a in shared_arrays)

        return Triangular(
            name=reduce_array.name,
            sw=sw,
            chromosome=chromosome,
            samples=shared_arrays[0].samples,
            variant_count=variant_count,
            minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
            r_squared_cutoff=r_squared_cutoff,
        )

    @property
    def variant_count(self) -> int:
        sample_count = self.vcf_file.sample_count
        variant_count = self.sw.unallocated_size // (
            np.float64().itemsize * sample_count
        )
        if variant_count < sample_count:
            raise MemoryError(
                f"There is only space for {variant_count:d} columns of the matrix, but "
                f"we need at least {sample_count:d} columns."
            )
        if variant_count >= self.vcf_file.variant_count:
            # We can fit the entire VCF file into memory.
            logger.debug(
                f"There is space for {variant_count:d} columns of the matrix, but "
                f"we only need {self.vcf_file.variant_count:d}."
            )
            variant_count = self.vcf_file.variant_count
        return variant_count

    def map_reduce(self) -> Triangular | None:
        self.variant_indices = self.vcf_file.variant_indices.copy()

        arrays: list[Triangular] = list()
        with self.vcf_file:
            while self.variant_indices.size > 0:
                try:
                    variant_count = self.variant_count
                    variant_indices = self.variant_indices[:variant_count]
                    self.variant_indices = self.variant_indices[variant_count:]
                    self.vcf_file.variant_indices = variant_indices

                    arrays.append(self.map())
                except MemoryError:
                    arrays = [self.reduce(*arrays)]

        return self.reduce(*arrays)


@dataclass
class TaskSyncCollection(SharedState):
    # Indicates that can run another task.
    can_run: Event = field(default_factory=mp.Event)
    # Ensures that only one multithreaded workload can run at a time.
    multithreading_lock: Lock = field(default_factory=mp.Lock)


class TriWorker(Process):
    def __init__(
        self,
        tsqr: TallSkinnyQR,
        tri_path: Path,
        t: TaskSyncCollection,
        *args,
        **kwargs,
    ) -> None:
        self.tsqr = tsqr
        self.tri_path = tri_path
        self.t = t

        if "name" not in kwargs:
            kwargs["name"] = f"tri-worker-chr{tsqr.vcf_file.chromosome}"

        super().__init__(t.exception_queue, *args, **kwargs)

    def func(self) -> None:
        logger.debug(f"Triangularizing chromosome {self.tsqr.vcf_file.chromosome}")
        tri = self.tsqr.map_reduce()

        if tri is None:
            vcf_file = self.tsqr.vcf_file
            raise ValueError(f"Could not triangularize {vcf_file.file_path}")

        tri.to_file(self.tri_path)
        tri.free()
        # Indicate that we can start another task as this one has finished.
        self.t.can_run.set()


class Task(NamedTuple):
    required_size: int
    proc: TriWorker


def calc_tri(
    chromosomes: Sequence[str | int],
    vcf_by_chromosome: Mapping[int | str, VCFFile],
    output_directory: Path,
    sw: SharedWorkspace,
    tri_paths: list[Path] | None = None,
    minor_allele_frequency_cutoff: float = 0.05,
    r_squared_cutoff: float = -np.inf,
) -> Mapping[int | str, Path]:
    """Generate triangular matrices for each chromosome.

    Args:
        chromosomes (Sequence[str  |  int]): The chromosomes to triangularize.
        vcf_by_chromosome (Mapping[int  |  str, VCFFile]): The VCF file objects that
            allow reading the genotypes.
        output_directory (Path): The function will first try to load existing triangular
            matrices from this directory. If they do not exist, they will be generated.
        sw (SharedWorkspace): The shared workspace.
        tri_paths (list[Path] | None, optional): Optionally specify additional paths to
            triangular matrices. Defaults to None.
        minor_allele_frequency_cutoff (float, optional): Defaults to 0.05.
        r_squared_cutoff (float, optional): Defaults to 0.

    Raises:
        ValueError: If a triangular matrix could not be found or generated.

    Returns:
        Mapping[int | str, Path]: The paths to the triangular matrix files.
    """
    tri_paths_by_chromosome: MutableMapping[int | str, Path] = dict()

    def check_tri_path(tri_path: Path) -> bool:
        if not tri_path.is_file():
            return False
        try:
            _, kwargs = Triangular.read_file_metadata(tri_path)
        except (ValueError, TypeError):
            return False
        chromosome = kwargs["chromosome"]
        samples = vcf_by_chromosome[chromosome].samples
        if set(kwargs["samples"]) == set(samples):
            logger.debug(
                f"Using existing triangularized file {tri_path} "
                f"for chromosome {chromosome}"
            )
            tri_paths_by_chromosome[chromosome] = tri_path
        else:
            logger.warning(
                f"Will re-calculate tri file {tri_path} "
                f"because samples do not match"
            )
        return chromosome in tri_paths_by_chromosome

    if tri_paths is not None:
        # Load from `--tri` flag.
        for tri_path in tri_paths:
            check_tri_path(tri_path)

    t = TaskSyncCollection()
    t.can_run.set()  # We can run the first task immediately.

    # Prepare the list of jobs to run.
    jobs: list[Task] = list()
    for chromosome in chromosomes:
        if chromosome == "X":
            # We only use autosomes for null model estimation, so we only
            # need to triangularize the autosomes.
            continue
        if chromosome in tri_paths_by_chromosome:
            # Already loaded from `--tri` flag.
            continue
        # Generate default path.
        tri_path = output_directory / Triangular.get_file_name(chromosome)
        if check_tri_path(tri_path):
            continue

        vcf_file = vcf_by_chromosome[chromosome]
        vcf_file.set_variants_from_cutoffs(
            minor_allele_frequency_cutoff,
            r_squared_cutoff,
        )

        tsqr = TallSkinnyQR(
            vcf_file=vcf_file,
            sw=sw,
            t=t,
        )
        tri_paths_by_chromosome[chromosome] = tri_path

        proc: TriWorker = TriWorker(tsqr, tri_path, t)
        required_size = (
            np.float64().itemsize * vcf_file.sample_count * vcf_file.variant_count
        )
        jobs.append(Task(required_size, proc))

    # Sort tasks by size so that we can run the largest tasks first. This means that we
    # are less likely to run into a situation where we only run one task at a time.
    jobs.sort(key=lambda job_tuple: job_tuple[0])
    logger.debug(f"Will run {len(jobs)} triangularize jobs")

    running: list[TriWorker] = list()
    barrier: bool = True
    try:
        while True:
            # Check if an error has occurred.
            try:
                raise t.exception_queue.get_nowait()
            except Empty:
                pass

            # Sleep for one second if a process is running.
            for proc in running:
                proc.join(timeout=1)
                if proc.is_alive():
                    break

            # Update list of running processes.
            running = [proc for proc in running if proc.is_alive()]

            # Check if we can exit.
            if len(running) == 0 and len(jobs) == 0:
                # All tasks have been completed.
                break

            # Update the barrier.
            if len(running) == 0:
                # All processes have exited so we can start more.
                barrier = True

            # Check if we can start another task.
            if not t.can_run.is_set():
                # The most recently started task has not yet initialized.
                continue
            if len(jobs) == 0:
                # No more tasks to run.
                continue
            if not barrier:
                # We are still waiting for processes to finish.
                continue

            # Calculate the amount of memory required to run the next task in parallel.
            unallocated_size = sw.unallocated_size
            sample_count = jobs[-1].proc.tsqr.vcf_file.sample_count
            extra_required_size = (
                (len(running) + 1) * np.float64().itemsize * sample_count**2
            )
            required_size = jobs[-1].required_size + extra_required_size
            logger.debug(
                f"We have {unallocated_size} bytes left in the shared "
                f"workspace. The next task requires {required_size} bytes to "
                "run in parallel."
            )

            if unallocated_size < required_size and len(running) > 0:
                # We have already started a task, but we don't have enough memory
                # to run the next task in parallel.
                # Set the barrier to wait for the all running tasks to complete before
                # starting next batch.
                logger.debug(
                    "Waiting for running tasks to complete before starting next batch."
                )
                barrier = False
                continue

            proc = jobs.pop().proc
            proc.start()
            running.append(proc)

            # Reset the event so that we don't start another task before
            # this one has initialized.
            t.can_run.clear()
    finally:
        t.should_exit.set()

        for _, proc in jobs:
            proc.terminate()
            proc.join(timeout=1)
            if proc.is_alive():
                proc.kill()
            proc.join()
            proc.close()

    for tri_path in tri_paths_by_chromosome.values():
        if not tri_path.is_file():
            raise ValueError(f'Could not find output file "{tri_path}"')

    return tri_paths_by_chromosome
