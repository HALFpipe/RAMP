# -*- coding: utf-8 -*-
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import Empty
from typing import (
    Collection,
    Iterable,
    Mapping,
    MutableMapping,
    NamedTuple,
    Sequence,
)

import numpy as np
from tqdm.auto import tqdm

from ..log import logger
from ..mem.wkspace import SharedWorkspace
from ..utils import IterationOrder, make_pool_or_null_context, soft_close
from ..vcf.base import VCFFile
from .base import TaskSyncCollection, Triangular
from .tsqr import TallSkinnyQR
from .worker import TriWorker


class Task(NamedTuple):
    required_size: int
    proc: TriWorker


def check_tri_path(
    tri_path: Path,
    samples_by_chromosome: Mapping[int | str, list[str]],
    sw: SharedWorkspace,
) -> tuple[int | str, Path] | None:
    try:
        tri = Triangular.from_file(tri_path, sw, np.float64)
        chromosome = tri.chromosome
        samples = samples_by_chromosome[chromosome]
        tri.free()
    except FileNotFoundError:
        return None
    except (ValueError, TypeError) as error:
        logger.warning(
            f"Will re-calculate tri file {tri_path} because of an error reading the "
            "existing file",
            exc_info=error,
        )
        return None
    if set(tri.samples) <= set(samples):
        logger.debug(
            f"Using existing triangularized file {tri_path} "
            f"for chromosome {chromosome}"
        )
        return chromosome, tri_path
    else:
        logger.warning(
            f"Will re-calculate tri file {tri_path} because samples do not match"
        )
        return None


@dataclass
class TriCalc:
    chromosomes: Collection[str | int]
    vcf_by_chromosome: Mapping[int | str, VCFFile]
    output_directory: Path
    sw: SharedWorkspace

    minor_allele_frequency_cutoff: float
    r_squared_cutoff: float

    num_threads: int = 1

    @property
    def samples_by_chromosome(self) -> Mapping[int | str, list[str]]:
        return {
            chromosome: vcf_file.samples
            for chromosome, vcf_file in self.vcf_by_chromosome.items()
        }

    def make_tri_paths_by_chromosome(
        self, tri_paths: Iterable[Path]
    ) -> MutableMapping[int | str, Path]:
        check = partial(
            check_tri_path, samples_by_chromosome=self.samples_by_chromosome, sw=self.sw
        )
        pool, iterator = make_pool_or_null_context(
            tri_paths,
            check,
            num_threads=self.num_threads,
            iteration_order=IterationOrder.UNORDERED,
        )
        tri_paths_by_chromosome: MutableMapping[int | str, Path] = dict()
        with pool:
            for result in iterator:
                if result is None:
                    continue
                chromosome, tri_path = result
                tri_paths_by_chromosome[chromosome] = tri_path
        return tri_paths_by_chromosome

    def check_tri_paths(
        self, tri_paths: Collection[Path]
    ) -> tuple[Collection[str | int], MutableMapping[int | str, Path]]:
        # Ensure the output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)

        chromosomes_to_check = set(self.chromosomes)
        chromosomes_to_check.discard("X")

        tri_paths_by_chromosome: dict[int | str, Path] = dict()

        # Load from `--tri` flag
        if len(tri_paths) > 0:
            for chromosome, tri_path in self.make_tri_paths_by_chromosome(
                tri_paths
            ).items():
                tri_paths_by_chromosome[chromosome] = tri_path
                chromosomes_to_check.remove(chromosome)

        # Load from output directory
        tri_paths = list()
        for chromosome in chromosomes_to_check:
            tri_path = self.output_directory / Triangular.get_file_name(chromosome)
            tri_paths_by_chromosome[chromosome] = tri_path
            tri_paths.append(tri_path)
        if len(tri_paths) > 0:
            for chromosome, tri_path in self.make_tri_paths_by_chromosome(
                tri_paths
            ).items():
                tri_paths_by_chromosome[chromosome] = tri_path
                chromosomes_to_check.remove(chromosome)

        return chromosomes_to_check, tri_paths_by_chromosome

    def get_tri_tasks(
        self,
        chromosomes: Collection[str | int],
        tri_paths_by_chromosome: MutableMapping[int | str, Path],
    ) -> tuple[TaskSyncCollection, list[Task]]:
        t = TaskSyncCollection()
        t.can_run.set()  # We can run the first task immediately.
        # Prepare the list of tasks to run.
        tasks: list[Task] = list()

        for chromosome in chromosomes:
            tri_path = tri_paths_by_chromosome[chromosome]
            vcf_file = self.vcf_by_chromosome[chromosome]
            vcf_file.set_variants_from_cutoffs(
                self.minor_allele_frequency_cutoff,
                self.r_squared_cutoff,
            )

            tsqr = TallSkinnyQR(
                vcf_file=vcf_file,
                sw=self.sw,
                t=t,
                num_threads=self.num_threads,
            )

            proc: TriWorker = TriWorker(tsqr, tri_path, t)
            required_size = (
                np.float64().itemsize * vcf_file.sample_count * vcf_file.variant_count
            )
            tasks.append(Task(required_size, proc))

        return t, tasks

    def run(self, tri_paths: Collection[Path]) -> Mapping[int | str, Path]:
        chromosomes_to_run, tri_paths_by_chromosome = self.check_tri_paths(tri_paths)
        t, tasks = self.get_tri_tasks(chromosomes_to_run, tri_paths_by_chromosome)

        # Sort tasks by size so that we can run the largest tasks first. This means that
        # we are less likely to run into a situation where we only run one task at a time
        tasks.sort(key=lambda task: task.required_size)
        logger.debug(f"Will run {len(tasks)} triangularize tasks")

        running: list[TriWorker] = list()
        try:
            check_running(self.sw, t, tasks, running)
        finally:
            t.should_exit.set()

            for _, proc in tasks:
                soft_close(proc)

        return tri_paths_by_chromosome


def calc_tri(
    chromosomes: Sequence[str | int],
    vcf_by_chromosome: Mapping[int | str, VCFFile],
    output_directory: Path,
    sw: SharedWorkspace,
    tri_paths: list[Path] | None = None,
    minor_allele_frequency_cutoff: float = 0.05,
    r_squared_cutoff: float = -np.inf,
    num_threads: int = 1,
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
    if tri_paths is None:
        tri_paths = list()
    return TriCalc(
        chromosomes,
        vcf_by_chromosome,
        output_directory,
        sw,
        minor_allele_frequency_cutoff,
        r_squared_cutoff,
        num_threads,
    ).run(tri_paths)


def check_running(
    sw: SharedWorkspace,
    t: TaskSyncCollection,
    tasks: list[Task],
    running: list[TriWorker],
) -> None:
    barrier: bool = True

    with tqdm(
        total=len(tasks),
        unit="chromosomes",
        desc="triangularizing genotypes",
    ) as progress_bar:
        while True:
            # Check if an error has occurred.
            try:
                raise t.exception_queue.get_nowait()
            except Empty:
                pass

            # Sleep for one second if a process is running
            wait(running)

            # Update progress bar with the number of processes that have finished
            progress_bar.update(sum(1 for proc in running if not proc.is_alive()))

            # Update list of running processes
            running = [proc for proc in running if proc.is_alive()]

            # Check if we can exit
            if len(running) == 0 and len(tasks) == 0:
                # All tasks have been completed
                break

            # Update the barrier
            if len(running) == 0:
                # All processes have exited so we can start more
                barrier = True

            # Check if we can start another task
            if not t.can_run.is_set():
                # The most recently started task has not yet initialized.
                continue
            if len(tasks) == 0:
                # No more tasks to run.
                continue
            if not barrier:
                # We are still waiting for processes to finish.
                continue

            # Calculate the amount of memory required to run the next task in parallel.
            unallocated_size = sw.unallocated_size
            sample_count = tasks[-1].proc.tsqr.vcf_file.sample_count
            extra_required_size = (
                (len(running) + 1) * np.float64().itemsize * sample_count**2
            )
            required_size = tasks[-1].required_size + extra_required_size
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

            proc = tasks.pop().proc
            proc.start()
            running.append(proc)

            # Reset the event so that we don't start another task before
            # this one has initialized.
            t.can_run.clear()


def wait(running: list[TriWorker]) -> None:
    for proc in running:
        proc.join(timeout=1)
        if proc.is_alive():
            break
