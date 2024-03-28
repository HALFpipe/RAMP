# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from queue import Empty
from typing import Mapping, MutableMapping, NamedTuple, Sequence

import numpy as np

from ..log import logger
from ..mem.wkspace import SharedWorkspace
from ..vcf.base import VCFFile
from .base import TaskSyncCollection, Triangular
from .tsqr import TallSkinnyQR
from .worker import TriWorker


class Task(NamedTuple):
    required_size: int
    proc: TriWorker


def check_tri_path(
    tri_path: Path, vcf_by_chromosome: Mapping[int | str, VCFFile], sw: SharedWorkspace
) -> tuple[int | str, Path] | None:
    if not tri_path.is_file():
        return None
    try:
        tri = Triangular.from_file(tri_path, sw)
        chromosome = tri.chromosome
        samples = vcf_by_chromosome[chromosome].samples
        tri.free()
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


def check_tri_paths(
    tri_paths: list[Path] | None,
    vcf_by_chromosome: Mapping[int | str, VCFFile],
    sw: SharedWorkspace,
) -> MutableMapping[int | str, Path]:
    tri_paths_by_chromosome: MutableMapping[int | str, Path] = dict()

    if tri_paths is None:
        return dict()

    # Load from `--tri` flag.
    for tri_path in tri_paths:
        result = check_tri_path(tri_path, vcf_by_chromosome, sw)
        if result is None:
            continue
        chromosome, tri_path = result
        tri_paths_by_chromosome[chromosome] = tri_path

    return tri_paths_by_chromosome


def get_tri_tasks(
    chromosomes: Sequence[str | int],
    vcf_by_chromosome: Mapping[int | str, VCFFile],
    tri_paths_by_chromosome: MutableMapping[int | str, Path],
    output_directory: Path,
    sw: SharedWorkspace,
    minor_allele_frequency_cutoff: float,
    r_squared_cutoff: float,
) -> tuple[TaskSyncCollection, list[Task]]:
    # Ensure the output directory exists
    output_directory.mkdir(parents=True, exist_ok=True)
    t = TaskSyncCollection()
    t.can_run.set()  # We can run the first task immediately.
    # Prepare the list of tasks to run.
    tasks: list[Task] = list()
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
        result = check_tri_path(tri_path, vcf_by_chromosome, sw)
        if result is not None:
            chromosome, tri_path = result
            tri_paths_by_chromosome[chromosome] = tri_path
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
        tasks.append(Task(required_size, proc))

    return t, tasks


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
    tri_paths_by_chromosome = check_tri_paths(tri_paths, vcf_by_chromosome, sw)

    t, tasks = get_tri_tasks(
        chromosomes,
        vcf_by_chromosome,
        tri_paths_by_chromosome,
        output_directory,
        sw,
        minor_allele_frequency_cutoff,
        r_squared_cutoff,
    )

    # Sort tasks by size so that we can run the largest tasks first. This means that we
    # are less likely to run into a situation where we only run one task at a time.
    tasks.sort(key=lambda task: task.required_size)
    logger.debug(f"Will run {len(tasks)} triangularize tasks")

    running: list[TriWorker] = list()
    try:
        check_running(sw, t, tasks, running)
    finally:
        t.should_exit.set()

        for _, proc in tasks:
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


def check_running(
    sw: SharedWorkspace,
    t: TaskSyncCollection,
    tasks: list[Task],
    running: list[TriWorker],
) -> None:
    barrier: bool = True

    while True:
        # Check if an error has occurred.
        try:
            raise t.exception_queue.get_nowait()
        except Empty:
            pass

        # Sleep for one second if a process is running
        wait(running)

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


def wait(running):
    for proc in running:
        proc.join(timeout=1)
        if proc.is_alive():
            break
