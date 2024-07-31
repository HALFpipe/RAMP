from queue import Empty
from typing import Callable

import numpy as np
from tqdm import tqdm

from ..compression.arr.base import FileArrayWriter
from ..eig.base import Eigendecomposition
from ..eig.collection import EigendecompositionCollection
from ..log import logger
from ..mem.arr import SharedArray
from ..utils import soft_close
from ..vcf.base import VCFFile
from .worker import (
    Calc,
    GenotypeReader,
    ScoreWriter,
    TaskProgress,
    TaskSyncCollection,
    Worker,
)


def calc_score(
    vcf_file: VCFFile,
    eigendecompositions: list[Eigendecomposition],
    inverse_variance_arrays: list[SharedArray],
    scaled_residuals_arrays: list[SharedArray],
    stat_file_array: FileArrayWriter[np.float64],
    phenotype_offset: int = 0,
    variant_offset: int = 0,
    num_threads: int = 1,
) -> None:
    # Merge the eigenvector arrays so that we can use a single reader process.
    job_count = len(eigendecompositions)
    ec = EigendecompositionCollection.from_eigendecompositions(
        vcf_file,
        eigendecompositions,
    )
    for eig in eigendecompositions:
        eig.free()
    vcf_file.set_samples(set(ec.samples))
    if vcf_file.samples != ec.samples:
        raise ValueError(
            "Sample order of eigendecompositions does not match the VCF file"
        )
    # Make sure that we can use all free memory.
    sw = ec.eigenvector_arrays[0].sw
    sw.squash()
    # We reuse sample x genotype matrix across all jobs, so we need to use
    # the total number of samples.
    sample_count = vcf_file.sample_count
    phenotype_count = sum(
        inverse_variance_array.shape[1]
        for inverse_variance_array in inverse_variance_arrays
    )
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
    genotypes_array = sw.alloc(name, sample_count, variant_count)
    name = SharedArray.get_name(sw, "rotated-genotypes")
    rotated_genotypes_array = sw.alloc(name, sample_count, variant_count)
    name = SharedArray.get_name(sw, "stat")
    stat_array: SharedArray = sw.alloc(name, 2, phenotype_count, variant_count)
    # Create the worker processes.
    t = TaskSyncCollection(job_count=job_count)
    reader_proc = GenotypeReader(t, vcf_file, genotypes_array)
    calc_proc = Calc(
        t,
        genotypes_array,
        ec,
        rotated_genotypes_array,
        inverse_variance_arrays,
        scaled_residuals_arrays,
        stat_array,
        num_threads,
    )
    writer_proc = ScoreWriter(
        t, stat_array, stat_file_array, phenotype_offset, variant_offset
    )
    # Start the loop
    procs = [reader_proc, calc_proc, writer_proc]
    with tqdm(
        total=vcf_file.variant_count,
        unit="variants",
        desc="calculating score statistics",
        leave=False,
    ) as progress_bar:

        def update_progress_bar() -> None:
            try:
                task_progress: TaskProgress = t.progress_queue.get_nowait()
                progress_bar.update(task_progress.variant_count)
            except Empty:
                pass

        try:
            check_procs(t, procs, update_progress_bar)
        finally:
            t.should_exit.set()
            for proc in procs:
                soft_close(proc)
            ec.free()
            genotypes_array.free()
            rotated_genotypes_array.free()
            stat_array.free()
            update_progress_bar()


def check_procs(
    t: TaskSyncCollection, procs: list[Worker], update_progress_bar: Callable[[], None]
) -> None:
    for proc in procs:
        proc.start()
        # Allow use of genotypes_array and stat_array.
    t.can_read.set()
    for can_calc in t.can_calc:
        can_calc.set()
    while True:
        # Error handling
        try:
            raise t.exception_queue.get_nowait()
        except Empty:
            pass
        for proc in procs:
            if proc.exitcode is not None and proc.exitcode != 0:
                raise RuntimeError(
                    f'Process "{proc.name}" exited with code {proc.exitcode}'
                )
                # Check if we are done
        update_progress_bar()
        for proc in procs:
            proc.join(timeout=1)
        if all(not proc.is_alive() for proc in procs):
            break
    update_progress_bar()
