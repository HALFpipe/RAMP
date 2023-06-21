# -*- coding: utf-8 -*-

from queue import Empty

import numpy as np

from ..compression.arr.base import FileArray
from ..eig import Eigendecomposition, EigendecompositionCollection
from ..log import logger
from ..mem.arr import SharedArray
from ..vcf.base import VCFFile
from .worker import Calc, GenotypeReader, ScoreWriter, TaskSyncCollection


def calc_score(
    vcf_file: VCFFile,
    eigendecompositions: list[Eigendecomposition],
    iv_arrays: list[SharedArray],
    ivsr_arrays: list[SharedArray],
    stat_file_array: FileArray,
    phenotype_offset: int = 0,
    variant_offset: int = 0,
) -> None:
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
        t, stat_array, stat_file_array, phenotype_offset, variant_offset
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
