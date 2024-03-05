# -*- coding: utf-8 -*-
import pytest
from gwas.eig import Eigendecomposition, EigendecompositionCollection
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from gwas.null_model.base import NullModelCollection
from gwas.score.worker import Calc, TaskSyncCollection
from gwas.vcf.base import VCFFile


@pytest.mark.parametrize("chromosome", [22], indirect=True)
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
def test_calc_worker(
    vcf_file: VCFFile,
    genotypes_array: SharedArray,
    sw: SharedWorkspace,
    eig: Eigendecomposition,
    nm: NullModelCollection,
    request,
) -> None:
    allocation_count = len(sw.allocations)

    sample_count, variant_count = genotypes_array.shape
    phenotype_count = nm.phenotype_count

    name = SharedArray.get_name(sw, "test-rotated-genotypes")
    test_rotated_genotypes_array = sw.alloc(name, sample_count, variant_count)
    request.addfinalizer(test_rotated_genotypes_array.free)
    name = SharedArray.get_name(sw, "test-stat")
    stat_array: SharedArray = sw.alloc(name, 2, phenotype_count, variant_count)
    request.addfinalizer(stat_array.free)

    ec = EigendecompositionCollection.from_eigendecompositions(
        vcf_file,
        [eig],
    )

    (
        inverse_variance_array,
        scaled_residuals_array,
    ) = nm.get_arrays_for_score_calc()
    inverse_variance_arrays: list[SharedArray] = [inverse_variance_array]
    scaled_residuals_arrays: list[SharedArray] = [scaled_residuals_array]

    t = TaskSyncCollection(job_count=1)

    for can_calc in t.can_calc:
        can_calc.set()

    t.read_count_queue.put_nowait(int(variant_count))
    t.read_count_queue.put_nowait(int(0))

    calc_worker = Calc(
        t,
        genotypes_array,
        ec,
        test_rotated_genotypes_array,
        inverse_variance_arrays,
        scaled_residuals_arrays,
        stat_array,
    )
    calc_worker.func()

    assert len(sw.allocations) == allocation_count + 2
