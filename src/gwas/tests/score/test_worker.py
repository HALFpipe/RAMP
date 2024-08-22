import pytest

from gwas.eig.base import Eigendecomposition
from gwas.eig.collection import EigendecompositionCollection
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from gwas.null_model.base import NullModelCollection
from gwas.score.worker import Calc, TaskSyncCollection
from gwas.utils import cpu_count, get_global_lock
from gwas.vcf.base import VCFFile


@pytest.mark.parametrize("chromosome", [22], indirect=True)
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
def test_calc_worker(
    vcf_file: VCFFile,
    genotypes_array: SharedArray,
    sw: SharedWorkspace,
    eigendecompositions: list[Eigendecomposition],
    null_model_collections: list[NullModelCollection],
    request: pytest.FixtureRequest,
) -> None:
    allocation_names = set(sw.allocations.keys())

    eig = eigendecompositions[0]
    nm = null_model_collections[0]

    sample_count, variant_count = genotypes_array.shape
    phenotype_count = nm.phenotype_count

    with get_global_lock():
        name = SharedArray.get_name(sw, "test-rotated-genotypes")
        test_rotated_genotypes_array = sw.alloc(name, sample_count, variant_count)
        name = SharedArray.get_name(sw, "test-stat")
        stat_array: SharedArray = sw.alloc(name, variant_count, phenotype_count * 2)
    request.addfinalizer(test_rotated_genotypes_array.free)
    request.addfinalizer(stat_array.free)

    ec = EigendecompositionCollection.from_eigendecompositions(
        vcf_file, [eig], base_samples=vcf_file.samples
    )
    request.addfinalizer(ec.free)

    (
        inverse_variance_array,
        scaled_residuals_array,
    ) = nm.get_arrays_for_score_calc()
    request.addfinalizer(inverse_variance_array.free)
    request.addfinalizer(scaled_residuals_array.free)
    inverse_variance_arrays: list[SharedArray] = [inverse_variance_array]
    scaled_residuals_arrays: list[SharedArray] = [scaled_residuals_array]

    t = TaskSyncCollection(job_count=1)

    for can_calc in t.can_calc:
        can_calc.set()

    t.read_count_queue.put(int(variant_count))
    t.read_count_queue.put(int(0))

    calc_worker = Calc(
        t,
        genotypes_array,
        ec,
        test_rotated_genotypes_array,
        inverse_variance_arrays,
        scaled_residuals_arrays,
        stat_array,
        num_threads=cpu_count(),
    )
    calc_worker.func()

    new_allocation_names = {
        test_rotated_genotypes_array.name,
        stat_array.name,
        ec.eigenvector_arrays[0].name,
        inverse_variance_array.name,
        scaled_residuals_array.name,
    }
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)
