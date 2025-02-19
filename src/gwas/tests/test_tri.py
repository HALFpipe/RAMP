from random import sample

import numpy as np
import pytest
from numpy import typing as npt
from upath import UPath

from gwas.mem.wkspace import SharedWorkspace
from gwas.tri.base import Triangular, is_lower_triangular
from gwas.tri.tsqr import scale
from gwas.utils.threads import cpu_count
from gwas.vcf.base import VCFFile

minor_allele_frequency_cutoff: float = 0.05


@pytest.fixture(scope="module")
def genotypes_array(vcf_file: VCFFile) -> npt.NDArray[np.float64]:
    vcf_file.set_variants_from_cutoffs(
        minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
    )
    a = np.zeros((vcf_file.variant_count, vcf_file.sample_count))
    with vcf_file:
        vcf_file.read(a)

    scale(vcf_file, a)

    return a


@pytest.fixture(scope="module")
def numpy_tri(genotypes_array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    a = genotypes_array

    r = np.linalg.qr(a, mode="r")
    if not isinstance(r, np.ndarray):
        raise TypeError("Numpy not return an array")

    np.testing.assert_allclose(
        a.transpose() @ a,
        r.transpose() @ r,
    )

    return r


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
@pytest.mark.parametrize("chromosome", [22], indirect=True)
def test_zero_variance(
    vcf_file: VCFFile, genotypes_array: npt.NDArray[np.float64]
) -> None:
    a = genotypes_array.copy()
    a[42, :] = 0

    with pytest.raises(ValueError):
        scale(vcf_file, a)


@pytest.mark.slow
@pytest.mark.parametrize("sample_size_label", ["large"], indirect=True)
@pytest.mark.parametrize("chromosome", [22], indirect=True)
def test_tri(
    vcf_file: VCFFile,
    genotypes_array: npt.NDArray[np.float64],
    numpy_tri: npt.NDArray[np.float64],
) -> None:
    log_size = 30  # 1 gigabyte
    sw = SharedWorkspace.create(size=2**log_size)

    vcf_file.set_variants_from_cutoffs(
        minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
    )
    assert genotypes_array.shape[0] == vcf_file.variant_count

    # Check that we cannot use a direct algorithm because we would
    # run out of memory
    with pytest.raises(MemoryError):
        sw.alloc("a", vcf_file.variant_count, vcf_file.sample_count)

    tri = Triangular.from_vcf(
        vcf_file=vcf_file,
        sw=sw,
    )
    assert isinstance(tri, Triangular)
    a = tri.to_numpy()

    eigenvalues = np.linalg.svd(a, full_matrices=False, compute_uv=False)
    numpy_eigenvalues = np.linalg.svd(numpy_tri, full_matrices=False, compute_uv=False)
    np.testing.assert_allclose(eigenvalues, numpy_eigenvalues, rtol=1e-5, atol=1e-8)

    # triangularizations are not unique, but their square is
    np.testing.assert_allclose(
        a @ a.transpose(),
        numpy_tri.transpose() @ numpy_tri,
    )

    sw.close()


@pytest.mark.slow
@pytest.mark.parametrize("sample_size_label", ["large"], indirect=True)
@pytest.mark.parametrize("chromosome", [22], indirect=True)
def test_tri_file(
    tmp_path: UPath,
    numpy_tri: npt.NDArray[np.float64],
    sw: SharedWorkspace,
    request: pytest.FixtureRequest,
) -> None:
    allocation_names = set(sw.allocations.keys())

    n = numpy_tri.shape[0]

    array = sw.alloc("a", n, n)
    request.addfinalizer(array.free)
    a = array.to_numpy()
    a[:] = numpy_tri.transpose()

    samples = [f"sample_{i}" for i in range(n)]

    tri = Triangular(
        name=array.name,
        sw=sw,
        chromosome=19,
        samples=samples,
        variant_count=199357,
        minor_allele_frequency_cutoff=0.01,
        r_squared_cutoff=-np.inf,
    )

    tri_path = tri.to_file(tmp_path, num_threads=cpu_count())
    assert f"chr{tri.chromosome}" in tri_path.name

    b = Triangular.from_file(tri_path, sw, np.float64)
    request.addfinalizer(b.free)
    assert f"chr{tri.chromosome}" in b.name
    assert b.samples == tri.samples
    assert b.chromosome == tri.chromosome
    assert b.variant_count == tri.variant_count
    assert np.isclose(
        b.minor_allele_frequency_cutoff,
        tri.minor_allele_frequency_cutoff,
    )

    size = tri_path.stat().st_size
    trunc_path = tmp_path / f"truncated-{tri_path.name}"
    with tri_path.open("rb") as src, trunc_path.open("wb") as dst:
        dst.write(src.read(size // 2))

    with pytest.raises(ValueError):
        _ = Triangular.from_file(trunc_path, sw, np.float64)

    corrupted_path = tmp_path / f"corrupted-{tri_path.name}"
    with tri_path.open("rb") as src, corrupted_path.open("wb") as dst:
        data = bytearray(src.read())
        for i in range(18, len(data), 4):
            data[i] = 0
        dst.write(data)

    with pytest.raises(ValueError):
        _ = Triangular.from_file(corrupted_path, sw, np.float64)

    new_allocation_names = {b.name, tri.name}
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)


def test_tri_subset_samples(sw: SharedWorkspace, request: pytest.FixtureRequest) -> None:
    allocation_names = set(sw.allocations.keys())

    k = 100

    a = np.random.rand(10000, k)
    indices: list[int] = sorted(sample(range(k), 80))

    b = a[:, indices]
    c = a.transpose() @ a
    x, y = np.meshgrid(indices, indices)
    d = c[x, y]
    # Sanity check.
    np.testing.assert_allclose(b.transpose() @ b, d)

    samples = [f"sample_{i:02d}" for i in range(k)]
    subset_samples = [samples[i] for i in indices]
    assert len(subset_samples) < len(samples)

    r = np.linalg.qr(a, mode="r")
    assert isinstance(r, np.ndarray) and not isinstance(r, tuple)
    r = r.transpose()  # Ensure that we have a lower triangular matrix
    assert is_lower_triangular(r)

    r1 = np.linalg.qr(b, mode="r")
    assert isinstance(r1, np.ndarray) and not isinstance(r1, tuple)
    r1 = r1.transpose()  # Ensure that we have a lower triangular matrix
    assert is_lower_triangular(r1)
    np.testing.assert_allclose(r1 @ r1.transpose(), d)

    r2 = np.linalg.qr(r[indices, :].transpose(), mode="r")
    assert isinstance(r2, np.ndarray)
    r2 = r2.transpose()  # Ensure that we have a lower triangular matrix
    assert is_lower_triangular(r2)
    np.testing.assert_allclose(np.abs(r2), np.abs(r1))
    np.testing.assert_allclose(r2 @ r2.transpose(), d)

    tri = Triangular.from_numpy(
        r,
        sw,
        chromosome=1,
        samples=samples,
        variant_count=10000,
        minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
        r_squared_cutoff=-np.inf,
    )
    request.addfinalizer(tri.free)

    assert isinstance(tri, Triangular)
    assert is_lower_triangular(tri.to_numpy())
    np.testing.assert_allclose(tri.to_numpy(), r)

    tri.subset_samples(subset_samples)
    r3 = tri.to_numpy()
    np.testing.assert_allclose(r3 @ r3.transpose(), d)

    r4 = np.linalg.qr(r3.transpose(), mode="r")
    assert isinstance(r4, np.ndarray)
    r4 = r4.transpose()  # Ensure that we have a lower triangular matrix
    assert is_lower_triangular(r4)
    np.testing.assert_allclose(np.abs(r4), np.abs(r1))

    new_allocation_names = {tri.name}
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)
