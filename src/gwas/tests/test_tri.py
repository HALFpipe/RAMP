# -*- coding: utf-8 -*-
from pathlib import Path
from random import sample

import numpy as np
import pytest
from numpy import typing as npt

from gwas.mem.wkspace import SharedWorkspace
from gwas.tri import Triangular
from gwas.vcf.base import VCFFile

sample_size_label = "large"
chromosome = 22
minor_allele_frequency_cutoff = 0.05


@pytest.fixture(scope="module")
def vcf_file(
    vcf_files_by_size_and_chromosome: dict[str, dict[int | str, VCFFile]]
) -> VCFFile:
    return vcf_files_by_size_and_chromosome[sample_size_label][chromosome]


@pytest.fixture(scope="module")
def dosage_array(vcf_file: VCFFile) -> npt.NDArray:
    vcf_file.set_variants_from_cutoffs(
        minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
    )
    a = np.zeros((vcf_file.variant_count, vcf_file.sample_count))
    with vcf_file:
        vcf_file.read(a)

    mean = np.mean(a, axis=1)
    minor_allele_frequency = mean / 2
    a -= mean[:, np.newaxis]

    standard_deviation = np.sqrt(
        2 * minor_allele_frequency * (1 - minor_allele_frequency)
    )
    a /= standard_deviation[:, np.newaxis]

    return a


@pytest.fixture(scope="module")
def numpy_tri(dosage_array: npt.NDArray) -> npt.NDArray:
    a = dosage_array

    r = np.linalg.qr(a, mode="r")
    if not isinstance(r, np.ndarray):
        raise TypeError("Numpy not return an array")

    assert np.allclose(
        a.transpose() @ a,
        r.transpose() @ r,
    )

    return r


@pytest.mark.slow
def test_tri(
    vcf_file: VCFFile,
    dosage_array: npt.NDArray,
    numpy_tri: npt.NDArray,
):
    log_size = 30  # 1 gigabyte.
    sw = SharedWorkspace.create(size=2**log_size)

    vcf_file.set_variants_from_cutoffs(
        minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
    )
    assert dosage_array.shape[0] == vcf_file.variant_count

    # Check that we cannot use a direct algorithm because we would
    # run out of memory.
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
    assert np.allclose(eigenvalues, numpy_eigenvalues)

    # triangularizations are not unique, but their square is
    assert np.allclose(
        a @ a.transpose(),
        numpy_tri.transpose() @ numpy_tri,
    )

    sw.close()
    sw.unlink()


@pytest.mark.slow
def test_tri_file(tmp_path: Path, numpy_tri: npt.NDArray, sw: SharedWorkspace):
    sw = SharedWorkspace.create()
    n = numpy_tri.shape[0]

    array = sw.alloc("a", n, n)
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

    tri_path = tri.to_file(tmp_path)
    assert f"chr{tri.chromosome}" in tri_path.name

    b = Triangular.from_file(tri_path, sw)
    assert f"chr{tri.chromosome}" in b.name
    assert b.chromosome == tri.chromosome
    assert b.variant_count == tri.variant_count
    assert np.isclose(
        b.minor_allele_frequency_cutoff,
        tri.minor_allele_frequency_cutoff,
    )

    b.free()
    tri.free()
    assert len(sw.allocations) == 1


def test_tri_subset_samples(sw: SharedWorkspace):
    k = 100

    A = np.random.rand(10000, k)
    indices: list[int] = sorted(sample(range(k), 80))

    B = A[:, indices]
    C = A.transpose() @ A
    x, y = np.meshgrid(indices, indices)
    D = C[x, y]
    # Sanity check.
    assert np.allclose(B.transpose() @ B, D)

    samples = [f"sample_{i:02d}" for i in range(k)]
    subset_samples = [samples[i] for i in indices]

    R = np.linalg.qr(A, mode="r")
    assert isinstance(R, np.ndarray) and not isinstance(R, tuple)

    tri = Triangular.from_numpy(
        R,
        sw,
        chromosome=1,
        samples=samples,
        variant_count=10000,
        minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
        r_squared_cutoff=-np.inf,
    )
    assert isinstance(tri, Triangular)
    assert np.allclose(tri.to_numpy(), R)

    tri.subset_samples(subset_samples)
    R3 = tri.to_numpy()
    assert np.allclose(R3.transpose() @ R3, D)

    R1 = np.linalg.qr(B, mode="r")
    assert isinstance(R1, np.ndarray) and not isinstance(R1, tuple)
    assert np.allclose(R1.transpose() @ R1, D)

    R2 = np.linalg.qr(R[:, indices], mode="r")
    assert isinstance(R2, np.ndarray)
    assert np.allclose(R2.transpose() @ R2, D)

    tri.free()
    assert len(sw.allocations) == 1
