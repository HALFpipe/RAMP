# -*- coding: utf-8 -*-
from pathlib import Path
from random import sample

import numpy as np
import pytest
from numpy import typing as npt

from gwas.mem.wkspace import SharedWorkspace
from gwas.tri import Triangular
from gwas.vcf import VCFFile

chromosome = 22
minor_allele_frequency_cutoff = 0.05


@pytest.fixture(scope="module")
def vcf_file(vcf_by_chromosome: dict[int | str, VCFFile]) -> VCFFile:
    return vcf_by_chromosome[chromosome]


@pytest.fixture(scope="module")
def dosage_array(vcf_file: VCFFile) -> npt.NDArray:
    a = np.zeros((vcf_file.variant_count, vcf_file.sample_count))

    with vcf_file:
        vcf_file.read(a)

    # calculate variant properties
    minor_allele_frequency = np.mean(a, axis=1) / 2

    # create filter vector
    include = np.ones(len(minor_allele_frequency), dtype=bool)

    include &= minor_allele_frequency >= minor_allele_frequency_cutoff
    include &= minor_allele_frequency <= (1 - minor_allele_frequency_cutoff)

    # apply scaling to variants that pass the filter
    a = a[include, :]

    mean = 2 * minor_allele_frequency
    a -= mean[include, np.newaxis]

    standard_deviation = np.sqrt(
        2 * minor_allele_frequency * (1 - minor_allele_frequency)
    )
    a /= standard_deviation[include, np.newaxis]

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
@pytest.mark.parametrize("log_size", [34, 30])
def test_tri(
    vcf_file: VCFFile, numpy_tri: npt.NDArray, sample_size: int, log_size: int
):
    sw = SharedWorkspace.create(size=2**log_size)

    if sample_size == 3421:
        # Check that we cannot use a direct algorithm because we would
        # run out of memory.
        with pytest.raises(ValueError):
            sw.alloc("a", vcf_file.variant_count, vcf_file.sample_count)

    tri = Triangular.from_vcf(
        vcf_file=vcf_file,
        sw=sw,
        minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
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
def test_tri_file(tmp_path: Path, numpy_tri: npt.NDArray):
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

    sw.close()
    sw.unlink()


def test_tri_subset_samples():
    sw = SharedWorkspace.create()

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
        R, sw, chromosome=1, samples=samples, variant_count=10000
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

    sw.close()
    sw.unlink()
