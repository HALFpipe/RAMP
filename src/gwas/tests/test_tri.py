# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pytest

from gwas.mem.wkspace import SharedWorkspace
from gwas.tri import Triangular
from gwas.vcf import VCFFile

vcf_path_zstd = Path("~/work/opensnp/3421/chr22.dose.vcf.zst")
minor_allele_frequency_cutoff = 0.05


@pytest.fixture(scope="module")
def vcf_file():
    vcf_file = VCFFile(vcf_path_zstd)
    return vcf_file


@pytest.fixture(scope="module")
def numpy_tri(vcf_file):
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

    # direct computation for better precision
    # https://doi.org/10.1145/1996092.1996103
    r = np.linalg.qr(a, mode="r")

    return r


@pytest.mark.slow
@pytest.mark.parametrize("log_size", [34, 32, 30])
def test_tri(vcf_file, numpy_tri, log_size):
    sw = SharedWorkspace.create(size=2**log_size)

    # check that we indeed cannot use a direct algorithm
    with pytest.raises(MemoryError):
        sw.alloc("a", vcf_file.variant_count, vcf_file.sample_count)

    tri = Triangular.from_vcf(
        vcf_file=vcf_file,
        sw=sw,
        minor_allele_frequency_cutoff=minor_allele_frequency_cutoff,
    )
    assert isinstance(tri, Triangular)
    a = tri.to_numpy()

    # triangularizations are not unique, but their square is
    assert np.allclose(
        a @ a.transpose(),
        numpy_tri.transpose() @ numpy_tri,
    )

    sw.close()
    sw.unlink()


@pytest.mark.slow
def test_tri_file(tmp_path, numpy_tri):
    sw = SharedWorkspace.create()

    n = numpy_tri.shape[0]

    array = sw.alloc("a", n, n)
    a = array.to_numpy()
    a[:] = numpy_tri.transpose()

    tri = Triangular(
        name=array.name,
        sw=sw,
        chromosome=19,
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
