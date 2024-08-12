from subprocess import check_call
from typing import Sequence

import numpy as np
import pandas as pd
import pytest
from gwas.mem.wkspace import SharedWorkspace
from gwas.testing.convert import convert_vcf_to_bgen
from gwas.tools import plink2
from gwas.vcf.base import Engine, VCFFile
from pytest_benchmark.fixture import BenchmarkFixture
from upath import UPath

from .conftest import ReadResult
from .utils import check_bias

engines: Sequence[Engine] = list(Engine.__members__.values())


@pytest.mark.parametrize("chromosome", [22], indirect=True)
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
@pytest.mark.parametrize("engine", engines)
def test_vcf_file(
    engine: Engine,
    sw: SharedWorkspace,
    vcf_gz_path: UPath,
    request: pytest.FixtureRequest,
) -> None:
    allocation_names = set(sw.allocations.keys())
    new_allocation_names: set[str] = set()

    vcf_file = VCFFile.from_path(vcf_gz_path, sw, engine=engine)
    new_allocation_names.update(vcf_file.shared_vcf_variants.allocation_names)
    request.addfinalizer(vcf_file.free)

    array1 = np.zeros((4000, vcf_file.sample_count), dtype=float)

    with vcf_file:
        vcf_file.variant_indices = np.arange(4000, dtype=np.uint32)
        vcf_file.read(array1)

    assert np.abs(array1).sum() > 0

    array2 = np.zeros((1000, vcf_file.sample_count), dtype=float)
    array3 = np.zeros((1000, vcf_file.sample_count), dtype=float)

    with vcf_file:
        vcf_file.variant_indices = np.arange(1000, dtype=np.uint32)
        vcf_file.read(array2)
        vcf_file.variant_indices += 1000
        vcf_file.variant_indices = np.arange(1000, 2000, dtype=np.uint32)
        vcf_file.read(array3)

    assert np.allclose(array1[:1000, :], array2)
    assert np.allclose(array1[1000:2000, :], array3)

    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)


def vcf_read(engine: Engine, vcf_path: UPath, sw: SharedWorkspace) -> ReadResult:
    vcf_file = VCFFile.from_path(vcf_path, sw, engine=engine)
    dosages = np.zeros((vcf_file.variant_count, vcf_file.sample_count))
    with vcf_file:
        vcf_file.read(dosages)
    vcf_variants = vcf_file.vcf_variants.copy()
    vcf_file.free()
    return ReadResult(vcf_variants, dosages)


@pytest.mark.parametrize("chromosome", [22], indirect=True)
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
@pytest.mark.parametrize("engine", engines)
def test_read(
    engine: Engine,
    vcf_gz_path: UPath,
    sw: SharedWorkspace,
    numpy_read_result: ReadResult,
    benchmark: BenchmarkFixture,
) -> None:
    read_result = benchmark(vcf_read, engine, vcf_gz_path, sw)

    numpy_variants = numpy_read_result.variants
    if engine == Engine.htslib:
        if "format_str" in numpy_variants.columns:
            # We do not need the format str column in htslib hence we drop it
            numpy_variants = numpy_variants.drop(columns=["format_str"])

    pd.testing.assert_frame_equal(
        numpy_variants,
        read_result.variants,
        check_dtype=False,
        check_exact=False,
    )
    assert np.allclose(numpy_read_result.dosages, read_result.dosages)


@pytest.mark.parametrize("chromosome", [22], indirect=True)
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
def test_cpp(
    vcf_path: UPath,
    sw: SharedWorkspace,
    request: pytest.FixtureRequest,
) -> None:
    vcf_file = VCFFile.from_path(vcf_path, sw, engine=Engine.cpp)
    request.addfinalizer(vcf_file.free)

    dosages = np.zeros((vcf_file.variant_count, vcf_file.sample_count))
    with vcf_file:
        vcf_file.read(dosages)


@pytest.mark.parametrize("chromosome", [22], indirect=True)
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
def test_converted(
    vcf_path: UPath,
    chromosome: int | str,
    tmp_path: UPath,
    sw: SharedWorkspace,
    request: pytest.FixtureRequest,
) -> None:
    converted_prefix = tmp_path / f"chr{chromosome}-converted"
    bgen_path = convert_vcf_to_bgen(vcf_path, converted_prefix)

    check_call(
        [
            *plink2,
            "--bgen",
            str(bgen_path),
            "ref-unknown",
            "--export",
            "vcf",
            "vcf-dosage=DS-force",
            "bgz",
            "--out",
            str(converted_prefix),
        ]
    )
    converted_vcf_path = converted_prefix.with_suffix(".vcf.gz")
    assert converted_vcf_path.is_file()

    variant_indices = np.arange(4000, dtype=np.uint32)

    vcf_file1 = VCFFile.from_path(vcf_path, sw, engine=Engine.cpp)
    request.addfinalizer(vcf_file1.free)
    array1 = np.zeros((4000, vcf_file1.sample_count), dtype=float)
    with vcf_file1:
        vcf_file1.variant_indices = variant_indices
        vcf_file1.read(array1)

    vcf_file2 = VCFFile.from_path(converted_vcf_path, sw, engine=Engine.cpp)
    request.addfinalizer(vcf_file2.free)
    array2 = np.zeros((4000, vcf_file2.sample_count), dtype=float)
    with vcf_file2:
        vcf_file2.variant_indices = variant_indices
        vcf_file2.read(array2)

    assert check_bias(array1.ravel(), array2.ravel())
