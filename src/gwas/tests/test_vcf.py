from subprocess import check_call
from typing import Sequence

import numpy as np
import pytest
from gwas.mem.wkspace import SharedWorkspace
from gwas.testing.convert import convert_vcf_to_bgen
from gwas.tools import plink2
from gwas.vcf.base import Engine, VCFFile
from pytest_benchmark.fixture import BenchmarkFixture
from upath import UPath

from .conftest import ReadResult, SampleSizeLabel

engines: Sequence[Engine] = list(Engine.__members__.values())


@pytest.mark.parametrize("chromosome", [22], indirect=True)
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
@pytest.mark.parametrize("engine", engines)
def test_vcf_file(
    engine: Engine,
    chromosome: int | str,
    sample_size_label: SampleSizeLabel,
    sw: SharedWorkspace,
    vcf_paths_by_size_and_chromosome: dict[str, dict[int | str, UPath]],
    request: pytest.FixtureRequest,
) -> None:
    allocation_names = set(sw.allocations.keys())
    new_allocation_names: set[str] = set()

    vcf_path = vcf_paths_by_size_and_chromosome[sample_size_label][chromosome]
    vcf_file = VCFFile.from_path(vcf_path, sw, engine=engine)
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
        vcf_file.read(array3)

    assert np.allclose(array1[:1000, :], array2)
    assert np.allclose(array1[1000:2000, :], array3)

    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)


def vcf_read(engine: Engine, vcf_path: UPath, sw: SharedWorkspace) -> ReadResult:
    vcf_file = VCFFile.from_path(vcf_path, sw, engine=engine)
    dosages = np.zeros((vcf_file.variant_count, vcf_file.sample_count))
    with vcf_file:
        vcf_file.read(dosages)
    vcf_file.free()
    return ReadResult(vcf_file.vcf_variants, dosages)


@pytest.mark.parametrize("chromosome", [22], indirect=True)
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
@pytest.mark.parametrize("engine", engines)
def test_read(
    engine: Engine,
    vcf_path: UPath,
    sw: SharedWorkspace,
    numpy_read_result: ReadResult,
    benchmark: BenchmarkFixture,
) -> None:
    read_result = benchmark(vcf_read, engine, vcf_path, sw)

    assert np.all(numpy_read_result.variants == read_result.variants)
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
    converted_bgen_path = convert_vcf_to_bgen(vcf_path, converted_prefix)

    check_call(
        [
            *plink2,
            "--bgen",
            str(converted_bgen_path),
            "ref-unknown",
            "--mac",
            "1",
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

    vcf_file = VCFFile.from_path(vcf_path, sw, engine=Engine.cpp)
    request.addfinalizer(vcf_file.free)
    array1 = np.zeros((4000, vcf_file.sample_count), dtype=float)
    with vcf_file:
        vcf_file.variant_indices = variant_indices
        vcf_file.read(array1)

    vcf_file = VCFFile.from_path(converted_vcf_path, sw, engine=Engine.cpp)
    request.addfinalizer(vcf_file.free)
    array2 = np.zeros((4000, vcf_file.sample_count), dtype=float)
    with vcf_file:
        vcf_file.variant_indices = variant_indices
        vcf_file.read(array2)

    np.testing.assert_allclose(array1, array2)
