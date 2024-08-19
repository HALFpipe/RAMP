# -*- coding: utf-8 -*-
from pathlib import Path
from subprocess import check_call
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
import pytest
from numpy import typing as npt
from pytest_benchmark.fixture import BenchmarkFixture
from tqdm.auto import tqdm

from gwas.compression.pipe import CompressedTextReader
from gwas.testing.convert import convert_vcf_to_bgen
from gwas.tools import plink2
from gwas.vcf.base import Engine, VCFFile
from gwas.vcf.variant import Variant

sample_size_label = "small"
chromosome: int = 22
engines: Sequence[Engine] = list(Engine.__members__.values())


@pytest.mark.parametrize("engine", engines)
def test_vcf_file(
    engine: Engine,
    vcf_paths_by_size_and_chromosome: dict[str, dict[int | str, Path]],
) -> None:
    vcf_path = vcf_paths_by_size_and_chromosome[sample_size_label][chromosome]
    vcf_file = VCFFile.from_path(vcf_path, engine=engine)

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


@pytest.fixture(scope="session")
def vcf_path(vcf_paths_by_size_and_chromosome: dict[str, dict[int | str, Path]]) -> Path:
    vcf_path = vcf_paths_by_size_and_chromosome[sample_size_label][chromosome]
    return vcf_path


class ReadResult(NamedTuple):
    variants: pd.DataFrame
    dosages: npt.NDArray[np.float64]


@pytest.fixture(scope="session")
def numpy_read_result(vcf_path: Path) -> ReadResult:
    with CompressedTextReader(vcf_path) as file_handle:
        array = np.loadtxt(file_handle, dtype=object)

    vcf_variants: list[Variant] = list()
    vcf_dosages = np.zeros(
        (array.shape[0], array.shape[1] - len(VCFFile.mandatory_columns))
    )
    for i, row in enumerate(tqdm(array)):
        variant = Variant.from_metadata_columns(*row[VCFFile.metadata_column_indices])
        vcf_variants.append(variant)
        genotype_fields = variant.format_str.split(":")
        dosage_field_index = genotype_fields.index("DS")
        for j, dosage in enumerate(row[len(VCFFile.mandatory_columns) :]):
            vcf_dosages[i, j] = float(dosage.split(":")[dosage_field_index])

    return ReadResult(VCFFile.make_data_frame(vcf_variants), vcf_dosages)


def vcf_read(engine: Engine, vcf_path: Path) -> ReadResult:
    vcf_file = VCFFile.from_path(vcf_path, engine=engine)
    dosages = np.zeros((vcf_file.variant_count, vcf_file.sample_count))
    with vcf_file:
        vcf_file.read(dosages)

    return ReadResult(vcf_file.vcf_variants, dosages)


@pytest.mark.parametrize("engine", engines)
def test_read(
    benchmark: BenchmarkFixture,
    vcf_path: Path,
    numpy_read_result: ReadResult,
    engine: Engine,
) -> None:
    read_result = benchmark(vcf_read, engine, vcf_path)

    assert np.all(numpy_read_result.variants == read_result.variants)
    assert np.allclose(numpy_read_result.dosages, read_result.dosages)


def test_cpp(vcf_paths_by_size_and_chromosome: dict[str, dict[int | str, Path]]) -> None:
    vcf_path = vcf_paths_by_size_and_chromosome["small"][chromosome]
    vcf_file = VCFFile.from_path(vcf_path, engine=Engine.cpp)

    # from pympler import muppy, summary

    # all_objects = muppy.get_objects()
    # rows = summary.summarize(all_objects)
    # memtrace = "\n".join(summary.format_(rows))
    # print(memtrace)

    dosages = np.zeros((vcf_file.variant_count, vcf_file.sample_count))
    with vcf_file:
        vcf_file.read(dosages)


def test_converted(vcf_path: Path, tmp_path: Path) -> None:
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

    vcf_file = VCFFile.from_path(vcf_path, engine=Engine.cpp)
    array1 = np.zeros((4000, vcf_file.sample_count), dtype=float)
    with vcf_file:
        vcf_file.variant_indices = variant_indices
        vcf_file.read(array1)

    vcf_file = VCFFile.from_path(converted_vcf_path, engine=Engine.cpp)
    array2 = np.zeros((4000, vcf_file.sample_count), dtype=float)
    with vcf_file:
        vcf_file.variant_indices = variant_indices
        vcf_file.read(array2)
