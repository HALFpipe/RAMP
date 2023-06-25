# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Literal, Mapping

import pytest

from gwas.mem.wkspace import SharedWorkspace
from gwas.tri import calc_tri
from gwas.utils import chromosome_to_int, chromosomes_set
from gwas.vcf.base import VCFFile, calc_vcf

base_path: Path = Path(os.environ["DATA_PATH"])
dataset: str = "opensnp"
chromosomes = sorted(chromosomes_set(), key=chromosome_to_int)
SampleSizeLabel = Literal["small", "medium", "large"]
sample_sizes: Mapping[SampleSizeLabel, int] = dict(small=100, medium=500, large=3421)


@pytest.fixture(scope="module")
def sw(request) -> SharedWorkspace:
    sw = SharedWorkspace.create()

    request.addfinalizer(sw.close)
    request.addfinalizer(sw.unlink)
    return sw


class DirectoryFactory:
    @staticmethod
    def get(name: str, sample_size: int | None = None) -> Path:
        p = Path(base_path / dataset / "pytest")

        if sample_size is not None:
            p = p / str(sample_size)

        p = p / name

        p.mkdir(parents=True, exist_ok=True)
        return p


@pytest.fixture(scope="session")
def directory_factory() -> DirectoryFactory:
    return DirectoryFactory()


@pytest.fixture(scope="session", params=sample_sizes.keys())
def sample_size_label(request) -> SampleSizeLabel:
    return request.param


@pytest.fixture(scope="session")
def sample_size(sample_size_label: SampleSizeLabel) -> int:
    return sample_sizes[sample_size_label]


@pytest.fixture(scope="session")
def vcf_paths_by_size_and_chromosome() -> dict[SampleSizeLabel, dict[int | str, Path]]:
    return {
        sample_size_label: {
            c: (base_path / dataset / str(sample_size) / f"chr{c}.dose.vcf.zst")
            for c in chromosomes
        }
        for sample_size_label, sample_size in sample_sizes.items()
    }


@pytest.fixture(scope="session")
def vcf_files_by_size_and_chromosome(
    vcf_paths_by_size_and_chromosome: Mapping[SampleSizeLabel, dict[int | str, Path]],
) -> Mapping[str, dict[int | str, VCFFile]]:
    vcf_files_by_size_and_chromosome: dict[str, dict[int | str, VCFFile]] = {
        sample_size_label: dict() for sample_size_label in sample_sizes.keys()
    }
    for (
        sample_size_label,
        vcf_paths_by_chromosome,
    ) in vcf_paths_by_size_and_chromosome.items():
        vcf_paths = [vcf_paths_by_chromosome[c] for c in chromosomes]
        vcf_files = calc_vcf(
            vcf_paths,
            base_path / dataset / "pytest" / str(sample_sizes[sample_size_label]),
        )
        for vcf_file in vcf_files:
            vcf_files_by_size_and_chromosome[sample_size_label][
                vcf_file.chromosome
            ] = vcf_file

    return vcf_files_by_size_and_chromosome


@pytest.fixture(scope="session")
def vcf_paths_by_chromosome(
    sample_size_label: str,
    vcf_paths_by_size_and_chromosome: Mapping[str, Mapping[int | str, Path]],
) -> Mapping[int | str, Path]:
    return vcf_paths_by_size_and_chromosome[sample_size_label]


@pytest.fixture(scope="session")
def vcf_paths(vcf_paths_by_chromosome: Mapping[int | str, Path]) -> list[Path]:
    return [vcf_paths_by_chromosome[c] for c in chromosomes]


@pytest.fixture(scope="session")
def vcf_files(
    sample_size_label: str,
    vcf_files_by_size_and_chromosome: Mapping[str, Mapping[int | str, VCFFile]],
) -> list[VCFFile]:
    return [vcf_files_by_size_and_chromosome[sample_size_label][c] for c in chromosomes]


@pytest.fixture(scope="session")
def vcf_files_by_chromosome(vcf_files: list[VCFFile]) -> Mapping[int | str, VCFFile]:
    return {vcf_file.chromosome: vcf_file for vcf_file in vcf_files}


@pytest.fixture(scope="session")
def raw_path():
    return base_path / dataset / "raw"


@pytest.fixture(scope="module")
def tri_paths_by_size_and_chromosome(
    vcf_files_by_size_and_chromosome: Mapping[
        SampleSizeLabel, Mapping[int | str, VCFFile]
    ],
    sw: SharedWorkspace,
) -> Mapping[SampleSizeLabel, Mapping[str | int, Path]]:
    return {
        sample_size_label: calc_tri(
            chromosomes,
            v,
            base_path / dataset / "tri" / f"{sample_sizes[sample_size_label]}",
            sw,
        )
        for sample_size_label, v in vcf_files_by_size_and_chromosome.items()
    }


@pytest.fixture(scope="module")
def tri_paths_by_chromosome(
    sample_size_label: str,
    tri_paths_by_size_and_chromosome: Mapping[str, Mapping[str | int, Path]],
) -> Mapping[str | int, Path]:
    return tri_paths_by_size_and_chromosome[sample_size_label]
