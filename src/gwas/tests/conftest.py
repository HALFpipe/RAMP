# -*- coding: utf-8 -*-
import os
import shelve
from pathlib import Path
from typing import Mapping

import pytest

from gwas.mem.wkspace import SharedWorkspace
from gwas.tri import calc_tri
from gwas.utils import Pool, chromosome_to_int, chromosomes_set
from gwas.vcf import VCFFile

base_path: Path = Path(os.environ["DATA_PATH"])
dataset: str = "opensnp"
chromosomes = sorted(chromosomes_set(), key=chromosome_to_int)


@pytest.fixture(scope="session")
def cache(request) -> shelve.Shelf:
    parent_path = base_path / dataset / "pytest"
    parent_path.mkdir(exist_ok=True, parents=True)

    c = shelve.open(str(parent_path / "cache"))
    request.addfinalizer(c.close)
    return c


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


@pytest.fixture(scope="session", params=[100, 500, 3421])
def sample_size(request) -> int:
    sample_size = request.param
    return sample_size


@pytest.fixture(scope="session")
def vcf_paths_by_chromosome(sample_size: int) -> dict[int | str, Path]:
    return {
        c: (base_path / dataset / str(sample_size) / f"chr{c}.dose.vcf.zst")
        for c in chromosomes
    }


@pytest.fixture(scope="session")
def vcf_paths(vcf_paths_by_chromosome) -> list[Path]:
    return [vcf_paths_by_chromosome[c] for c in chromosomes]


@pytest.fixture(scope="session")
def vcf_files(
    cache: shelve.Shelf,
    sample_size: int,
    vcf_paths: list[Path],
) -> list[VCFFile]:
    key = f"vcf_files_{sample_size}"
    if key not in cache:
        with Pool() as pool:
            cache[key] = list(pool.map(VCFFile, vcf_paths))
    return cache[key]


@pytest.fixture(scope="session")
def vcf_by_chromosome(vcf_files: list[VCFFile]):
    return {vcf_file.chromosome: vcf_file for vcf_file in vcf_files}


@pytest.fixture(scope="session")
def raw_path():
    return base_path / dataset / "raw"


@pytest.fixture(scope="module")
def tri_paths_by_chromosome(
    vcf_by_chromosome: Mapping[int | str, VCFFile],
    sw: SharedWorkspace,
) -> Mapping[str | int, Path]:
    return calc_tri(chromosomes, vcf_by_chromosome, base_path / dataset, sw)
