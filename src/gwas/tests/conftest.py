# -*- coding: utf-8 -*-
from multiprocessing import Pool
from pathlib import Path

import pytest

from gwas.utils import chromosome_to_int, chromosomes_set
from gwas.vcf import VCFFile

chromosomes = sorted(chromosomes_set(), key=chromosome_to_int)
sample_size: int = 500


@pytest.fixture(scope="session")
def vcf_paths():
    return [
        Path(f"/scratch/ds-opensnp/{sample_size}/chr{c}.dose.vcf.zst")
        for c in chromosomes
    ]


@pytest.fixture(scope="session")
def vcf_files(vcf_paths: list[Path]):
    with Pool() as pool:
        return list(pool.map(VCFFile, vcf_paths))


@pytest.fixture(scope="session")
def vcf_by_chromosome(vcf_files: list[VCFFile]):
    return {vcf_file.chromosome: vcf_file for vcf_file in vcf_files}


@pytest.fixture(scope="session")
def raw_path():
    return Path("/scratch/ds-opensnp/3421/raw")


@pytest.fixture(scope="session")
def tri_by_chromosome(vcf_paths: list[Path]):
    tris: dict[int | str, Path] = dict()
    for vcf_path in vcf_paths:
        c: int | str = vcf_path.stem.split(".")[0].removeprefix("chr")
        if isinstance(c, str) and c.isdigit():
            c = int(c)
        s = vcf_path.parent.name
        tris[c] = Path(f"/scratch/ds-opensnp/tri/{s}/chr{c}.tri.txt.gz")
    return tris
