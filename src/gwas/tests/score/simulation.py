# -*- coding: utf-8 -*-
from dataclasses import dataclass
from pathlib import Path
from random import sample, seed
from subprocess import check_call

import numpy as np
import pytest
from numpy import typing as npt

from gwas.utils import Pool

from ..conftest import DirectoryFactory
from ..utils import gcta64, is_bfile, is_pfile, plink2

minor_allele_frequency_cutoff: float = 0.001
causal_variant_count: int = 1000
heritability: float = 0.6
simulation_count: int = 16
covariate_count: int = 4
missing_value_rate: float = 0.05
missing_value_pattern_count: int = 3


@pytest.fixture(scope="module")
def pfile_paths(
    directory_factory: DirectoryFactory, sample_size: int, vcf_paths: list[Path]
):
    tmp_path = Path(directory_factory.get("pfile", sample_size))

    pfiles: list[Path] = list()
    commands: list[list[str]] = list()
    for vcf_path in vcf_paths:
        pfile_path = tmp_path / vcf_path.name.split(".")[0]
        pfiles.append(pfile_path)

        if is_pfile(pfile_path):
            continue

        commands.append(
            [
                plink2,
                "--silent",
                "--vcf",
                str(vcf_path),
                "--extract-if-info",
                "R2 >= 0.5",
                "--threads",
                str(1),
                "--memory",
                str(2**10),
                "--make-pgen",
                "--out",
                str(pfile_path),
            ]
        )

    with Pool() as pool:
        pool.map(check_call, commands)

    return pfiles


@pytest.fixture(scope="module")
def bfile_path(
    directory_factory: DirectoryFactory, sample_size: int, pfile_paths: list[Path]
):
    tmp_path = Path(directory_factory.get("bfile", sample_size))

    pfile_list_path = tmp_path / "pfile-list.txt"
    with pfile_list_path.open("wt") as file_handle:
        file_handle.write("\n".join(map(str, pfile_paths)))

    bfile_path = tmp_path / "call"
    if not is_bfile(bfile_path):
        check_call(
            [
                plink2,
                "--silent",
                "--pmerge-list",
                str(pfile_list_path),
                "--maf",
                str(minor_allele_frequency_cutoff),
                "--mind",
                "0.1",
                "--geno",
                "0.1",
                "--hwe",
                "1e-50",
                "--make-bed",
                "--out",
                str(bfile_path),
            ]
        )

    return bfile_path


@pytest.fixture(scope="module")
def variants(
    directory_factory: DirectoryFactory,
    sample_size: int,
    pfile_paths: list[Path],
):
    tmp_path = Path(directory_factory.get("variants", sample_size))

    variants: list[str] = list()
    for pfile_path in pfile_paths:
        out_path = tmp_path / pfile_path.name
        afreq_path = out_path.parent / f"{out_path.name}.afreq"

        if not afreq_path.is_file():
            check_call(
                [
                    plink2,
                    "--silent",
                    "--pfile",
                    str(pfile_path),
                    "--freq",
                    "--nonfounders",
                    "--out",
                    str(out_path),
                ]
            )

        with afreq_path.open("rt") as file_handle:
            for line in file_handle:
                if line.startswith("#"):
                    continue
                _, variant, _, _, _, _ = line.split()
                variants.append(variant)

    return variants


@dataclass
class SimulationResult:
    phen: npt.NDArray[np.str_]
    par: npt.NDArray[np.str_]


@pytest.fixture(scope="module")
def simulation(
    directory_factory: DirectoryFactory,
    sample_size: int,
    bfile_path: Path,
    variants: list[str],
):
    """
    This needs a lot of memory, so we do this before we allocate the shared workspace
    """
    tmp_path = Path(directory_factory.get("variables", sample_size))

    seed(42)
    causal_variants = sample(variants, k=causal_variant_count)
    variant_list_path = tmp_path / "causal.snplist"
    with variant_list_path.open("wt") as file_handle:
        for variant in causal_variants:
            file_handle.write(f"{variant}\n")

    simulation_path = tmp_path / "simulation"
    phen_path = simulation_path.with_suffix(".phen")
    par_path = simulation_path.with_suffix(".par")

    if not phen_path.is_file() or not par_path.is_file():
        check_call(
            [
                gcta64,
                "--bfile",
                str(bfile_path),
                "--simu-qt",
                "--simu-hsq",
                str(heritability),
                "--simu-causal-loci",
                str(variant_list_path),
                "--simu-rep",
                str(simulation_count),
                "--simu-seed",
                "101",
                "--out",
                str(simulation_path),
            ]
        )

    phen = np.loadtxt(phen_path, dtype=str)
    par = np.loadtxt(par_path, skiprows=1, dtype=str)

    patterns = [
        np.random.choice(
            a=[False, True],
            size=sample_size,
            p=[1 - missing_value_rate, missing_value_rate],
        )
        for _ in range(missing_value_pattern_count)
    ]

    for i in range(simulation_count):
        pattern_index = np.random.choice(missing_value_pattern_count)
        for j in range(sample_size):
            if patterns[pattern_index][j]:
                phen[j, 2 + i] = "NaN"

    return SimulationResult(phen, par)
