# -*- coding: utf-8 -*-
import gzip
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from subprocess import check_call
from typing import Mapping

import numpy as np
import pytest
from numpy import typing as npt

from gwas.eig import Eigendecomposition
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from gwas.null_model.base import NullModelCollection
from gwas.pheno import VariableCollection
from gwas.rmw import Scorefile, ScorefileHeader
from gwas.utils import Pool, chromosome_to_int, chromosomes_set
from gwas.vcf.base import VCFFile

from ..conftest import DirectoryFactory
from ..utils import rmw, to_bgzip
from .rmw_debug import rmw_debug
from .simulation import (
    SimulationResult,
    bfile_path,
    covariate_count,
    pfile_paths,
    simulation,
    simulation_count,
    variants,
)

pytest_fixtures = [bfile_path, pfile_paths, simulation, variants, rmw_debug]


@pytest.fixture(scope="module")
def other_chromosomes(chromosome: int | str) -> list[str | int]:
    return sorted(chromosomes_set() - {chromosome, "X"}, key=chromosome_to_int)


@pytest.fixture(scope="module", params=list(range(simulation_count)))
def phenotype_index(request) -> int:
    return request.param


@pytest.fixture(scope="module")
def vc(
    simulation: SimulationResult,
    sw: SharedWorkspace,
    request,
) -> VariableCollection:
    np.random.seed(47)

    samples = list(simulation.phen[:, 1])
    phenotype_names = [f"phenotype_{i + 1:02d}" for i in range(simulation_count)]
    covariate_names = [f"covariate_{i + 1:02d}" for i in range(covariate_count)]

    phenotypes = simulation.phen[:, 2:].astype(float)
    covariates = np.random.normal(size=(len(samples), covariate_count))
    covariates -= covariates.mean(axis=0)

    vc = VariableCollection.from_arrays(
        samples,
        phenotype_names,
        phenotypes,
        covariate_names,
        covariates,
        sw,
    )

    request.addfinalizer(vc.free)

    return vc


@pytest.fixture(scope="module")
def eig(
    other_chromosomes: list[str | int],
    tri_paths_by_chromosome: dict[str | int, Path],
    sw: SharedWorkspace,
    request,
) -> Eigendecomposition:
    eig = Eigendecomposition.from_files(
        *(tri_paths_by_chromosome[c] for c in other_chromosomes),
        sw=sw,
    )
    request.addfinalizer(eig.free)
    return eig


@pytest.fixture(scope="module")
def nm(
    vc: VariableCollection,
    eig: Eigendecomposition,
    request,
) -> NullModelCollection:
    nm = NullModelCollection.from_eig(eig, vc, method="fastlmm")
    request.addfinalizer(nm.free)

    return nm


@pytest.fixture(scope="module")
def vcf_file(
    chromosome: int | str, vcf_files_by_chromosome: Mapping[int | str, VCFFile]
) -> VCFFile:
    return vcf_files_by_chromosome[chromosome]


@pytest.fixture(scope="module")
def rmw_commands(
    directory_factory: DirectoryFactory,
    chromosome: int | str,
    sample_size: int,
    vcf_file: VCFFile,
    vc: VariableCollection,
    eig: Eigendecomposition,
) -> list[tuple[Path, list[str]]]:
    vcf_path = vcf_file.file_path
    rmw_path = Path(directory_factory.get("rmw", sample_size))

    if vcf_path.suffix != ".gz":
        vcf_gz_path = to_bgzip(rmw_path, vcf_path)
    else:
        vcf_gz_path = vcf_path

    def format_row(a) -> str:
        if a.size == 1:
            return np.format_float_scientific(a)
        else:
            return " ".join(map(np.format_float_scientific, a))

    kinship = (eig.eigenvectors * eig.eigenvalues) @ eig.eigenvectors.transpose()
    kinship_path = rmw_path / "Empirical.Kinship.gz"
    with gzip.open(kinship_path, "wt") as file_handle:
        file_handle.write(" ".join(vc.samples))
        file_handle.write("\n")

        for i in range(len(vc.samples)):
            row = format_row(kinship[i, : i + 1])
            file_handle.write(f"{row}\n")

    covariates = vc.covariates.to_numpy()
    phenotypes = vc.phenotypes.to_numpy()

    rmw_commands: list[tuple[Path, list[str]]] = list()

    commands_path = rmw_path / "commands.txt"
    with commands_path.open("wt") as file_handle:
        pass  # Truncate file

    for j in range(vc.phenotype_count):
        base = vc.phenotype_names[j]

        ped_path = rmw_path / f"{base}.ped"
        with ped_path.open("wt") as file_handle:
            for i, s in enumerate(vc.samples):
                c = format_row(covariates[i, 1:])  # Skip intercept
                p = format_row(phenotypes[i, j])
                file_handle.write(f"{s} {s} 0 0 1 {p} {c}\n")

        dat_path = rmw_path / f"{base}.dat"
        with dat_path.open("wt") as file_handle:
            file_handle.write(f"T {base}\n")
            for name in vc.covariate_names[1:]:  # Skip intercept
                file_handle.write(f"C {name}\n")

        prefix = str(rmw_path / f"chr{chromosome}")

        scorefile = f"{prefix}.{base}.singlevar.score.txt.gz"

        command = [
            rmw,
            "--ped",
            str(ped_path),
            "--dat",
            str(dat_path),
            "--vcf",
            str(vcf_gz_path),
            "--dosage",
            "--prefix",
            str(prefix),
            "--LDwindow",
            "100",
            "--zip",
            "--thin",
            "--kinFile",
            str(kinship_path),
            "--noPhoneHome",
            "--useCovariates",
        ]
        with commands_path.open("at") as file_handle:
            file_handle.write(" ".join(command))
            file_handle.write("\n")

        rmw_commands.append((Path(scorefile), command))

    return rmw_commands


@pytest.fixture(scope="module")
def rmw_scorefile_paths(
    rmw_commands: list[tuple[Path, list[str]]],
) -> list[Path]:
    commands: list[list[str]] = list()
    scorefiles: list[Path] = list()
    for scorefile, command in rmw_commands:
        scorefiles.append(scorefile)
        if not Path(scorefile).is_file():
            commands.append(command)

    with Pool() as pool:
        pool.map(check_call, commands)

    for path in scorefiles:
        assert path.is_file()

    return scorefiles


@dataclass
class RmwScore:
    header: list[ScorefileHeader]
    array: npt.NDArray


@cache
def read_scorefile(path: Path) -> tuple[ScorefileHeader, npt.NDArray]:
    header, array = Scorefile.read(path)
    return header, array


@pytest.fixture(scope="module")
def rmw_score(
    rmw_scorefile_paths: list[Path],
) -> RmwScore:
    headers = list()
    arrays = list()
    with Pool() as pool:
        for header, array in pool.imap(read_scorefile, rmw_scorefile_paths):
            headers.append(header)
            arrays.append(array)

    return RmwScore(
        headers,
        np.vstack(arrays).transpose(),
    )


@pytest.fixture(scope="module")
def genotypes_array(
    vcf_file: VCFFile,
    vc: VariableCollection,
    sw: SharedWorkspace,
    request,
) -> SharedArray:
    sample_count = vc.sample_count
    variant_count = sw.unallocated_size // (
        np.float64().itemsize * 2 * (vc.phenotype_count + vc.sample_count)
    )
    variant_count = min(variant_count, vcf_file.variant_count)

    vcf_file.variant_indices = vcf_file.variant_indices[:variant_count]

    name = SharedArray.get_name(sw, "genotypes")
    genotypes_array = sw.alloc(name, sample_count, variant_count)
    request.addfinalizer(genotypes_array.free)

    genotypes = genotypes_array.to_numpy()
    with vcf_file:
        vcf_file.read(genotypes.transpose())

    return genotypes_array


@pytest.fixture(scope="module")
def rotated_genotypes_array(
    genotypes_array: SharedArray,
    eig: Eigendecomposition,
    sw: SharedWorkspace,
    request,
) -> SharedArray:
    genotypes = genotypes_array.to_numpy()
    sample_count, variant_count = genotypes.shape
    assert eig.sample_count == sample_count

    name = SharedArray.get_name(sw, "rotated-genotypes")
    rotated_genotypes_array = sw.alloc(name, sample_count, variant_count)
    request.addfinalizer(rotated_genotypes_array.free)

    mean = genotypes.mean(axis=0)
    demeaned_genotypes = genotypes - mean

    rotated_genotypes = rotated_genotypes_array.to_numpy()
    rotated_genotypes[:] = eig.eigenvectors.transpose() @ demeaned_genotypes

    return rotated_genotypes_array
