# -*- coding: utf-8 -*-
import gzip
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from subprocess import check_call
from typing import IO, Mapping

import numpy as np
import pytest
from numpy import typing as npt

from gwas.eig import Eigendecomposition
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from gwas.null_model.base import NullModelCollection
from gwas.pheno import VariableCollection
from gwas.rmw import Scorefile, ScorefileHeader
from gwas.score.command import GwasCommand
from gwas.utils import (
    Pool,
    chromosome_to_int,
    chromosomes_set,
    make_sample_boolean_vectors,
    to_str,
)
from gwas.vcf.base import VCFFile

from ..conftest import DirectoryFactory
from ..utils import rmw, to_bgzip
from .rmw_debug import rmw_debug
from .simulation import (
    SimulationResult,
    bfile_path,
    covariate_count,
    missing_value_pattern_count,
    pfile_paths,
    simulation,
    simulation_count,
    variants,
)

pytest_fixtures = [bfile_path, pfile_paths, simulation, variants, rmw_debug]


@pytest.fixture(scope="session")
def other_chromosomes(chromosome: int | str) -> list[str | int]:
    return sorted(chromosomes_set() - {chromosome, "X"}, key=chromosome_to_int)


@pytest.fixture(scope="session", params=list(range(simulation_count)))
def phenotype_index(request) -> int:
    return request.param


@pytest.fixture(scope="session")
def variable_collections(
    simulation: SimulationResult,
    sw: SharedWorkspace,
    request,
) -> list[VariableCollection]:
    np.random.seed(47)

    samples = list(simulation.phen[:, 1])
    phenotype_names = [f"phenotype_{i + 1:02d}" for i in range(simulation_count)]
    covariate_names = [f"covariate_{i + 1:02d}" for i in range(covariate_count)]

    phenotypes = simulation.phen[:, 2:].astype(float)
    covariates = np.random.normal(size=(len(samples), covariate_count))
    covariates -= covariates.mean(axis=0)

    base_variable_collection = VariableCollection.from_arrays(
        samples,
        phenotype_names,
        phenotypes,
        covariate_names,
        covariates,
        sw,
        missing_value_strategy="listwise_deletion",
    )

    variable_collections = GwasCommand.split_by_missing_values(base_variable_collection)
    assert len(variable_collections) == missing_value_pattern_count

    should_be_missing = np.array(simulation.patterns)[
        simulation.pattern_indices
    ].transpose()
    assert (np.isnan(phenotypes) == should_be_missing).all()

    base_variable_collection.free()
    for i, variable_collection in enumerate(variable_collections):
        variable_collection.name = f"variableCollection-{i + 1:d}"
        request.addfinalizer(variable_collection.free)

    vc_by_phenotype = {
        phenotype_name: variable_collection
        for variable_collection in variable_collections
        for phenotype_name in variable_collection.phenotype_names
    }
    for i in range(simulation_count):
        phenotype_name = phenotype_names[i]
        vc = vc_by_phenotype[phenotype_name]
        phenotype_samples = [
            sample
            for sample, missing in zip(samples, should_be_missing[:, i])
            if not missing
        ]
        assert vc.samples == phenotype_samples

    return variable_collections


@pytest.fixture(scope="session")
def eigendecompositions(
    other_chromosomes: list[str | int],
    tri_paths_by_chromosome: dict[str | int, Path],
    variable_collections: list[VariableCollection],
    sw: SharedWorkspace,
    request,
) -> list[Eigendecomposition]:
    eigendecompositions = [
        Eigendecomposition.from_files(
            *(tri_paths_by_chromosome[c] for c in other_chromosomes),
            samples=variable_collection.samples,
            sw=sw,
        )
        for variable_collection in variable_collections
    ]

    for eigendecomposition in eigendecompositions:
        request.addfinalizer(eigendecomposition.free)

    return eigendecompositions


@pytest.fixture(scope="session")
def null_model_collections(
    variable_collections: list[VariableCollection],
    eigendecompositions: list[Eigendecomposition],
    request,
) -> list[NullModelCollection]:
    null_model_collections = [
        NullModelCollection.from_eig(
            eigendecomposition, variable_collection, method="fastlmm"
        )
        for variable_collection, eigendecomposition in zip(
            variable_collections, eigendecompositions
        )
    ]

    for null_model_collection in null_model_collections:
        request.addfinalizer(null_model_collection.free)

    return null_model_collections


@pytest.fixture(scope="session")
def vcf_file(
    chromosome: int | str, vcf_files_by_chromosome: Mapping[int | str, VCFFile]
) -> VCFFile:
    return vcf_files_by_chromosome[chromosome]


def format_row(a) -> str:
    if a.size == 1:
        return to_str(a)
    else:
        return " ".join(map(to_str, a))


@pytest.fixture(scope="session")
def rmw_commands(
    directory_factory: DirectoryFactory,
    chromosome: int | str,
    sample_size: int,
    vcf_file: VCFFile,
    variable_collections: list[VariableCollection],
    eigendecompositions: list[Eigendecomposition],
) -> list[tuple[Path, list[str]]]:
    vcf_path = vcf_file.file_path
    rmw_path = Path(directory_factory.get("rmw", sample_size))

    if vcf_path.suffix != ".gz":
        vcf_gz_path = to_bgzip(rmw_path, vcf_path)
    else:
        vcf_gz_path = vcf_path

    rmw_commands: list[tuple[Path, list[str]]] = list()

    commands_path = rmw_path / "commands.txt"
    file_handle: IO[str]
    with commands_path.open("wt") as file_handle:
        pass  # Truncate file

    for variable_collection, eigendecomposition in zip(
        variable_collections, eigendecompositions
    ):
        kinship = (
            eigendecomposition.eigenvectors * eigendecomposition.eigenvalues
        ) @ eigendecomposition.eigenvectors.transpose()
        kinship_path = rmw_path / f"{variable_collection.name}_empirical_kinship.txt.gz"
        with gzip.open(kinship_path, "wt") as file_handle:
            file_handle.write(" ".join(variable_collection.samples))
            file_handle.write("\n")

            for i in range(len(variable_collection.samples)):
                row = format_row(kinship[i, : i + 1])
                file_handle.write(f"{row}\n")

        covariates = variable_collection.covariates.to_numpy()
        phenotypes = variable_collection.phenotypes.to_numpy()

        for j in range(variable_collection.phenotype_count):
            base = variable_collection.phenotype_names[j]

            ped_path = rmw_path / f"{base}.ped"
            with ped_path.open("wt") as file_handle:
                for i, s in enumerate(variable_collection.samples):
                    c = format_row(covariates[i, 1:])  # Skip intercept
                    p = format_row(phenotypes[i, j])
                    file_handle.write(f"{s} {s} 0 0 1 {p} {c}\n")

            dat_path = rmw_path / f"{base}.dat"
            with dat_path.open("wt") as file_handle:
                file_handle.write(f"T {base}\n")
                for name in variable_collection.covariate_names[1:]:  # Skip intercept
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def genotypes_array(
    vcf_file: VCFFile,
    sw: SharedWorkspace,
    request,
) -> SharedArray:
    sample_count = len(vcf_file.samples)
    variant_count = sw.unallocated_size // (
        np.float64().itemsize * (1 + missing_value_pattern_count) * sample_count
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


@pytest.fixture(scope="session")
def rotated_genotypes_arrays(
    vcf_file: VCFFile,
    genotypes_array: SharedArray,
    eigendecompositions: list[Eigendecomposition],
    sw: SharedWorkspace,
    request,
) -> list[SharedArray]:
    genotypes = genotypes_array.to_numpy()
    _, variant_count = genotypes.shape

    sample_boolean_vectors = make_sample_boolean_vectors(
        vcf_file.samples, (eig.samples for eig in eigendecompositions)
    )

    rotated_genotypes_arrays: list[SharedArray] = list()
    for eig, sample_boolean_vector in zip(eigendecompositions, sample_boolean_vectors):
        sample_count = eig.sample_count

        name = SharedArray.get_name(sw, "rotated-genotypes")
        rotated_genotypes_array = sw.alloc(name, sample_count, variant_count)
        request.addfinalizer(rotated_genotypes_array.free)

        mean = genotypes.mean(axis=0, where=sample_boolean_vector[:, np.newaxis])
        demeaned_genotypes = genotypes[sample_boolean_vector, :] - mean

        rotated_genotypes = rotated_genotypes_array.to_numpy()
        rotated_genotypes[:] = eig.eigenvectors.transpose() @ demeaned_genotypes

        rotated_genotypes_arrays.append(rotated_genotypes_array)

    return rotated_genotypes_arrays
