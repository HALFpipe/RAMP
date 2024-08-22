import gzip
from dataclasses import dataclass
from functools import cache
from subprocess import check_call
from typing import Any

import numpy as np
import pytest
from numpy import typing as npt
from upath import UPath

from gwas.eig.base import Eigendecomposition
from gwas.eig.calc import calc_eigendecompositions
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from gwas.null_model.base import NullModelCollection
from gwas.null_model.calc import calc_null_model_collections
from gwas.pheno import VariableCollection
from gwas.raremetalworker.ped import format_row
from gwas.raremetalworker.score import (
    RaremetalworkerScoreCommand,
    make_raremetalworker_score_commands,
)
from gwas.raremetalworker.scorefile import Scorefile, ScorefileHeader
from gwas.score.command import GwasCommand
from gwas.testing.simulate import SimulationResult
from gwas.utils import (
    Pool,
    chromosome_to_int,
    chromosomes_set,
    cpu_count,
    get_global_lock,
)
from gwas.vcf.base import VCFFile

from ..conftest import DirectoryFactory
from .rmw_debug import rmw_debug
from .simulation import (
    bfile_path,
    covariate_count,
    missing_value_pattern_count,
    pfile_paths,
    simulation,
    simulation_count,
)

pytest_fixtures = [bfile_path, pfile_paths, simulation, rmw_debug]


@pytest.fixture(scope="session")
def other_chromosomes(chromosome: int | str) -> list[str | int]:
    return sorted(chromosomes_set() - {chromosome, "X"}, key=chromosome_to_int)


@pytest.fixture(scope="session", params=list(range(simulation_count)))
def phenotype_index(request: pytest.FixtureRequest) -> int:
    i = request.param
    assert isinstance(i, int)
    return i


@pytest.fixture(scope="session")
def variable_collections(
    simulation: SimulationResult,
    sw: SharedWorkspace,
    request: pytest.FixtureRequest,
) -> list[VariableCollection]:
    allocation_names = set(sw.allocations.keys())
    new_allocation_names: set[str] = set()

    base_variable_collection = simulation.to_variable_collection(
        sw, covariate_count=covariate_count
    )
    new_allocation_names.add(base_variable_collection.phenotypes.name)
    new_allocation_names.add(base_variable_collection.covariates.name)
    request.addfinalizer(base_variable_collection.free)

    variable_collections = GwasCommand.split_by_missing_values(base_variable_collection)
    for variable_collection in variable_collections:
        new_allocation_names.add(variable_collection.phenotypes.name)
        new_allocation_names.add(variable_collection.covariates.name)
        request.addfinalizer(variable_collection.free)
    assert len(variable_collections) == missing_value_pattern_count

    should_be_missing = simulation.patterns[:, simulation.pattern_indices]
    assert (np.isnan(simulation.phenotypes) == should_be_missing).all()

    for i, variable_collection in enumerate(variable_collections):
        variable_collection.name = f"variableCollection-{i + 1:d}"

    vc_by_phenotype = {
        phenotype_name: variable_collection
        for variable_collection in variable_collections
        for phenotype_name in variable_collection.phenotype_names
    }
    for i in range(simulation_count):
        phenotype_name = simulation.phenotype_names[i]
        vc = vc_by_phenotype[phenotype_name]
        phenotype_samples = [
            sample
            for sample, missing in zip(
                simulation.samples, should_be_missing[:, i], strict=True
            )
            if not missing
        ]
        assert vc.samples == phenotype_samples

    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)
    return variable_collections


@pytest.fixture(scope="session")
def eigendecompositions(
    chromosome: int | str,
    other_chromosomes: list[str | int],
    tri_paths_by_chromosome: dict[str | int, UPath],
    variable_collections: list[VariableCollection],
    sw: SharedWorkspace,
    request: pytest.FixtureRequest,
) -> list[Eigendecomposition]:
    allocation_names = set(sw.allocations.keys())
    new_allocation_names: set[str] = set()

    eigendecompositions = calc_eigendecompositions(
        *(tri_paths_by_chromosome[c] for c in other_chromosomes),
        sw=sw,
        samples_lists=[v.samples for v in variable_collections],
        chromosome=chromosome,
        num_threads=cpu_count(),
    )

    for eigendecomposition in eigendecompositions:
        new_allocation_names.add(eigendecomposition.name)
        request.addfinalizer(eigendecomposition.free)

    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)
    return eigendecompositions


@pytest.fixture(scope="session")
def null_model_collections(
    variable_collections: list[VariableCollection],
    eigendecompositions: list[Eigendecomposition],
    request: pytest.FixtureRequest,
) -> list[NullModelCollection]:
    null_model_collections = calc_null_model_collections(
        eigendecompositions,
        variable_collections,
        num_threads=cpu_count(),
    )

    for null_model_collection in null_model_collections:
        request.addfinalizer(null_model_collection.free)

    return null_model_collections


@pytest.fixture(scope="session")
def raremetalworker_score_commands(
    directory_factory: DirectoryFactory,
    chromosome: int | str,
    sample_size: int,
    vcf_gz_file: VCFFile,
    variable_collections: list[VariableCollection],
    eigendecompositions: list[Eigendecomposition],
) -> list[RaremetalworkerScoreCommand]:
    vcf_gz_path = vcf_gz_file.file_path
    rmw_path = directory_factory.get("rmw", sample_size)

    assert vcf_gz_path.suffix == ".gz"
    commands: list[RaremetalworkerScoreCommand] = list()
    for variable_collection, eigendecomposition in zip(
        variable_collections, eigendecompositions, strict=True
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

        commands.extend(
            make_raremetalworker_score_commands(
                chromosome,
                variable_collection,
                vcf_gz_path,
                rmw_path,
                kinship_path=kinship_path,
            )
        )

    return commands


@pytest.fixture(scope="session")
def raremetalworker_scorefile_paths(
    raremetalworker_score_commands: list[RaremetalworkerScoreCommand],
) -> list[UPath]:
    commands: list[list[str]] = list()
    scorefile_paths: list[UPath] = list()
    for command, scorefile_path in raremetalworker_score_commands:
        scorefile_paths.append(scorefile_path)
        if not scorefile_path.is_file():
            commands.append(command)

    with Pool() as pool:
        pool.map(check_call, commands)

    for path in scorefile_paths:
        assert path.is_file()

    return scorefile_paths


@dataclass
class RmwScore:
    header: list[ScorefileHeader]
    array: npt.NDArray[Any]


@cache
def read_scorefile(path: UPath) -> tuple[ScorefileHeader, npt.NDArray[np.float64]]:
    header, array = Scorefile.read(path)
    return header, array


@pytest.fixture(scope="session")
def rmw_score(
    raremetalworker_scorefile_paths: list[UPath],
) -> RmwScore:
    headers = list()
    arrays = list()
    with Pool() as pool:
        for header, array in pool.imap(read_scorefile, raremetalworker_scorefile_paths):
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
    request: pytest.FixtureRequest,
) -> SharedArray:
    allocation_names = set(sw.allocations.keys())

    sample_count = len(vcf_file.samples)
    variant_count = sw.unallocated_size // (
        np.float64().itemsize * (1 + missing_value_pattern_count) * sample_count
    )
    variant_count = min(variant_count, vcf_file.variant_count)

    vcf_file.variant_indices = vcf_file.variant_indices[:variant_count]
    with get_global_lock():
        name = SharedArray.get_name(sw, "genotypes")
        genotypes_array = sw.alloc(name, sample_count, variant_count)
    request.addfinalizer(genotypes_array.free)

    genotypes = genotypes_array.to_numpy()
    with vcf_file:
        vcf_file.read(genotypes.transpose())

    new_allocation_names = {name}
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)
    return genotypes_array
