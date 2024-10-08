import pytest
from upath import UPath

from gwas.testing.convert import (
    convert_vcf_to_pfiles,
    get_pfile_variant_ids,
    merge_pfiles_to_bfile,
)
from gwas.testing.simulate import SimulationResult, simulate
from gwas.utils.threads import cpu_count
from gwas.vcf.base import VCFFile

from ..conftest import DirectoryFactory

minor_allele_frequency_cutoff: float = 0.001
causal_variant_count: int = 1000
heritability: float = 0.6
base_simulation_count: int = 6
covariate_count: int = 4
missing_value_rate: float = 0.05
missing_value_pattern_count: int = 3


@pytest.fixture(scope="session", params=[base_simulation_count])
def simulation_count(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture(scope="session")
def pfile_paths(
    directory_factory: DirectoryFactory, sample_size: int, vcf_files: list[VCFFile]
) -> list[UPath]:
    tmp_path = directory_factory.get("pfile", sample_size)

    vcf_paths = [
        vcf_file.file_path for vcf_file in vcf_files if vcf_file.chromosome != "X"
    ]
    pfiles = convert_vcf_to_pfiles(vcf_paths, tmp_path, num_threads=cpu_count())

    return pfiles


@pytest.fixture(scope="session")
def bfile_path(
    directory_factory: DirectoryFactory, sample_size: int, pfile_paths: list[UPath]
) -> UPath:
    tmp_path = directory_factory.get("bfile", sample_size)

    bfile_path = tmp_path / "plink"
    merge_pfiles_to_bfile(pfile_paths, bfile_path)

    return bfile_path


@pytest.fixture(scope="session")
def simulation(
    directory_factory: DirectoryFactory,
    sample_size: int,
    bfile_path: UPath,
    pfile_paths: list[UPath],
    simulation_count: int,
) -> SimulationResult:
    """
    This needs a lot of memory, so we do this before we allocate the shared workspace
    """
    tmp_path = directory_factory.get("variables", sample_size)

    simulation_path = tmp_path / f"simulation.{simulation_count}"

    variant_ids = get_pfile_variant_ids(pfile_paths)
    return simulate(
        bfile_path=bfile_path,
        simulation_path=simulation_path,
        variant_ids=variant_ids,
        causal_variant_count=causal_variant_count,
        heritability=heritability,
        simulation_count=simulation_count,
        seed=1337,
        missing_value_rate=missing_value_rate,
        missing_value_pattern_count=missing_value_pattern_count,
    )
