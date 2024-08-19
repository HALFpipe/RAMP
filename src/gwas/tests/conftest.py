# -*- coding: utf-8 -*-
import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import Literal, Mapping

import pytest
from psutil import virtual_memory
from pytest import FixtureRequest

from gwas.compression.convert import to_bgzip
from gwas.log import add_handler, setup_logging_queue, teardown_logging
from gwas.mem.wkspace import SharedWorkspace
from gwas.tri.calc import calc_tri
from gwas.utils import apply_num_threads, chromosome_to_int, chromosomes_set
from gwas.vcf.base import VCFFile, calc_vcf, load_vcf

base_path: Path = Path(os.environ["DATA_PATH"])
dataset: str = "opensnp"
chromosomes = sorted(chromosomes_set(), key=chromosome_to_int)
SampleSizeLabel = Literal["small", "medium", "large"]
sample_sizes: Mapping[SampleSizeLabel, int] = dict(small=100, medium=500, large=3421)


@pytest.fixture(scope="session", autouse=True)
def num_threads() -> None:
    apply_num_threads(cpu_count())


@pytest.fixture(scope="session", autouse=True)
def logging(request: FixtureRequest) -> None:
    logging_plugin = request.config.pluginmanager.get_plugin("logging-plugin")
    if logging_plugin is None:
        raise ValueError("Logging plugin not found")
    add_handler(logging_plugin.log_cli_handler)
    setup_logging_queue()
    request.addfinalizer(teardown_logging)


@pytest.fixture(scope="session", params=[22, "X"])
def chromosome(request: FixtureRequest) -> str | int:
    chromosome = request.param
    assert isinstance(chromosome, (int, str))
    return chromosome


@pytest.fixture(scope="session")
def sw(request: FixtureRequest) -> SharedWorkspace:
    if "SLURM_MEM_PER_CPU" in os.environ:
        size_per_cpu = int(os.environ["SLURM_MEM_PER_CPU"]) << 20
        size = size_per_cpu * int(os.environ["SLURM_CPUS_ON_NODE"])
    else:
        size = virtual_memory().available
    size = int(size * (2 / 3))
    size = min(size, 48 * 2**30)
    sw = SharedWorkspace.create(size=size)

    request.addfinalizer(sw.close)
    request.addfinalizer(sw.unlink)
    return sw


class DirectoryFactory:
    @staticmethod
    def get(
        name: str | None = None,
        sample_size: int | None = None,
        sample_size_label: SampleSizeLabel | None = None,
    ) -> Path:
        p = Path(base_path / dataset / "pytest")

        if sample_size_label is not None:
            if sample_size is not None:
                raise ValueError(
                    "sample_size and sample_size_label are mutually exclusive"
                )
            sample_size = sample_sizes[sample_size_label]
        if sample_size is not None:
            p = p / str(sample_size)

        if name is not None:
            p = p / name

        p.mkdir(parents=True, exist_ok=True)
        return p


@pytest.fixture(scope="session")
def directory_factory() -> DirectoryFactory:
    return DirectoryFactory()


@pytest.fixture(scope="session", params=sample_sizes.keys())
def sample_size_label(request: FixtureRequest) -> SampleSizeLabel:
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
def cache_path_by_size() -> Mapping[SampleSizeLabel, Path]:
    return {
        sample_size_label: DirectoryFactory.get(
            sample_size=sample_sizes[sample_size_label]
        )
        for sample_size_label in sample_sizes.keys()
    }


@pytest.fixture(scope="session")
def vcf_files_by_size_and_chromosome(
    vcf_paths_by_size_and_chromosome: Mapping[SampleSizeLabel, dict[int | str, Path]],
    cache_path_by_size: Mapping[SampleSizeLabel, Path],
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
            cache_path=cache_path_by_size[sample_size_label],
            num_threads=cpu_count(),
        )
        for vcf_file in vcf_files:
            vcf_files_by_size_and_chromosome[sample_size_label][vcf_file.chromosome] = (
                vcf_file
            )

    return vcf_files_by_size_and_chromosome


@pytest.fixture(scope="session")
def cache_path(
    sample_size_label: SampleSizeLabel,
    cache_path_by_size: Mapping[SampleSizeLabel, Path],
) -> Path:
    return cache_path_by_size[sample_size_label]


@pytest.fixture(scope="session")
def vcf_paths_by_chromosome(
    sample_size_label: SampleSizeLabel,
    vcf_paths_by_size_and_chromosome: Mapping[SampleSizeLabel, Mapping[int | str, Path]],
) -> Mapping[int | str, Path]:
    return vcf_paths_by_size_and_chromosome[sample_size_label]


@pytest.fixture(scope="session")
def vcf_paths(vcf_paths_by_chromosome: Mapping[int | str, Path]) -> list[Path]:
    return [vcf_paths_by_chromosome[c] for c in chromosomes]


@pytest.fixture(scope="session")
def vcf_files(
    sample_size_label: SampleSizeLabel,
    vcf_files_by_size_and_chromosome: Mapping[
        SampleSizeLabel, Mapping[int | str, VCFFile]
    ],
) -> list[VCFFile]:
    return [vcf_files_by_size_and_chromosome[sample_size_label][c] for c in chromosomes]


@pytest.fixture(scope="session")
def vcf_files_by_chromosome(vcf_files: list[VCFFile]) -> Mapping[int | str, VCFFile]:
    return {vcf_file.chromosome: vcf_file for vcf_file in vcf_files}


@pytest.fixture(scope="session")
def raw_path() -> Path:
    return base_path / dataset / "raw"


@pytest.fixture(scope="session")
def tri_paths_by_size_and_chromosome(
    vcf_files_by_size_and_chromosome: Mapping[
        SampleSizeLabel, Mapping[int | str, VCFFile]
    ],
    cache_path_by_size: Mapping[SampleSizeLabel, Path],
    sw: SharedWorkspace,
) -> Mapping[SampleSizeLabel, Mapping[str | int, Path]]:
    allocation_names = set(sw.allocations.keys())
    tri_paths_by_size_and_chromosome: Mapping[
        SampleSizeLabel, Mapping[str | int, Path]
    ] = {
        sample_size_label: calc_tri(
            chromosomes=chromosomes,
            vcf_by_chromosome=v,
            output_directory=cache_path_by_size[sample_size_label],
            sw=sw,
            num_threads=cpu_count(),
        )
        for sample_size_label, v in vcf_files_by_size_and_chromosome.items()
    }
    assert set(sw.allocations.keys()) <= allocation_names
    return tri_paths_by_size_and_chromosome


@pytest.fixture(scope="session")
def tri_paths_by_chromosome(
    sample_size_label: SampleSizeLabel,
    tri_paths_by_size_and_chromosome: Mapping[SampleSizeLabel, Mapping[str | int, Path]],
) -> Mapping[str | int, Path]:
    tri_paths_by_chromosome = tri_paths_by_size_and_chromosome[sample_size_label]
    assert len(tri_paths_by_chromosome) > 0
    return tri_paths_by_chromosome


@pytest.fixture(scope="session")
def vcf_file(
    chromosome: int | str,
    vcf_files_by_chromosome: Mapping[int | str, VCFFile],
) -> VCFFile:
    return vcf_files_by_chromosome[chromosome]


@pytest.fixture(scope="session")
def vcf_gz_file(
    vcf_file: VCFFile,
    sample_size_label: SampleSizeLabel,
    cache_path_by_size: Mapping[SampleSizeLabel, Path],
    directory_factory: DirectoryFactory,
) -> VCFFile:
    tmp_path = directory_factory.get("bgzip", sample_size_label=sample_size_label)
    vcf_gz_path = to_bgzip(vcf_file.file_path, tmp_path)
    vcf_file = load_vcf(cache_path_by_size[sample_size_label], vcf_gz_path)
    vcf_file.file_path = vcf_gz_path
    return vcf_file
