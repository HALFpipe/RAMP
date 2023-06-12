# -*- coding: utf-8 -*-
import gzip
from dataclasses import dataclass
from pathlib import Path
from random import sample, seed
from subprocess import check_call

import numpy as np
import pytest
from numpy import typing as npt

from gwas.eig import Eigendecomposition
from gwas.mem.wkspace import SharedWorkspace
from gwas.pheno import VariableCollection
from gwas.rmw import Scorefile, ScorefileHeader
from gwas.tri import Triangular
from gwas.utils import Pool, chromosome_to_int, chromosomes_set
from gwas.vcf.base import VCFFile

from ..conftest import DirectoryFactory
from ..utils import gcta64, is_bfile, is_pfile, plink2, rmw, to_bgzip

chromosome: int | str = 22
other_chromosomes = sorted(chromosomes_set() - {chromosome, "X"}, key=chromosome_to_int)

effect_size: float = 2
minor_allele_frequency_cutoff: float = 0.05
causal_variant_count: int = int(3e4)
heritability: float = 0.6
simulation_count: int = 16
covariate_count: int = 4


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
                "--threads",
                str(1),
                "--memory",
                str(2**10),
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
    directory_factory: DirectoryFactory, sample_size: int, pfile_paths: list[Path]
):
    tmp_path = Path(directory_factory.get("variants", sample_size))

    (pfile_path,) = [p for p in pfile_paths if p.name.startswith(f"chr{chromosome}")]
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

    variants: list[str] = list()
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
            file_handle.write(f"{variant}\t{effect_size}\n")

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

    return SimulationResult(phen, par)


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
    directory_factory: DirectoryFactory,
    sample_size: int,
    tri_paths_by_chromosome: dict[str | int, Path],
    sw: SharedWorkspace,
    request,
) -> Eigendecomposition:
    tmp_path = Path(directory_factory.get("eig", sample_size))

    tris: list[Triangular] = [
        Triangular.from_file(tri_paths_by_chromosome[c], sw) for c in other_chromosomes
    ]

    eig = Eigendecomposition.from_tri(*tris)
    eig.to_file(tmp_path)

    request.addfinalizer(eig.free)

    return eig


@pytest.fixture(scope="module")
def rmw_scorefile_paths(
    directory_factory: DirectoryFactory,
    sample_size: int,
    vcf_files_by_chromosome: dict[int | str, VCFFile],
    vc: VariableCollection,
    eig: Eigendecomposition,
) -> list[Path]:
    rmw_path = Path(directory_factory.get("rmw", sample_size))

    vcf_path = vcf_files_by_chromosome[chromosome].file_path
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

    commands: list[list[str]] = list()
    scorefiles: list[Path] = list()
    for j in range(vc.phenotype_count):
        base = vc.phenotype_names[j]

        ped_path = rmw_path / f"{base}.ped"
        with ped_path.open("wt") as file_handle:
            for i, s in enumerate(vc.samples):
                c = format_row(covariates[i, 1:])  # skip intercept
                p = format_row(phenotypes[i, j])
                file_handle.write(f"{s} {s} 0 0 1 {p} {c}\n")

        dat_path = rmw_path / f"{base}.dat"
        with dat_path.open("wt") as file_handle:
            file_handle.write(f"T {base}\n")
            for name in vc.covariate_names:  # skip intercept
                file_handle.write(f"C {name}\n")

        prefix = str(rmw_path / f"chr{chromosome}")

        scorefile = f"{prefix}.{base}.singlevar.score.txt.gz"
        scorefiles.append(Path(scorefile))

        if Path(scorefile).is_file():
            continue

        commands.append(
            [
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
        )

    with Pool() as pool:
        pool.map(check_call, commands)

    for path in scorefiles:
        assert path.is_file()

    return scorefiles


@dataclass
class RmwScore:
    header: list[ScorefileHeader]
    array: npt.NDArray


@pytest.fixture(scope="module")
def rmw_score(
    rmw_scorefile_paths: list[Path],
) -> RmwScore:
    headers = list()
    arrays = list()
    for path in rmw_scorefile_paths:
        header, array = Scorefile.read(path)
        headers.append(header)
        arrays.append(array)

    return RmwScore(
        headers,
        np.vstack(arrays).transpose(),
    )
