# -*- coding: utf-8 -*-
import gzip
from collections import OrderedDict
from multiprocessing import Pool
from pathlib import Path
from random import choices
from subprocess import check_call

import numpy as np
import pytest
from numpy import typing as npt

from gwas.eig import Eigendecomposition
from gwas.mem.wkspace import SharedWorkspace
from gwas.pheno import VariableCollection
from gwas.tri import Triangular
from gwas.utils import chromosome_to_int, chromosomes_set
from gwas.var import VarianceRatio
from gwas.vcf import VCFFile

from .utils import gcta64, plink2, rmw, to_bgzip

chromosome: int | str = 22
other_chromosomes = sorted(chromosomes_set() - {chromosome, "X"}, key=chromosome_to_int)

effect_size: float = 2
minor_allele_frequency_cutoff: float = 0.05
causal_variant_count = 2**5
heritability: float = 0.8
simulation_count: int = 16


@pytest.fixture(scope="session")
def pfile_by_chromosome(tmpdir_factory, vcf_files):
    tmp_path = Path(tmpdir_factory.mktemp("pfile"))

    pfiles: dict[str | int, Path] = OrderedDict()
    commands: list[list[str]] = list()
    for vcf_file in vcf_files:
        pfile_path = tmp_path / f"chr{vcf_file.chromosome}"
        commands.append(
            [
                plink2,
                "--silent",
                "--vcf",
                str(vcf_file.file_path),
                "--threads",
                str(1),
                "--memory",
                str(2**10),
                "--out",
                str(pfile_path),
            ]
        )
        pfiles[vcf_file.chromosome] = pfile_path

    with Pool() as pool:
        pool.map(check_call, commands)

    return pfiles


@pytest.fixture(scope="session")
def bfile_path(tmpdir_factory, pfile_by_chromosome):
    tmp_path = Path(tmpdir_factory.mktemp("bfile"))

    pfile_paths = sorted(pfile_by_chromosome.values())

    pfile_list_path = tmp_path / "pfile-list.txt"
    with pfile_list_path.open("wt") as file_handle:
        file_handle.write("\n".join(map(str, pfile_paths)))

    bfile_path = tmp_path / "call"
    check_call(
        [
            plink2,
            "--silent",
            "--pmerge-list",
            str(pfile_list_path),
            "--make-bed",
            "--out",
            str(bfile_path),
        ]
    )

    return bfile_path


@pytest.fixture(scope="session")
def variants(tmpdir_factory, pfile_by_chromosome):
    tmp_path = Path(tmpdir_factory.mktemp("variants"))

    pfile_path = pfile_by_chromosome[chromosome]
    afreq_path = tmp_path / pfile_path.name

    check_call(
        [
            plink2,
            "--silent",
            "--pfile",
            str(pfile_path),
            "--freq",
            "--nonfounders",
            "--out",
            str(afreq_path),
        ]
    )

    variants: list[str] = list()
    with (afreq_path.parent / f"{afreq_path.name}.afreq").open("rt") as file_handle:
        for line in file_handle:
            if line.startswith("#"):
                continue
            _, variant, _, _, maf, _ = line.split()
            if (
                minor_allele_frequency_cutoff
                < float(maf)
                < (1 - minor_allele_frequency_cutoff)
            ):
                variants.append(variant)

    return variants


@pytest.fixture(scope="session")
def simulated_phenotypes(
    tmpdir_factory,
    bfile_path: Path,
    variants: list[str],
):
    """
    This needs a lot of memory, so we do this before we allocate the shared workspace
    """
    tmp_path = Path(tmpdir_factory.mktemp("variables"))

    variants = choices(variants, k=causal_variant_count)

    variant_list_path = tmp_path / "variant-list.txt"
    with variant_list_path.open("wt") as file_handle:
        for variant in variants:
            file_handle.write(f"{variant}\t{effect_size}\n")

    simulation_path = tmp_path / "simulation"

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
            "--out",
            str(simulation_path),
        ]
    )

    phenotype_path = simulation_path.parent / f"{simulation_path.name}.phen"
    phenotypes = np.loadtxt(phenotype_path, dtype=str)

    return phenotypes


@pytest.fixture(scope="session")
def sw(request) -> SharedWorkspace:
    sw = SharedWorkspace.create(size=16 * 2**30)
    request.addfinalizer(sw.close)
    request.addfinalizer(sw.unlink)
    return sw


@pytest.fixture(scope="session")
def vc(
    simulated_phenotypes: npt.NDArray[np.str_],
    sw: SharedWorkspace,
) -> VariableCollection:
    samples = list(simulated_phenotypes[:, 1])
    phenotypes = simulated_phenotypes[:, 2:].astype(float)
    covariates = np.random.normal(size=(len(samples), 16))
    covariates -= covariates.mean(axis=0)
    return VariableCollection.from_arrays(
        samples,
        covariates,
        phenotypes,
        sw,
    )


@pytest.fixture(scope="session")
def eig(tri_by_chromosome, sw) -> Eigendecomposition:
    tris: list[Triangular] = [
        Triangular.from_file(tri_by_chromosome[c], sw) for c in other_chromosomes
    ]
    return Eigendecomposition.from_tri(*tris)


@pytest.fixture(scope="session")
def rmw_score(
    tmpdir_factory,
    vcf_by_chromosome: dict[int | str, VCFFile],
    vc: VariableCollection,
    eig: Eigendecomposition,
):
    tmp_path = Path(tmpdir_factory.mktemp("rmw"))

    vcf_path = vcf_by_chromosome[chromosome].file_path
    vcf_gz_path = to_bgzip(tmp_path, vcf_path)

    def format_row(a) -> str:
        if a.size == 1:
            return np.format_float_scientific(a)
        else:
            return " ".join(map(np.format_float_scientific, a))

    kinship = (eig.eigenvectors * eig.eigenvalues) @ eig.eigenvectors.transpose()
    kinship_path = tmp_path / "Empirical.Kinship.gz"
    with gzip.open(kinship_path, "wt") as file_handle:
        file_handle.write(" ".join(vc.samples))
        file_handle.write("\n")

        for i, sample in enumerate(vc.samples):
            row = format_row(kinship[i, : i + 1])
            file_handle.write(f"{sample} {row}\n")

    covariates = vc.covariates.to_numpy()
    phenotypes = vc.phenotypes.to_numpy()

    commands: list[list[str]] = list()
    for j in range(vc.phenotype_count):
        base = f"phenotype_{j + 1:02d}"
        ped_path = tmp_path / f"{base}.ped"
        with ped_path.open("wt") as file_handle:
            for i, sample in enumerate(vc.samples):
                c = format_row(covariates[i, 1:])  # skip intercept
                p = format_row(phenotypes[i, j])
                file_handle.write(f"{sample} {sample} 0 0 1 {c} {p}\n")

        dat_path = tmp_path / f"{base}.dat"
        with dat_path.open("wt") as file_handle:
            for i in range(vc.covariate_count - 1):  # skip intercept
                file_handle.write(f"C covariate_{i + 1:02d}\n")
            file_handle.write("T x\n")

        rmw_path = tmp_path / f"chr{chromosome}.{base}"
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
                str(rmw_path),
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

    return tmp_path


def test_score(
    vcf_by_chromosome: dict[int | str, VCFFile],
    vc: VariableCollection,
    eig: Eigendecomposition,
    rmw_score,
):
    vr = VarianceRatio.from_eig(eig, vc)
    regression_weights = vr.regression_weights.to_numpy()
    rotated_residuals = vr.rotated_residuals.to_numpy()
    inv_covariance = vr.inv_covariance.to_numpy()

    assert regression_weights is not None
    assert rotated_residuals is not None
    assert inv_covariance is not None

    # sample_count = vc.sample_count
    # assert eig.sample_count == sample_count
    # phenotype_count = vc.phenotype_count

    # vcf_file = vcf_by_chromosome[chromosome]
    # variant_count = sw.unallocated_size // (
    #     np.float64().itemsize * 2 * (vc.phenotype_count + vc.sample_count)
    # )
    # variant_count = min(variant_count, vcf_file.variant_count)

    # genotypes_array = sw.alloc("genotypes", sample_count, variant_count)
    # genotypes = genotypes_array.to_numpy()
    # with vcf_file:
    #     variants = vcf_file.read(genotypes.transpose())

    # minor_allele_count = genotypes.sum(axis=0)
    # mean = minor_allele_count / sample_count
    # genotypes -= mean

    # rotated_genotypes_array = sw.alloc(
    #      "rotated_genotypes", sample_count, variant_count,
    # )
    # denominator_array = sw.alloc("denominator", phenotype_count, variant_count)
    # numerator_array = sw.alloc("numerator", phenotype_count, variant_count)

    # variant_count = len(variants)
    # assert variant_count == vcf_file.variant_count
    # genotypes_array.resize(sample_count, variant_count)
    # genotypes = genotypes_array.to_numpy()

    # rotated_genotypes_array.resize(sample_count, variant_count)
    # rotated_genotypes = rotated_genotypes_array.to_numpy()

    # rotated_genotypes[:] = eig.eigenvectors.transpose() @ genotypes

    assert False
