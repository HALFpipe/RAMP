# -*- coding: utf-8 -*-
import gzip
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
from gwas.rmw import read_scorefile
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
heritability: float = 1
simulation_count: int = 16
covariate_count: int = 4


@pytest.fixture(scope="session")
def pfile_paths(tmpdir_factory, vcf_paths):
    tmp_path = Path(tmpdir_factory.mktemp("pfile"))

    pfiles: list[Path] = list()
    commands: list[list[str]] = list()
    for vcf_path in vcf_paths:
        pfile_path = tmp_path / vcf_path.name.split(".")[0]
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
        pfiles.append(pfile_path)

    with Pool() as pool:
        pool.map(check_call, commands)

    return pfiles


@pytest.fixture(scope="session")
def bfile_path(tmpdir_factory, pfile_paths):
    tmp_path = Path(tmpdir_factory.mktemp("bfile"))

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
def variants(tmpdir_factory, pfile_paths):
    tmp_path = Path(tmpdir_factory.mktemp("variants"))

    (pfile_path,) = [p for p in pfile_paths if p.name.startswith(f"chr{chromosome}")]
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
    covariates = np.random.normal(size=(len(samples), covariate_count))
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
            file_handle.write(f"{row}\n")

    covariates = vc.covariates.to_numpy()
    phenotypes = vc.phenotypes.to_numpy()

    commands: list[list[str]] = list()
    prefixes: list[str] = list()
    for j in range(vc.phenotype_count):
        base = f"phenotype_{j + 1:02d}"
        ped_path = tmp_path / f"{base}.ped"
        with ped_path.open("wt") as file_handle:
            for i, sample in enumerate(vc.samples):
                c = format_row(covariates[i, 1:])  # skip intercept
                p = format_row(phenotypes[i, j])
                file_handle.write(f"{sample} {sample} 0 0 1 {p} {c}\n")

        dat_path = tmp_path / f"{base}.dat"
        with dat_path.open("wt") as file_handle:
            file_handle.write("T x\n")
            for i in range(vc.covariate_count - 1):  # skip intercept
                file_handle.write(f"C covariate_{i + 1:02d}\n")

        prefix = str(tmp_path / f"chr{chromosome}.{base}")
        prefixes.append(prefix)
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

    return np.vstack(
        [read_scorefile(f"{prefix}.x.singlevar.score.txt.gz") for prefix in prefixes]
    ).transpose()


def test_score(
    vcf_by_chromosome: dict[int | str, VCFFile],
    vc: VariableCollection,
    eig: Eigendecomposition,
    rmw_score,
):
    # load genotypes
    sample_count = vc.sample_count
    assert eig.sample_count == sample_count
    vcf_file = vcf_by_chromosome[chromosome]
    sw = eig.sw
    variant_count = sw.unallocated_size // (
        np.float64().itemsize * 2 * (vc.phenotype_count + vc.sample_count)
    )
    variant_count = min(variant_count, vcf_file.variant_count)

    genotypes_array = sw.alloc("genotypes", sample_count, variant_count)
    genotypes = genotypes_array.to_numpy()
    with vcf_file:
        variants = vcf_file.read(genotypes.transpose())

    positions = np.array([v.position for v in variants], dtype=int)

    alternate_allele_count = genotypes.sum(axis=0)
    mean = alternate_allele_count / sample_count
    alternate_allele_frequency = mean / 2
    genotypes -= mean

    rotated_genotypes_array = sw.alloc(
        "rotated_genotypes",
        sample_count,
        variant_count,
    )

    variant_count = len(variants)
    assert variant_count == vcf_file.variant_count
    genotypes_array.resize(sample_count, variant_count)
    genotypes = genotypes_array.to_numpy()

    rotated_genotypes_array.resize(sample_count, variant_count)
    rotated_genotypes = rotated_genotypes_array.to_numpy()

    rotated_genotypes[:] = eig.eigenvectors.transpose() @ genotypes

    # parse rmw scorefile columns
    assert np.all(rmw_score["POS"] == positions[:, np.newaxis])
    assert np.all(rmw_score["N_INFORMATIVE"] == vc.sample_count)
    assert np.allclose(
        rmw_score["FOUNDER_AF"], alternate_allele_frequency[:, np.newaxis]
    )
    assert np.allclose(rmw_score["ALL_AF"], alternate_allele_frequency[:, np.newaxis])
    assert np.allclose(
        rmw_score["INFORMATIVE_ALT_AC"],
        alternate_allele_count[:, np.newaxis],
    )

    rmw_u_stat = rmw_score["U_STAT"]
    rmw_sqrt_v_stat = rmw_score["SQRT_V_STAT"]
    rmw_effsize = rmw_score["ALT_EFFSIZE"]

    finite = np.isfinite(rmw_u_stat).all(axis=1) & np.isfinite(rmw_sqrt_v_stat).all(
        axis=1
    )

    for method in ["ml", "pml", "fastlmm", "reml"]:
        vr = VarianceRatio.from_eig(eig, vc, method=method)

        regression_weights = vr.regression_weights.to_numpy()
        scaled_residuals = vr.scaled_residuals.to_numpy()
        variance = vr.variance.to_numpy()

        assert regression_weights is not None
        assert scaled_residuals is not None
        assert variance is not None

        u_stat = rotated_genotypes.transpose() @ scaled_residuals
        v_stat = np.square(rotated_genotypes).transpose() @ variance
        sqrt_v_stat = np.sqrt(v_stat)

        effsize = u_stat / v_stat

        assert np.square(u_stat[finite, :] - rmw_u_stat[finite, :]).mean() < 0.1
        assert (
            np.square(
                sqrt_v_stat[finite, :] - rmw_sqrt_v_stat[finite, :],
            ).mean()
            < 0.1
        )

        assert (
            np.corrcoef(u_stat[finite, :].ravel(), rmw_u_stat[finite, :].ravel())[1, 0]
            > 0.9
        )
        assert (
            np.corrcoef(
                sqrt_v_stat[finite, :].ravel(), rmw_sqrt_v_stat[finite, :].ravel()
            )[1, 0]
            > 0.9
        )
        assert (
            np.corrcoef(effsize[finite, :].ravel(), rmw_effsize[finite, :].ravel())[
                1, 0
            ]
            > 0.9
        )
