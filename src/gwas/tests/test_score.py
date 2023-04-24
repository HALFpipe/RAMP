# -*- coding: utf-8 -*-
import gzip
from dataclasses import dataclass
from pathlib import Path
from random import sample, seed
from subprocess import check_call

import numpy as np
import pytest
import scipy
from numpy import typing as npt

from gwas.eig import Eigendecomposition
from gwas.log import logger
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from gwas.pheno import VariableCollection
from gwas.rmw import CombinedScorefile, Scorefile, ScorefileHeader
from gwas.score import calc_score, calc_u_stat, calc_v_stat
from gwas.tri import Triangular
from gwas.utils import Pool, chromosome_to_int, chromosomes_set
from gwas.var import NullModelCollection
from gwas.vcf import VCFFile

from .conftest import DirectoryFactory
from .utils import gcta64, is_bfile, is_pfile, plink2, rmw, to_bgzip

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
    vcf_by_chromosome: dict[int | str, VCFFile],
    vc: VariableCollection,
    eig: Eigendecomposition,
) -> list[Path]:
    rmw_path = Path(directory_factory.get("rmw", sample_size))

    vcf_path = vcf_by_chromosome[chromosome].file_path
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


def compare_scorefile_arrays(array, test_array):
    for record, test_record in zip(array, test_array):
        for a, b in zip(record, test_record):
            if np.issubdtype(type(a), np.floating):
                if np.isnan(a):
                    assert np.isnan(b)
                    continue
            assert a == b


def test_scorefile(
    tmp_path: Path,
    rmw_scorefile_paths: list[Path],
):
    scorefile = rmw_scorefile_paths[0]

    header, array = Scorefile.read(scorefile)

    test_scorefile = tmp_path / "test.score.txt"
    Scorefile.write(test_scorefile, header, array)

    test_header, test_array = Scorefile.read(test_scorefile)

    assert header == test_header
    compare_scorefile_arrays(array, test_array)


def test_combined_scorefile(
    tmp_path: Path,
    rmw_scorefile_paths: list[Path],
):
    header, array = CombinedScorefile.from_scorefiles(rmw_scorefile_paths)

    test_scorefile = tmp_path / "test.score.txt"
    CombinedScorefile.write(test_scorefile, header, array)

    test_prefix = tmp_path / "test"
    test_scorefiles = CombinedScorefile.to_scorefiles(test_scorefile, test_prefix)

    for scorefile, test_scorefile in zip(rmw_scorefile_paths, test_scorefiles):
        header, array = Scorefile.read(scorefile)
        test_header, test_array = Scorefile.read(test_scorefile)

        assert header == test_header
        compare_scorefile_arrays(array, test_array)


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


@pytest.fixture(scope="module")
def rotated_genotypes(
    vcf_by_chromosome: dict[int | str, VCFFile],
    vc: VariableCollection,
    sw: SharedWorkspace,
    eig: Eigendecomposition,
    rmw_score: RmwScore,
    request,
):
    r = rmw_score.array

    sample_count = vc.sample_count
    assert eig.sample_count == sample_count
    vcf_file = vcf_by_chromosome[chromosome]
    variant_count = sw.unallocated_size // (
        np.float64().itemsize * 2 * (vc.phenotype_count + vc.sample_count)
    )
    variant_count = min(variant_count, vcf_file.variant_count)

    name = SharedArray.get_name(sw, "genotypes")
    genotypes_array = sw.alloc(name, sample_count, variant_count)
    request.addfinalizer(genotypes_array.free)
    name = SharedArray.get_name(sw, "rotated-genotypes")
    rotated_genotypes_array = sw.alloc(name, sample_count, variant_count)
    request.addfinalizer(rotated_genotypes_array.free)

    genotypes = genotypes_array.to_numpy()
    with vcf_file:
        variants = vcf_file.read(genotypes.transpose())

    positions = np.array([v.position for v in variants], dtype=int)

    alternate_allele_count = genotypes.sum(axis=0)
    mean = alternate_allele_count / sample_count
    alternate_allele_frequency = mean / 2
    genotypes -= mean

    variant_count = len(variants)
    assert variant_count == vcf_file.variant_count
    genotypes_array.resize(sample_count, variant_count)
    genotypes = genotypes_array.to_numpy()

    rotated_genotypes_array.resize(sample_count, variant_count)
    rotated_genotypes = rotated_genotypes_array.to_numpy()

    rotated_genotypes[:] = eig.eigenvectors.transpose() @ genotypes

    assert np.all(r["POS"] == positions[:, np.newaxis])
    assert np.all(r["N_INFORMATIVE"] == vc.sample_count)
    assert np.allclose(r["FOUNDER_AF"], alternate_allele_frequency[:, np.newaxis])
    assert np.allclose(r["ALL_AF"], alternate_allele_frequency[:, np.newaxis])
    assert np.allclose(
        r["INFORMATIVE_ALT_AC"],
        alternate_allele_count[:, np.newaxis],
    )

    return rotated_genotypes


@pytest.fixture(scope="module")
def expected_effect(
    rmw_score: npt.NDArray,
    simulation: SimulationResult,
) -> npt.NDArray:
    causal_variants = set(simulation.par[:, 0])
    variant_tuples = rmw_score[["CHROM", "POS", "REF", "ALT"]][:, 0]
    expected_effect = np.array(
        [
            ":".join(map(str, variant_tuple)) in causal_variants
            for variant_tuple in variant_tuples
        ]
    )
    return expected_effect


@pytest.fixture(scope="module", params=["ml", "pml", "reml", "fastlmm"])
def nm(
    vc: VariableCollection,
    eig: Eigendecomposition,
    request,
) -> NullModelCollection:
    method = request.param
    nm = NullModelCollection.from_eig(eig, vc, method=method)
    request.addfinalizer(nm.free)

    return nm


def regress(a, b, indices):
    a = a[indices][:, np.newaxis]
    b = b[indices][:, np.newaxis]

    x = np.hstack([np.ones_like(a), a])
    (intercept, slope), (sum_residuals,), _, _ = np.linalg.lstsq(x, b, rcond=None)
    mean_residuals = sum_residuals / a.size

    yield from map(float, (intercept, slope, mean_residuals))


def check_bias(a, b, indices, atol, check_slope=True):
    intercept, slope, mean_residuals = regress(a, b, indices)

    logger.debug(
        f"intercept={intercept:f} "
        f"|slope - 1|={np.abs(1 - slope):f} "
        f"mean_residuals={mean_residuals:f}"
    )
    assert np.isclose(intercept, 0, atol=atol)
    if check_slope:
        assert np.isclose(slope, 1, atol=atol)
    assert np.isclose(mean_residuals, 0, atol=atol)


def test_score(
    vc: VariableCollection,
    nm: NullModelCollection,
    rotated_genotypes: npt.NDArray,
    rmw_score: RmwScore,
    expected_effect: npt.NDArray,
):
    r = rmw_score.array

    variant_count = rotated_genotypes.shape[1]

    # parse rmw scorefile columns
    rmw_u_stat = r["U_STAT"]
    rmw_sqrt_v_stat = r["SQRT_V_STAT"]
    # rmw_effsize = r["ALT_EFFSIZE"]
    rmw_pvalue = r["PVALUE"]
    log_rmw_pvalue = np.log(rmw_pvalue)

    finite = np.isfinite(rmw_u_stat).all(axis=1)
    finite &= np.isfinite(rmw_sqrt_v_stat).all(axis=1)

    u_stat = np.empty((variant_count, vc.phenotype_count))
    v_stat = np.empty((variant_count, vc.phenotype_count))

    calc_u_stat(
        nm,
        rotated_genotypes,
        u_stat,
    )

    squared_genotypes = np.square(rotated_genotypes)
    invalid = calc_v_stat(
        nm,
        squared_genotypes,
        u_stat,
        v_stat,
    )

    # regression_weights = nm.regression_weights.to_numpy()
    sqrt_v_stat = np.sqrt(v_stat)
    # effsize = u_stat / v_stat
    chi2 = np.square(u_stat) / v_stat
    log_pvalue = scipy.stats.chi2(1).logsf(chi2)

    assert np.all(np.isfinite(u_stat[finite, :]))
    assert np.all(np.isfinite(sqrt_v_stat[finite, :]))

    to_compare = finite[:, np.newaxis] & ~invalid

    atol = 1e-2
    if vc.sample_count < 2**10:
        atol = 5e-2

    check_slope = True
    if nm.method in {"pml", "reml", "fastlmm"}:
        check_slope = False

    check_bias(u_stat, rmw_u_stat, to_compare, atol, check_slope)
    check_bias(sqrt_v_stat, rmw_sqrt_v_stat, to_compare, atol, check_slope)
    # check_bias(effsize, rmw_effsize, to_compare, atol)
    check_bias(log_pvalue, log_rmw_pvalue, to_compare, atol, check_slope)

    if vc.sample_count > 100:
        difference = (log_pvalue - log_rmw_pvalue)[finite, :]
        # less than twenty percent have a difference greater 0.1
        assert (np.abs(difference) < 1e-1).mean() > 0.8

        expected_effect = np.broadcast_to(expected_effect[:, np.newaxis], u_stat.shape)
        pr = scipy.stats.pearsonr(expected_effect[to_compare], u_stat[to_compare])
        assert np.isclose(pr.pvalue, 0)


def test_calc_score(
    tmp_path: Path,
    vcf_by_chromosome: dict[int | str, VCFFile],
    vc: VariableCollection,
    nm: NullModelCollection,
    sw: SharedWorkspace,
    eig: Eigendecomposition,
    rmw_score: RmwScore,
):
    vcf_file = vcf_by_chromosome[chromosome]

    score_path = tmp_path / "score.txt.zst"

    calc_score(
        vcf_file,
        vc,
        nm,
        eig,
        sw,
        score_path,
    )

    header, array = CombinedScorefile.read(score_path)

    def compare_summaries(a, b, compare_name=True):
        if compare_name:
            assert a.name == b.name
        assert np.isclose(a.mean, b.mean)
        assert np.isclose(a.variance, b.variance)
        assert np.isclose(a.minimum, b.minimum)
        assert np.isclose(a.maximum, b.maximum)

    for rmw_header in rmw_score.header:
        for rmw_summary, summary in zip(
            rmw_header.covariate_summaries, header.covariate_summaries
        ):
            compare_summaries(rmw_summary, summary)

    for i, rmw_header in enumerate(rmw_score.header):
        summary = header.trait_summaries[i]
        compare_summaries(rmw_header.trait_summaries[0], summary)
        compare_summaries(rmw_header.analyzed_trait, summary, compare_name=False)

        assert rmw_header.samples == header.samples
        assert rmw_header.analyzed_samples == header.analyzed_samples
        assert rmw_header.covariates == header.covariates

    rmw_u_stat = rmw_score.array["U_STAT"]
    u_stat = np.vstack(
        [array[f"U_STAT[{phenotype_name}]"] for phenotype_name in vc.phenotype_names]
    ).transpose()

    rmw_v_stat = np.square(rmw_score.array["SQRT_V_STAT"])
    v_stat = np.vstack(
        [array[f"V_STAT[{phenotype_name}]"] for phenotype_name in vc.phenotype_names]
    ).transpose()

    finite = np.isfinite(rmw_u_stat).all(axis=1)
    finite &= np.isfinite(rmw_v_stat).all(axis=1)
    to_compare = finite[:, np.newaxis] & ~(u_stat == 0)

    atol = 1e-2
    if vc.sample_count < 2**10:
        atol = 5e-2

    check_slope = True
    if nm.method in {"pml", "reml", "fastlmm"}:
        check_slope = False

    check_bias(u_stat, rmw_u_stat, to_compare, atol, check_slope)
    check_bias(v_stat, rmw_v_stat, to_compare, atol, check_slope)
