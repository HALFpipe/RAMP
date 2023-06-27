# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import scipy
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import typing as npt

from gwas.eig import Eigendecomposition
from gwas.log import logger
from gwas.null_model.base import NullModelCollection
from gwas.pheno import VariableCollection
from gwas.score.calc import calc_u_stat, calc_v_stat
from gwas.vcf.base import VCFFile

from ...src.gwas.mem.arr import SharedArray
from ..utils import check_bias
from .conftest import RmwScore
from .rmw_debug import RmwDebug


def plot_stat(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    # Capitalize first letter of labels
    title = title[0].upper() + title[1:]
    xlabel = xlabel[0].upper() + xlabel[1:]
    ylabel = ylabel[0].upper() + ylabel[1:]

    (color,) = sns.color_palette("hls", 1)

    figure, axes = plt.subplots(figsize=(6, 6), dpi=600)
    axes.set_title(title)
    axes.scatter(
        x=x,
        y=y,
        color=color,
        alpha=0.005,
        edgecolors=None,  # type: ignore
    )
    axes.axline((0, 0), slope=1, color="black")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    figure.savefig(f"{path}.scatter.png")
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(6, 6), dpi=600)
    axes.set_title(title)
    sns.residplot(
        x=x,
        y=y,
        scatter_kws=dict(
            color=color,
            alpha=0.005,
            edgecolors=None,
        ),
        ax=axes,
    )
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    figure.savefig(f"{path}.residuals.png")
    plt.close(figure)


def test_rotate_demeaned_genotypes(
    genotype_array: SharedArray,
    eig: Eigendecomposition,
    rotated_genotypes_array: SharedArray,
) -> None:
    genotypes = genotype_array.to_numpy()
    rotated_genotypes = rotated_genotypes_array.to_numpy()

    mean = genotypes.sum(axis=0)

    rotated_with_mean = np.asfortranarray(eig.eigenvectors.transpose() @ genotypes)
    scipy.linalg.blas.dger(
        alpha=-1,
        x=eig.eigenvectors.sum(axis=0),
        y=mean,
        a=rotated_with_mean,
        overwrite_a=True,
        overwrite_y=False,
    )
    assert np.allclose(rotated_with_mean, rotated_genotypes)


def test_genotypes_array(
    vcf_file: VCFFile,
    genotype_array: SharedArray,
    vc: VariableCollection,
    rmw_score: RmwScore,
) -> None:
    r = rmw_score.array
    genotypes = genotype_array.to_numpy()

    sample_count, variant_count = genotypes.shape
    positions = np.asanyarray(vcf_file.variants.position)

    alternate_allele_count = genotypes.sum(axis=0)
    mean = alternate_allele_count / sample_count
    alternate_allele_frequency = mean / 2

    assert np.all(r["POS"] == positions[:, np.newaxis])
    assert np.all(r["N_INFORMATIVE"] == vc.sample_count)
    assert np.allclose(r["FOUNDER_AF"], alternate_allele_frequency[:, np.newaxis])
    assert np.allclose(r["ALL_AF"], alternate_allele_frequency[:, np.newaxis])
    assert np.allclose(
        r["INFORMATIVE_ALT_AC"],
        alternate_allele_count[:, np.newaxis],
    )


def test_score(
    tmp_path: Path,
    phenotype_index: int,
    nm: NullModelCollection,
    eig: Eigendecomposition,
    rotated_genotypes_array: SharedArray,
    rmw_score: RmwScore,
    rmw_debug: RmwDebug,
) -> None:
    rotated_genotypes = rotated_genotypes_array.to_numpy()

    phenotype_count = 1
    rotated_genotype = rotated_genotypes[:, 0]
    assert np.allclose(
        eig.eigenvectors.transpose() @ rmw_debug.genotype,
        rotated_genotype,
        atol=1e-6,
        rtol=1e-6,
    )

    sample_count, variant_count = rotated_genotypes.shape

    # Parse rmw scorefile columns
    rmw_u_stat = rmw_score.array["U_STAT"]
    rmw_sqrt_v_stat = rmw_score.array["SQRT_V_STAT"]
    rmw_effsize = rmw_score.array["ALT_EFFSIZE"]
    rmw_pvalue = rmw_score.array["PVALUE"]

    rmw_u_stat = rmw_u_stat[:, phenotype_index, np.newaxis]
    rmw_sqrt_v_stat = rmw_sqrt_v_stat[:, phenotype_index, np.newaxis]
    rmw_effsize = rmw_effsize[:, phenotype_index, np.newaxis]
    rmw_pvalue = rmw_pvalue[:, phenotype_index, np.newaxis]

    log_rmw_pvalue = np.log(rmw_pvalue)

    finite = np.isfinite(rmw_u_stat).all(axis=1)
    finite &= np.isfinite(rmw_sqrt_v_stat).all(axis=1)

    u_stat = np.empty((variant_count, phenotype_count))
    v_stat = np.empty((variant_count, phenotype_count))

    half_scaled_residuals = nm.half_scaled_residuals.to_numpy()
    half_scaled_residuals = half_scaled_residuals[:, phenotype_index, np.newaxis]
    variance = nm.variance.to_numpy()
    variance = variance[:, phenotype_index, np.newaxis]

    inverse_variance = np.reciprocal(variance)
    sqrt_inverse_variance = np.power(variance, -0.5)
    scaled_residuals = sqrt_inverse_variance * half_scaled_residuals

    calc_u_stat(
        scaled_residuals,
        rotated_genotypes,
        u_stat,
    )

    squared_genotypes = np.square(rotated_genotypes)
    invalid = calc_v_stat(
        inverse_variance,
        squared_genotypes,
        u_stat,
        v_stat,
    )

    sqrt_v_stat = np.sqrt(v_stat)
    effsize = u_stat / v_stat
    chi2 = np.square(u_stat) / v_stat
    log_pvalue = scipy.stats.chi2(1).logsf(chi2)

    assert np.all(np.isfinite(u_stat[finite, :]))
    assert np.all(np.isfinite(sqrt_v_stat[finite, :]))

    to_compare = finite[:, np.newaxis] & ~invalid

    is_ok = True

    has_no_bias = check_bias(u_stat, rmw_u_stat, to_compare)
    is_ok = is_ok and has_no_bias

    has_no_bias = check_bias(sqrt_v_stat, rmw_sqrt_v_stat, to_compare, tolerance=1e-1)
    is_ok = is_ok and has_no_bias

    same_maximum = np.isclose(
        nm.log_likelihood[phenotype_index],
        rmw_debug.log_likelihood_hat,
        atol=1e-3,
        rtol=1e-3,
    )
    if same_maximum:
        has_no_bias = check_bias(
            effsize,
            rmw_effsize,
            to_compare,
            tolerance=np.abs(rmw_effsize[to_compare]).mean() / 1e3,
            check_residuals=False,
        )
        is_ok = is_ok and has_no_bias

        has_no_bias = check_bias(log_pvalue, log_rmw_pvalue, to_compare)
        is_ok = is_ok and has_no_bias
    else:
        logger.info(
            f"log likelihood is {nm.log_likelihood[phenotype_index]} "
            f"(rmw is {rmw_debug.log_likelihood_hat})"
        )
    if not is_ok or not same_maximum:
        title = f"OpenSNP (n={sample_count})"
        for name, rmw_stat, stat in [
            ("U", u_stat, rmw_u_stat),
            ("sqrt(V)", sqrt_v_stat, rmw_sqrt_v_stat),
            ("effect size", effsize, rmw_effsize),
            ("log p-value", log_pvalue, log_rmw_pvalue),
        ]:
            plot_stat(
                rmw_stat[to_compare],
                stat[to_compare],
                title,
                f"RAREMETALWORKER {name}",
                name,
                tmp_path / f"{name}",
            )

    assert is_ok


# def test_calc_score(
#     tmp_path: Path,
#     vcf_by_chromosome: dict[int | str, VCFFile],
#     vc: VariableCollection,
#     nm: NullModelCollection,
#     sw: SharedWorkspace,
#     eig: Eigendecomposition,
#     rmw_score: RmwScore,
# ):
#     vcf_file = vcf_by_chromosome[chromosome]

#     score_path = tmp_path / "score.txt.zst"

#     calc_score(
#         vcf_file,
#         vc,
#         nm,
#         eig,
#         sw,
#         score_path,
#     )

#     header, array = CombinedScorefile.read(score_path)

#     def compare_summaries(a, b, compare_name=True):
#         if compare_name:
#             assert a.name == b.name
#         assert np.isclose(a.mean, b.mean)
#         assert np.isclose(a.variance, b.variance)
#         assert np.isclose(a.minimum, b.minimum)
#         assert np.isclose(a.maximum, b.maximum)

#     for rmw_header in rmw_score.header:
#         for rmw_summary, summary in zip(
#             rmw_header.covariate_summaries, header.covariate_summaries
#         ):
#             compare_summaries(rmw_summary, summary)

#     for i, rmw_header in enumerate(rmw_score.header):
#         summary = header.trait_summaries[i]
#         compare_summaries(rmw_header.trait_summaries[0], summary)
#         compare_summaries(rmw_header.analyzed_trait, summary, compare_name=False)

#         assert rmw_header.samples == header.samples
#         assert rmw_header.analyzed_samples == header.analyzed_samples
#         assert rmw_header.covariates == header.covariates

#     rmw_u_stat = rmw_score.array["U_STAT"]
#     u_stat = np.vstack(
#         [array[f"U_STAT[{phenotype_name}]"] for phenotype_name in vc.phenotype_names]
#     ).transpose()

#     rmw_v_stat = np.square(rmw_score.array["SQRT_V_STAT"])
#     v_stat = np.vstack(
#         [array[f"V_STAT[{phenotype_name}]"] for phenotype_name in vc.phenotype_names]
#     ).transpose()

#     finite = np.isfinite(rmw_u_stat).all(axis=1)
#     finite &= np.isfinite(rmw_v_stat).all(axis=1)
#     to_compare = finite[:, np.newaxis] & ~(u_stat == 0)

#     atol = 1e-2
#     if vc.sample_count < 2**10:
#         atol = 5e-2

#     check_slope = True
#     if nm.method in {"pml", "reml", "fastlmm"}:
#         check_slope = False

#     check_bias(u_stat, rmw_u_stat, to_compare, atol, check_slope)
#     check_bias(v_stat, rmw_v_stat, to_compare, atol, check_slope)
