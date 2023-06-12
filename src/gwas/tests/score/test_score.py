# -*- coding: utf-8 -*-
# from pathlib import Path

import numpy as np
import pytest

# import scipy
from numpy import typing as npt

from gwas.eig import Eigendecomposition
from gwas.log import logger
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from gwas.null_model.base import NullModelCollection
from gwas.pheno import VariableCollection

# from gwas.score.calc import calc_u_stat, calc_v_stat
from gwas.vcf.base import VCFFile

from .conftest import RmwScore, SimulationResult, chromosome


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
        vcf_file.read(genotypes.transpose())

    positions = np.asanyarray(vcf_file.variants.position)

    alternate_allele_count = genotypes.sum(axis=0)
    mean = alternate_allele_count / sample_count
    alternate_allele_frequency = mean / 2
    genotypes -= mean

    variant_count = positions.size
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


# def test_score(
#     vc: VariableCollection,
#     nm: NullModelCollection,
#     rotated_genotypes: npt.NDArray,
#     rmw_score: RmwScore,
#     expected_effect: npt.NDArray,
# ):
#     r = rmw_score.array

#     variant_count = rotated_genotypes.shape[1]

#     # parse rmw scorefile columns
#     rmw_u_stat = r["U_STAT"]
#     rmw_sqrt_v_stat = r["SQRT_V_STAT"]
#     # rmw_effsize = r["ALT_EFFSIZE"]
#     rmw_pvalue = r["PVALUE"]
#     log_rmw_pvalue = np.log(rmw_pvalue)

#     finite = np.isfinite(rmw_u_stat).all(axis=1)
#     finite &= np.isfinite(rmw_sqrt_v_stat).all(axis=1)

#     u_stat = np.empty((variant_count, vc.phenotype_count))
#     v_stat = np.empty((variant_count, vc.phenotype_count))

#     calc_u_stat(
#         nm,
#         rotated_genotypes,
#         u_stat,
#     )

#     squared_genotypes = np.square(rotated_genotypes)
#     invalid = calc_v_stat(
#         nm,
#         squared_genotypes,
#         u_stat,
#         v_stat,
#     )

#     # regression_weights = nm.regression_weights.to_numpy()
#     sqrt_v_stat = np.sqrt(v_stat)
#     # effsize = u_stat / v_stat
#     chi2 = np.square(u_stat) / v_stat
#     log_pvalue = scipy.stats.chi2(1).logsf(chi2)

#     assert np.all(np.isfinite(u_stat[finite, :]))
#     assert np.all(np.isfinite(sqrt_v_stat[finite, :]))

#     to_compare = finite[:, np.newaxis] & ~invalid

#     atol = 1e-2
#     if vc.sample_count < 2**10:
#         atol = 5e-2

#     check_slope = True
#     if nm.method in {"pml", "reml", "fastlmm"}:
#         check_slope = False

#     check_bias(u_stat, rmw_u_stat, to_compare, atol, check_slope)
#     check_bias(sqrt_v_stat, rmw_sqrt_v_stat, to_compare, atol, check_slope)
#     # check_bias(effsize, rmw_effsize, to_compare, atol)
#     check_bias(log_pvalue, log_rmw_pvalue, to_compare, atol, check_slope)

#     if vc.sample_count > 100:
#         difference = (log_pvalue - log_rmw_pvalue)[finite, :]
#         # less than twenty percent have a difference greater 0.1
#         assert (np.abs(difference) < 1e-1).mean() > 0.8

#         expected_effect = np.broadcast_to(
#             expected_effect[:, np.newaxis], u_stat.shape
#         )
#         pr = scipy.stats.pearsonr(expected_effect[to_compare], u_stat[to_compare])
#         assert np.isclose(pr.pvalue, 0)


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
