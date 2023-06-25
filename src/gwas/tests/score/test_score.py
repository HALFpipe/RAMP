# -*- coding: utf-8 -*-
from typing import Mapping

import numpy as np
import pytest
import scipy
from numpy import typing as npt

from gwas.eig import Eigendecomposition
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from gwas.null_model.base import NullModelCollection
from gwas.pheno import VariableCollection
from gwas.score.calc import calc_u_stat, calc_v_stat
from gwas.vcf.base import VCFFile

from ..utils import check_bias
from .conftest import RmwScore
from .rmw_debug import RmwDebug


@pytest.fixture(scope="module")
def rotated_genotypes(
    chromosome: int | str,
    vcf_files_by_chromosome: Mapping[int | str, VCFFile],
    vc: VariableCollection,
    sw: SharedWorkspace,
    eig: Eigendecomposition,
    rmw_score: RmwScore,
    request,
):
    r = rmw_score.array

    sample_count = vc.sample_count
    assert eig.sample_count == sample_count
    vcf_file = vcf_files_by_chromosome[chromosome]
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
def nm(
    vc: VariableCollection,
    eig: Eigendecomposition,
    request,
) -> NullModelCollection:
    nm = NullModelCollection.from_eig(eig, vc, method="fastlmm")
    request.addfinalizer(nm.free)

    return nm


def test_score(
    phenotype_index: int,
    vc: VariableCollection,
    nm: NullModelCollection,
    eig: Eigendecomposition,
    rotated_genotypes: npt.NDArray,
    rmw_score: RmwScore,
    rmw_debug: RmwDebug,
):
    phenotype_count = 1
    rotated_genotype = rotated_genotypes[:, 0]
    assert np.allclose(
        eig.eigenvectors.transpose() @ rmw_debug.genotype,
        rotated_genotype,
    )

    variant_count = rotated_genotypes.shape[1]

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

    assert check_bias(u_stat, rmw_u_stat, to_compare)
    assert check_bias(sqrt_v_stat, rmw_sqrt_v_stat, to_compare)
    assert check_bias(effsize, rmw_effsize, to_compare, check_residuals=False)
    assert check_bias(log_pvalue, log_rmw_pvalue, to_compare)


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
