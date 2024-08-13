from itertools import chain

import numpy as np
import pytest
import scipy
import seaborn as sns
from gwas.eig.base import Eigendecomposition
from gwas.eig.collection import EigendecompositionCollection
from gwas.log import logger
from gwas.mem.arr import SharedArray
from gwas.mem.wkspace import SharedWorkspace
from gwas.null_model.base import NullModelCollection
from gwas.pheno import VariableCollection
from gwas.score.calc import calc_u_stat, calc_v_stat
from gwas.utils import get_global_lock, make_sample_boolean_vectors
from gwas.vcf.base import VCFFile
from matplotlib import pyplot as plt
from numpy import typing as npt
from upath import UPath

from ..utils import check_bias
from .conftest import RmwScore
from .rmw_debug import RmwDebug
from .simulation import simulation_count


@pytest.fixture(scope="module")
def rotated_genotypes_arrays(
    vcf_file: VCFFile,
    genotypes_array: SharedArray,
    eigendecompositions: list[Eigendecomposition],
    sw: SharedWorkspace,
    request: pytest.FixtureRequest,
) -> list[SharedArray]:
    allocation_names = set(sw.allocations.keys())

    genotypes = genotypes_array.to_numpy()
    _, variant_count = genotypes.shape

    sample_boolean_vectors = make_sample_boolean_vectors(
        vcf_file.samples, (eig.samples for eig in eigendecompositions)
    )

    rotated_genotypes_arrays: list[SharedArray] = list()
    for eig, sample_boolean_vector in zip(
        eigendecompositions, sample_boolean_vectors, strict=True
    ):
        sample_count = eig.sample_count
        with get_global_lock():
            name = SharedArray.get_name(sw, "rotated-genotypes")
            rotated_genotypes_array = sw.alloc(name, sample_count, variant_count)
        request.addfinalizer(rotated_genotypes_array.free)

        mean = genotypes.mean(axis=0, where=sample_boolean_vector[:, np.newaxis])
        demeaned_genotypes = genotypes[sample_boolean_vector, :] - mean

        rotated_genotypes = rotated_genotypes_array.to_numpy()
        rotated_genotypes[:] = eig.eigenvectors.transpose() @ demeaned_genotypes

        rotated_genotypes_arrays.append(rotated_genotypes_array)

    new_allocation_names = {r.name for r in rotated_genotypes_arrays}
    assert set(sw.allocations.keys()) <= (allocation_names | new_allocation_names)
    return rotated_genotypes_arrays


@pytest.mark.parametrize("chromosome", [22], indirect=True)
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
def test_rotate_demeaned_genotypes(
    vcf_file: VCFFile,
    genotypes_array: SharedArray,
    eigendecompositions: list[Eigendecomposition],
    rotated_genotypes_arrays: list[SharedArray],
) -> None:
    ec = EigendecompositionCollection.from_eigendecompositions(
        vcf_file,
        eigendecompositions,
        base_samples=vcf_file.samples,
    )

    genotypes = genotypes_array.to_numpy()
    for eigenvector_array, sample_boolean_vector, rotated_genotypes_array in zip(
        ec.eigenvector_arrays,
        ec.sample_boolean_vectors,
        rotated_genotypes_arrays,
        strict=True,
    ):
        eigenvectors = eigenvector_array.to_numpy()

        rotated_genotypes = rotated_genotypes_array.to_numpy()

        mean = genotypes.mean(axis=0, where=sample_boolean_vector[:, np.newaxis])

        rotated_with_mean = np.asfortranarray(eigenvectors.transpose() @ genotypes)
        scipy.linalg.blas.dger(
            alpha=-1,
            x=eigenvectors.sum(axis=0),
            y=mean,
            a=rotated_with_mean,
            overwrite_a=True,
            overwrite_y=False,
        )
        assert np.allclose(rotated_with_mean, rotated_genotypes)


@pytest.mark.parametrize("chromosome", [22], indirect=True)
@pytest.mark.parametrize("sample_size_label", ["small"], indirect=True)
def test_genotypes_array(
    vcf_file: VCFFile,
    genotypes_array: SharedArray,
    variable_collections: list[VariableCollection],
    rmw_score: RmwScore,
) -> None:
    rmw = rmw_score.array
    genotypes = genotypes_array.to_numpy()

    sample_count, _ = genotypes.shape
    positions = np.asanyarray(vcf_file.vcf_variants.position)

    assert np.all(rmw["POS"] == positions[:, np.newaxis])
    for phenotype_index, sample_count in enumerate(
        chain.from_iterable(
            [variable_collection.sample_count] * variable_collection.phenotype_count
            for variable_collection in variable_collections
        )
    ):
        assert (rmw["N_INFORMATIVE"][:, phenotype_index] == sample_count).all()

    vc_by_phenotype = {
        phenotype_name: variable_collection
        for variable_collection in variable_collections
        for phenotype_name in variable_collection.phenotype_names
    }
    phenotype_names = [
        phenotype_name
        for variable_collection in variable_collections
        for phenotype_name in variable_collection.phenotype_names
    ]
    base_samples = vcf_file.samples
    for i in range(simulation_count):
        phenotype_name = phenotype_names[i]
        vc = vc_by_phenotype[phenotype_name]
        sample_boolean_vector = np.fromiter(
            (sample in vc.samples for sample in base_samples), dtype=np.bool_
        )
        sample_count = sample_boolean_vector.sum()

        alternate_allele_count = genotypes.sum(
            axis=0, where=sample_boolean_vector[:, np.newaxis]
        )
        mean = alternate_allele_count / sample_count
        alternate_allele_frequency = mean / 2

        assert np.allclose(rmw["FOUNDER_AF"][:, i], alternate_allele_frequency)
        assert np.allclose(rmw["ALL_AF"][:, i], alternate_allele_frequency)
        assert np.allclose(
            rmw["INFORMATIVE_ALT_AC"][:, i],
            alternate_allele_count,
        )


def plot_stat(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    title: str,
    xlabel: str,
    ylabel: str,
    path: UPath,
) -> None:
    # Capitalize first letter of labels
    title = title[0].upper() + title[1:]
    xlabel = xlabel[0].upper() + xlabel[1:]
    ylabel = ylabel[0].upper() + ylabel[1:]

    colors = sns.color_palette("hls", n_colors=1)
    if not isinstance(colors, list):
        raise TypeError("Expected list of colors")
    (color,) = colors

    figure, axes = plt.subplots(figsize=(6, 6), dpi=600)
    axes.set_title(title)
    axes.scatter(
        x=x,
        y=y,
        color=color,
        alpha=0.005,
        edgecolors=None,
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


def test_score(
    tmp_path: UPath,
    phenotype_index: int,
    null_model_collections: list[NullModelCollection],
    eigendecompositions: list[Eigendecomposition],
    rotated_genotypes_arrays: list[SharedArray],
    rmw_score: RmwScore,
    rmw_debug: RmwDebug,
) -> None:
    variable_collection_index = 0
    inner_index = phenotype_index

    for null_model_collection in null_model_collections:
        if inner_index - null_model_collection.phenotype_count < 0:
            break
        inner_index -= null_model_collection.phenotype_count
        variable_collection_index += 1

    logger.debug(
        f"Phenotype index {phenotype_index} is in variable collection "
        f"{variable_collection_index} at position {inner_index}"
    )

    null_model_collection = null_model_collections[variable_collection_index]
    eigendecomposition = eigendecompositions[variable_collection_index]
    rotated_genotypes_array = rotated_genotypes_arrays[variable_collection_index]

    rotated_genotypes = rotated_genotypes_array.to_numpy()

    phenotype_count = 1
    rotated_genotype = rotated_genotypes[:, 0]
    assert np.allclose(
        eigendecomposition.eigenvectors.transpose() @ rmw_debug.genotype,
        rotated_genotype,
        atol=1e-6,
        rtol=1e-6,
    )

    sample_count, variant_count = rotated_genotypes.shape

    # Parse rmw scorefile columns
    rmw_array = rmw_score.array[:variant_count, :]
    rmw_u_stat = rmw_array["U_STAT"]
    rmw_sqrt_v_stat = rmw_array["SQRT_V_STAT"]
    rmw_effsize = rmw_array["ALT_EFFSIZE"]
    rmw_pvalue = rmw_array["PVALUE"]

    rmw_u_stat = rmw_u_stat[:, phenotype_index, np.newaxis]
    rmw_sqrt_v_stat = rmw_sqrt_v_stat[:, phenotype_index, np.newaxis]
    rmw_effsize = rmw_effsize[:, phenotype_index, np.newaxis]
    rmw_pvalue = rmw_pvalue[:, phenotype_index, np.newaxis]

    log_rmw_pvalue = np.log(rmw_pvalue)

    finite = np.isfinite(rmw_u_stat).all(axis=1)
    finite &= np.isfinite(rmw_sqrt_v_stat).all(axis=1)

    u_stat = np.empty((variant_count, phenotype_count))
    v_stat = np.empty((variant_count, phenotype_count))

    half_scaled_residuals = null_model_collection.half_scaled_residuals.to_numpy()
    half_scaled_residuals = half_scaled_residuals[:, inner_index, np.newaxis]
    variance = null_model_collection.variance.to_numpy()
    variance = variance[:, inner_index, np.newaxis]

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

    log_likelihood = null_model_collection.log_likelihood[inner_index]
    same_maximum = np.isclose(
        log_likelihood,
        rmw_debug.log_likelihood_hat,
        atol=1e-3,
        rtol=1e-3,
    )
    has_no_effsize_bias = check_bias(
        effsize,
        rmw_effsize,
        to_compare,
        tolerance=np.abs(rmw_effsize[to_compare]).mean() / 1e3,
        check_residuals=False,
    )
    has_no_log_pvalue_bias = check_bias(log_pvalue, log_rmw_pvalue, to_compare)
    if same_maximum:
        is_ok = is_ok and has_no_effsize_bias
        is_ok = is_ok and has_no_log_pvalue_bias
    else:
        logger.info(
            f"log likelihood is {log_likelihood} "
            f"(rmw is {rmw_debug.log_likelihood_hat})"
        )
    if not is_ok:
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
