from random import seed
from typing import Type

import numpy as np
import pytest
import scipy
from jax import numpy as jnp
from pytest import FixtureRequest

from gwas.eig.base import Eigendecomposition
from gwas.log import logger
from gwas.mem.wkspace import SharedWorkspace
from gwas.null_model.base import NullModelCollection, NullModelResult
from gwas.null_model.ml import MaximumLikelihood
from gwas.null_model.mlb import OptimizeInput
from gwas.null_model.pml import ProfileMaximumLikelihood
from gwas.null_model.reml import RestrictedMaximumLikelihood
from gwas.pheno import VariableCollection


@pytest.mark.parametrize(
    "ml_class",
    [MaximumLikelihood, ProfileMaximumLikelihood, RestrictedMaximumLikelihood],
)
def test_var(ml_class: Type[ProfileMaximumLikelihood]) -> None:
    seed(0xABC)

    # simulate data
    sample_count: int = 2**10
    covariate_count: int = 4

    covariates = np.random.normal(size=(sample_count, covariate_count))
    covariate_weights = np.random.normal(size=(covariate_count, 1))

    mean = (covariates @ covariate_weights).ravel()

    eigenvalues = np.exp(np.arange(sample_count) / sample_count * 10)
    eigenvalues = eigenvalues / eigenvalues.sum() * eigenvalues.size
    eigenvalues[-1] -= eigenvalues.sum() - sample_count
    eigenvalues[0] -= eigenvalues.sum() - sample_count

    kinship = scipy.stats.random_correlation(eigs=eigenvalues).rvs()

    phenotype = scipy.stats.multivariate_normal(
        mean=mean,
        cov=8 * kinship,
        allow_singular=True,
    ).rvs(size=1)[:, np.newaxis]
    phenotype += scipy.stats.norm().rvs(size=(sample_count, 1))

    eigenvalues, eigenvectors = np.linalg.eigh(kinship)

    eigenvalues = jnp.asarray(eigenvalues)
    rotated_covariates = jnp.asarray(eigenvectors.transpose() @ covariates)
    rotated_phenotype = jnp.asarray(eigenvectors.transpose() @ phenotype)

    o: OptimizeInput = (eigenvalues, rotated_covariates, rotated_phenotype)

    ml = ml_class.create()
    optimize_result = ml.optimize(o)
    terms = optimize_result.x

    heritability = terms[1] / terms[:2].sum()
    logger.debug(heritability)
    assert heritability > 0


def test_collection(sw: SharedWorkspace, request: FixtureRequest) -> None:
    sample_count = 100
    samples = [f"sample_{i}" for i in range(sample_count)]
    eig = Eigendecomposition.empty(
        chromosome=22,
        samples=samples,
        variant_count=100000,
        sw=sw,
    )
    request.addfinalizer(eig.free)

    phenotype_count = 10
    covariate_count = 2
    phenotype_names = [f"phenotype_{i}" for i in range(phenotype_count)]
    covariate_names = [f"covariate_{i}" for i in range(covariate_count)]
    vc = VariableCollection.from_arrays(
        samples=samples,
        phenotype_names=phenotype_names,
        phenotypes=np.random.normal(size=(sample_count, phenotype_count)),
        covariate_names=covariate_names,
        covariates=np.random.normal(size=(sample_count, covariate_count)),
        sw=sw,
    )
    request.addfinalizer(vc.free)
    nmc = NullModelCollection.empty(eig=eig, vc=vc)
    request.addfinalizer(nmc.free)

    for phenotype_index in range(phenotype_count):
        per_covariate_values = np.full(
            shape=(covariate_count + 1,),
            fill_value=float(phenotype_index),
        )
        per_sample_values = np.full(
            shape=(sample_count,),
            fill_value=float(phenotype_index),
        )
        result = NullModelResult(
            log_likelihood=float(phenotype_index),
            heritability=float(phenotype_index),
            genetic_variance=float(phenotype_index),
            error_variance=float(phenotype_index),
            regression_weights=per_covariate_values,
            standard_errors=per_covariate_values,
            halfway_scaled_residuals=per_sample_values,
            variance=per_sample_values,
        )
        nmc.put(phenotype_index, result)

    a = np.arange(phenotype_count)
    np.testing.assert_array_almost_equal(nmc.log_likelihood, a)
    np.testing.assert_array_almost_equal(nmc.heritability, a)
    np.testing.assert_array_almost_equal(nmc.genetic_variance, a)
    np.testing.assert_array_almost_equal(nmc.error_variance, a)
    for i in range(covariate_count + 1):
        np.testing.assert_array_almost_equal(nmc.regression_weights[:, i], a)
        np.testing.assert_array_almost_equal(nmc.standard_errors[:, i], a)
    for i in range(sample_count):
        np.testing.assert_array_almost_equal(nmc.halfway_scaled_residuals[i, :], a)
        np.testing.assert_array_almost_equal(nmc.variance[i, :], a)


def test_apply():
    pass
