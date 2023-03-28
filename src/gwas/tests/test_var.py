# -*- coding: utf-8 -*-
from random import seed

import numpy as np
import pytest
import scipy
import torch

from gwas.ml import (
    MaximumLikelihood,
    ProfileMaximumLikelihood,
    RestrictedMaximumLikelihood,
)


@pytest.mark.parametrize(
    "driver", [MaximumLikelihood, ProfileMaximumLikelihood, RestrictedMaximumLikelihood]
)
def test_var(driver):
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

    eigenvalues = torch.tensor(eigenvalues)
    rotated_covariates = torch.tensor(eigenvectors.transpose() @ covariates)
    rotated_phenotype = torch.tensor(eigenvectors.transpose() @ phenotype)

    ml = driver(
        sample_count,
        eigenvalues,
        rotated_covariates,
        rotated_phenotype,
    )
    optimize_result = ml.optimize()
    terms = optimize_result.x

    heritability = terms[1] / terms[:2].sum()
    print(heritability)
    assert heritability > 0
