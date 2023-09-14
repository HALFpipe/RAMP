# -*- coding: utf-8 -*-

from typing import Type

import numpy as np
import pytest
import torch

from gwas.log import logger
from gwas.null_model.fastlmm import FaSTLMM
from gwas.null_model.ml import (
    MaximumLikelihood,
    MaximumPenalizedLikelihood,
    OptimizeInput,
    ProfileMaximumLikelihood,
    RestrictedMaximumLikelihood,
)

from ..utils import check_bias
from .rmw_debug import RmwDebug


def test_fastlmm(
    rmw_debug: RmwDebug,
) -> None:
    sample_count, covariate_count = rmw_debug.x.shape
    ml = FaSTLMM(sample_count, covariate_count, enable_softplus_penalty=False)

    optimize_input = OptimizeInput(
        torch.tensor(rmw_debug.d),
        torch.tensor(rmw_debug.trans_u_x),
        torch.tensor(rmw_debug.trans_u_y[:, np.newaxis]),
    )

    # Test for intermediate values
    for (
        variance_ratio,
        rmw_sigma,
        rmw_constant,
        rmw_log_likelihood,
        rmw_beta,
        rmw_factor,
    ) in zip(
        rmw_debug.delta,
        rmw_debug.sigma,
        rmw_debug.constant,
        rmw_debug.log_likelihood,
        rmw_debug.beta,
        rmw_debug.factor,
    ):
        terms = torch.tensor([variance_ratio, 1], dtype=torch.float64)

        r = ml.get_regression_weights(terms, optimize_input)

        constant = float(torch.log(r.variance).sum())
        assert np.isclose(constant, rmw_constant, atol=1e-3, rtol=1e-3)

        sigma = float(torch.square(r.scaled_residuals).mean())
        assert np.isclose(sigma, rmw_sigma, atol=1e-3)

        (weights, _, residuals, variance) = ml.get_standard_errors(
            terms, optimize_input
        )
        # Sanity check as these values should just be passed through
        assert np.allclose(weights, r.regression_weights, atol=1e-3)
        assert np.allclose(residuals, r.scaled_residuals, atol=1e-3)

        # Compare to raremetalworker
        assert np.allclose(weights.ravel(), rmw_beta, atol=1e-3)
        factor = (residuals * np.sqrt(variance)).ravel().numpy()
        assert check_bias(factor, rmw_factor)

        log_likelihood = -0.5 * ml.minus_two_log_likelihood(terms, optimize_input)
        assert np.isclose(rmw_log_likelihood, log_likelihood, atol=1e-3)

    # Test for final values
    terms = torch.tensor([rmw_debug.delta_hat, 1], dtype=torch.float64)
    (_, _, residuals, _) = ml.get_standard_errors(terms, optimize_input)
    genetic_variance = float(np.square(residuals).mean())
    terms *= genetic_variance
    (_, genetic_variance, error_variance) = ml.get_heritability(terms)

    assert np.isclose(genetic_variance, rmw_debug.sigma_g2_hat, atol=1e-3, rtol=1e-3)
    assert np.isclose(error_variance, rmw_debug.sigma_e2_hat, atol=1e-2)

    (weights, _, residuals, variance) = ml.get_standard_errors(terms, optimize_input)
    assert np.allclose(weights.ravel(), rmw_debug.beta_hat, atol=1e-3)
    assert np.allclose(
        (residuals * np.sqrt(variance)).ravel(),
        rmw_debug.residuals,
        atol=1e-3,
        rtol=1e-3,
    )
    assert np.allclose(variance.ravel(), rmw_debug.sigma2, atol=1e-3)


@pytest.mark.parametrize(
    "ml_class",
    [
        MaximumLikelihood,
        ProfileMaximumLikelihood,
        MaximumPenalizedLikelihood,
        RestrictedMaximumLikelihood,
        FaSTLMM,
    ],
)
def test_optimize(
    rmw_debug: RmwDebug,
    ml_class: Type[ProfileMaximumLikelihood],
) -> None:
    sample_count, covariate_count = rmw_debug.x.shape
    ml = ml_class(sample_count, covariate_count)

    optimize_input = OptimizeInput(
        torch.tensor(rmw_debug.d),
        torch.tensor(rmw_debug.trans_u_x),
        torch.tensor(rmw_debug.trans_u_y[:, np.newaxis]),
    )

    optimize_result = ml.optimize(optimize_input)
    terms = torch.Tensor(optimize_result.x)
    (heritability, genetic_variance, error_variance) = ml.get_heritability(terms)

    logger.info(
        f"genetic_variance is {genetic_variance} (rmw is {rmw_debug.sigma_g2_hat})"
    )
    logger.info(f"error_variance is {error_variance} (rmw is {rmw_debug.sigma_e2_hat})")

    rmw_heritability = rmw_debug.sigma_g2_hat / (
        rmw_debug.sigma_g2_hat + rmw_debug.sigma_e2_hat
    )
    logger.info(f"heritability is {heritability} (rmw is {rmw_heritability})")
    log_likelihood = -0.5 * ml.minus_two_log_likelihood(terms, optimize_input)
    logger.info(
        f"log likelihood is {log_likelihood} "
        f"(rmw is {rmw_debug.log_likelihood_hat})"
    )
    if ml_class == ProfileMaximumLikelihood:
        pass
    elif ml_class == FaSTLMM:
        same_maximum = np.isclose(
            log_likelihood, rmw_debug.log_likelihood_hat, atol=1e-3
        )
        better_maximum = log_likelihood > rmw_debug.log_likelihood_hat
        assert better_maximum or same_maximum
        if better_maximum:
            difference = log_likelihood - rmw_debug.log_likelihood_hat
            logger.info(f"Found better maximum by {difference}")
        if same_maximum:
            assert np.isclose(
                genetic_variance, rmw_debug.sigma_g2_hat, atol=1e-3, rtol=1e-3
            )
            assert np.isclose(
                error_variance, rmw_debug.sigma_e2_hat, atol=1e-3, rtol=1e-3
            )
            assert np.isclose(heritability, rmw_heritability, atol=1e-3)
