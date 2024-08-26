from typing import Type

import numpy as np
import pytest
from jax import numpy as jnp

from gwas.log import logger
from gwas.null_model.fastlmm import FaSTLMM
from gwas.null_model.ml import MaximumLikelihood
from gwas.null_model.mlb import OptimizeInput, setup_jax
from gwas.null_model.mpl import MaximumPenalizedLikelihood
from gwas.null_model.pml import ProfileMaximumLikelihood
from gwas.null_model.reml import RestrictedMaximumLikelihood

from ..utils import assert_both_close, check_bias, check_types
from .rmw_debug import RmwDebug


@pytest.fixture(autouse=True, scope="module")
def setup() -> None:
    setup_jax()


def test_fastlmm(
    rmw_debug: RmwDebug,
) -> None:
    ml = FaSTLMM.create(enable_softplus_penalty=False)

    optimize_input: OptimizeInput = (
        jnp.asarray(rmw_debug.d),
        jnp.asarray(rmw_debug.trans_u_x),
        jnp.asarray(rmw_debug.trans_u_y[:, np.newaxis]),
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
        strict=False,
    ):
        terms = jnp.asarray([variance_ratio, 1])

        r = check_types(ml.get_regression_weights)(terms, optimize_input)

        constant = float(jnp.log(r.variance).sum())
        assert np.isclose(constant, rmw_constant, atol=1e-3, rtol=1e-3)

        sigma = float(jnp.square(r.scaled_residuals).mean())
        assert np.isclose(sigma, rmw_sigma, atol=1e-3)

        se = check_types(ml.get_standard_errors)(terms, optimize_input)
        # Sanity check as these values should just be passed through
        assert np.allclose(se.regression_weights, r.regression_weights, atol=1e-3)
        assert np.allclose(se.scaled_residuals, r.scaled_residuals, atol=1e-3)

        # Compare to raremetalworker
        assert np.allclose(se.regression_weights.ravel(), rmw_beta, atol=1e-3)
        factor = np.asarray(se.scaled_residuals * np.sqrt(se.variance)).ravel()
        assert check_bias(factor, rmw_factor)

        log_likelihood = -0.5 * check_types(ml.minus_two_log_likelihood)(
            terms, optimize_input
        )

        assert np.isclose(rmw_log_likelihood, log_likelihood, atol=1e-3)

    # Test for final values
    terms = jnp.asarray([rmw_debug.delta_hat, 1])
    se = ml.get_standard_errors(terms, optimize_input)
    genetic_variance = float(np.square(se.scaled_residuals).mean())
    terms *= genetic_variance
    (_, genetic_variance, error_variance) = ml.get_heritability(terms)

    assert np.isclose(genetic_variance, rmw_debug.sigma_g2_hat, atol=1e-3, rtol=1e-3)
    assert np.isclose(error_variance, rmw_debug.sigma_e2_hat, atol=1e-2)

    se = ml.get_standard_errors(terms, optimize_input)
    assert np.allclose(se.regression_weights.ravel(), rmw_debug.beta_hat, atol=1e-3)
    assert np.allclose(
        (se.scaled_residuals * np.sqrt(se.variance)).ravel(),
        rmw_debug.residuals,
        atol=1e-3,
        rtol=1e-3,
    )
    assert np.allclose(se.variance.ravel(), rmw_debug.sigma2, atol=1e-3)


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
    ml = ml_class.create()

    optimize_input: OptimizeInput = (
        jnp.asarray(rmw_debug.d),
        jnp.asarray(rmw_debug.trans_u_x),
        jnp.asarray(rmw_debug.trans_u_y[:, np.newaxis]),
    )

    optimize_result = ml.optimize(optimize_input)
    terms = jnp.asarray(optimize_result.x)
    (heritability, genetic_variance, error_variance) = ml.get_heritability(terms)

    logger.info(
        f"genetic_variance is {genetic_variance} (rmw is {rmw_debug.sigma_g2_hat})"
    )
    logger.info(f"error_variance is {error_variance} (rmw is {rmw_debug.sigma_e2_hat})")

    rmw_heritability = rmw_debug.sigma_g2_hat / (
        rmw_debug.sigma_g2_hat + rmw_debug.sigma_e2_hat
    )
    logger.info(f"heritability is {heritability} (rmw is {rmw_heritability})")
    log_likelihood = -0.5 * check_types(ml.minus_two_log_likelihood)(
        terms, optimize_input
    )
    assert log_likelihood.dtype == np.float64

    logger.info(
        f"log likelihood is {log_likelihood} (rmw is {rmw_debug.log_likelihood_hat})"
    )
    if ml_class == ProfileMaximumLikelihood:
        pass
    elif ml_class == FaSTLMM:
        same_maximum = np.isclose(
            log_likelihood, rmw_debug.log_likelihood_hat, atol=1e-4
        )
        better_maximum = log_likelihood > rmw_debug.log_likelihood_hat
        assert better_maximum or same_maximum
        if better_maximum:
            difference = log_likelihood - rmw_debug.log_likelihood_hat
            logger.info(f"Found better maximum by {difference}")
        if same_maximum:
            assert_both_close(
                genetic_variance,
                rmw_debug.sigma_g2_hat,
                error_variance,
                rmw_debug.sigma_e2_hat,
                atol=1e-3,
                rtol=1e-3,
            )
            assert np.isclose(heritability, rmw_heritability, atol=1e-3)
