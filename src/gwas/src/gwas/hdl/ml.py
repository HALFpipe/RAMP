import jax
from jax import numpy as jnp
from jax.tree_util import Partial
from jaxtyping import Array, Float64, Integer

from .base import Reference, Sample1, Sample2
from .base import eig_count as eig_count
from .base import snp_count as snp_count
from .try_hard import try_hard


def scale_eigenvalues1(
    params: Float64[Array, " 2"],
    eigenvalues: Float64[Array, " ... eig_count"],
    snp_count_: Integer[Array, " ..."],
    sample_count: Integer[Array, ""],
    n_ref: Integer[Array, ""] | None = None,
    limit: Float64[Array, ""] | None = None,
) -> Float64[Array, " ... eig_count"]:
    if n_ref is None:
        n_ref = jnp.array(335265)
    if limit is None:
        limit = jnp.exp(-18.0)
    slope, intercept = params

    scaled_eigenvalues = (
        slope / snp_count_ * jnp.square(eigenvalues)
        - slope * eigenvalues / n_ref
        + intercept * eigenvalues / sample_count
    )
    scaled_eigenvalues = jnp.where(scaled_eigenvalues < limit, limit, scaled_eigenvalues)
    return scaled_eigenvalues


def minus_two_log_likelihood1(
    params: Float64[Array, " 2"],
    rotated_effects: Float64[Array, " eig_count"],
    sample_count: Integer[Array, ""],
    r: Reference,
    n_ref: Integer[Array, ""] | None = None,
    limit: Float64[Array, ""] | None = None,
) -> Float64[Array, ""]:
    scaled_eigenvalues = scale_eigenvalues1(
        params,
        r.eigenvalues,
        r.snp_count,
        sample_count,
        n_ref,
        limit,
    )
    return (
        jnp.log(scaled_eigenvalues).sum()
        + (jnp.square(rotated_effects) / scaled_eigenvalues).sum()
    )


def minus_two_log_likelihood2(
    params: Float64[Array, " 2"],
    scaled_eigenvalues1: Float64[Array, " eig_count"],
    scaled_eigenvalues2: Float64[Array, " eig_count"],
    rotated_effects1: Float64[Array, " eig_count"],
    rotated_effects2: Float64[Array, " eig_count"],
    r: Reference,
    sample_count: Float64[Array, ""],
    limit: Float64[Array, ""] | None = None,
) -> Float64[Array, ""]:
    if limit is None:
        limit = jnp.exp(-18.0)
    slope, intercept = params

    scaled_eigenvalues12 = (
        slope / r.snp_count * jnp.square(r.eigenvalues)
        + intercept * r.eigenvalues / sample_count
    )

    # resid of b_star2 ~ b_star1
    rotated_residuals = (
        rotated_effects2 - scaled_eigenvalues12 / scaled_eigenvalues1 * rotated_effects1
    )
    scaled_eigenvalues = (
        scaled_eigenvalues2 - jnp.square(scaled_eigenvalues12) / scaled_eigenvalues1
    )
    scaled_eigenvalues = jnp.where(scaled_eigenvalues < limit, limit, scaled_eigenvalues)

    return (
        jnp.log(scaled_eigenvalues).sum()
        + (jnp.square(rotated_residuals) / scaled_eigenvalues).sum()
    )


def add_intercept(
    ld_scores: Float64[Array, " snp_count"], snp_count_: Integer[Array, ""]
) -> Float64[Array, " snp_count 2"]:
    intercept = jnp.where(jnp.arange(ld_scores.size) < snp_count_, 1.0, 0.0)
    return jnp.column_stack((intercept, ld_scores))


def get_sqrt_weights(
    a11: Float64[Array, " snp_count"],
    regressors: Float64[Array, " snp_count 2"],
    snp_count_: Integer[Array, ""],
    sample_count: Integer[Array, ""],
):
    ld_scores = regressors[:, 1]
    x, _, _, _ = jnp.linalg.lstsq(regressors, a11)
    slope = x[1] * snp_count_
    sqrt_weights = slope * ld_scores / snp_count_ + 1 / sample_count
    return sqrt_weights


def get_initial_params1(
    a11: Float64[Array, " snp_count"],
    ld_scores: Float64[Array, " snp_count"],
    snp_count_: Integer[Array, ""],
    sample_count: Integer[Array, ""],
) -> Float64[Array, " 2"]:
    regressors = add_intercept(ld_scores, snp_count_)

    sqrt_weights = get_sqrt_weights(a11, regressors, snp_count_, sample_count)

    weighted_a11 = a11 / sqrt_weights
    weighted_regressors = regressors / sqrt_weights[:, jnp.newaxis]

    x, _, _, _ = jnp.linalg.lstsq(weighted_regressors, weighted_a11)
    slope = x[1] * snp_count_
    return jnp.array([slope, 1])


def get_initial_params2(
    a11: Float64[Array, " snp_count"],
    a22: Float64[Array, " snp_count"],
    a12: Float64[Array, " snp_count"],
    sample_proportion1: Float64[Array, ""],
    sample_proportion2: Float64[Array, ""],
    rho12: Float64[Array, ""],
    ld_scores: Float64[Array, " snp_count"],
    snp_count_: Integer[Array, ""],
    sample_count: Integer[Array, ""],
    limit: Float64[Array, ""] | None = None,
) -> Float64[Array, " 2"]:
    if limit is None:
        limit = jnp.exp(-18.0)
    regressors = add_intercept(ld_scores, snp_count_)
    sqrt_weights1 = get_sqrt_weights(a11, regressors, snp_count_, sample_count)
    sqrt_weights2 = get_sqrt_weights(a22, regressors, snp_count_, sample_count)

    x, _, _, _ = jnp.linalg.lstsq(regressors, a12)
    slope = x[1] * snp_count_
    weights = sqrt_weights1 * sqrt_weights2 + jnp.square(
        slope * ld_scores / snp_count_
        + sample_proportion1 * sample_proportion2 * rho12 / sample_count,
    )
    weights = jnp.where(weights < limit, limit, weights)
    sqrt_weights = jnp.sqrt(weights)

    weighted_a12 = a12 / sqrt_weights
    weighted_regressors = regressors / sqrt_weights[:, jnp.newaxis]

    x, _, _, _ = jnp.linalg.lstsq(weighted_regressors, weighted_a12)
    slope = x[1] * snp_count_
    return jnp.array([slope, rho12])


@jax.jit
def estimate1(
    s: Sample1,
    r: Reference,
    key: jax.Array,
    initial_params: Float64[Array, " 2"] | None = None,
    n_ref: Integer[Array, ""] | None = None,
    limit: Float64[Array, ""] | None = None,
) -> tuple[Float64[Array, ""], Float64[Array, " 2"]]:
    sample_count = jnp.maximum(s.median_sample_count, 1.0)  # Avoid division by zero

    if initial_params is None:
        initial_params = get_initial_params1(
            jnp.square(s.marginal_effects), r.ld_scores, r.snp_count, sample_count
        )

    minus_two_log_likelihood = Partial(
        minus_two_log_likelihood1,
        rotated_effects=s.rotated_effects,
        sample_count=sample_count,
        r=r,
        n_ref=n_ref,
        limit=limit,
    )
    return try_hard(
        minus_two_log_likelihood,
        initial_params,
        key,
        lower_bound=jnp.array([0, 0]),
        upper_bound=jnp.array([1, 10]),
    )


def get_sample_count2(
    s: Sample2,
) -> tuple[Float64[Array, ""], Float64[Array, ""], Integer[Array, ""]]:
    m = jnp.minimum(s.sample1.min_sample_count, s.sample2.min_sample_count)
    sample_proportion1 = m / s.sample1.median_sample_count
    sample_proportion2 = m / s.sample2.median_sample_count
    sample_count = m / sample_proportion1 / sample_proportion2
    return sample_proportion1, sample_proportion2, sample_count


@jax.jit
def estimate2(
    s: Sample2,
    ref: Reference,
    key: jax.Array,
    initial_params: Float64[Array, " 2"] | None = None,
    n_ref: Array | None = None,
    limit: Float64[Array, ""] | None = None,
) -> tuple[Float64[Array, ""], Float64[Array, " 2"]]:
    sample_proportion1, sample_proportion2, sample_count = get_sample_count2(s)

    if initial_params is None:
        marginal_effects = s.sample1.marginal_effects * s.sample2.marginal_effects
        initial_params = get_initial_params2(
            jnp.square(s.sample1.marginal_effects),
            jnp.square(s.sample2.marginal_effects),
            jnp.square(marginal_effects),
            sample_proportion1,
            sample_proportion2,
            s.correlation,
            ref.ld_scores,
            ref.snp_count,
            sample_count,
            limit=limit,
        )

    scale_eigenvalues = Partial(
        scale_eigenvalues1,
        eigenvalues=ref.eigenvalues,
        snp_count_=ref.snp_count,
        n_ref=n_ref,
        limit=limit,
    )
    scaled_eigenvalues1 = scale_eigenvalues(
        s.params1, sample_count=s.sample1.median_sample_count
    )
    scaled_eigenvalues2 = scale_eigenvalues(
        s.params2, sample_count=s.sample2.median_sample_count
    )

    minus_two_log_likelihood = Partial(
        minus_two_log_likelihood2,
        scaled_eigenvalues1=scaled_eigenvalues1,
        scaled_eigenvalues2=scaled_eigenvalues2,
        rotated_effects1=s.sample1.rotated_effects,
        rotated_effects2=s.sample2.rotated_effects,
        r=ref,
        sample_count=sample_count,
        limit=limit,
    )
    return try_hard(
        minus_two_log_likelihood,
        initial_params,
        key,
        lower_bound=jnp.array([-1, -10]),
        upper_bound=jnp.array([1, 10]),
    )
