from functools import partial
from typing import override

import jax
from chex import dataclass
from jax import numpy as jnp
from jaxtyping import Array, Float

from .mlb import OptimizeInput
from .mlb import terms_count as terms_count
from .pml import ProfileMaximumLikelihood


def logdet(a: Float[Array, " n n"]) -> Float[Array, ""]:
    """A re-implementation of torch.logdet that returns infinity instead of NaN, which
    prevents an error in autodiff.

    Args:
        a (Float[Array, "..."]): _description_

    Returns:
        _type_: _description_
    """
    sign, logabsdet = jnp.linalg.slogdet(a)
    logdet = jnp.where(sign == -1.0, jnp.inf, logabsdet)
    return logdet


@dataclass(frozen=True, eq=True)
class RestrictedMaximumLikelihood(ProfileMaximumLikelihood):
    @override
    @partial(jax.jit, static_argnums=0)
    def minus_two_log_likelihood(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> Float[Array, "..."]:
        t = self.get_minus_two_log_likelihood_terms(terms, o)
        penalty = logdet(t.r.scaled_covariates.transpose() @ t.r.scaled_covariates)
        deviation = (t.r.scaled_phenotype * t.r.scaled_residuals).sum()
        minus_two_log_likelihood = t.logarithmic_determinant + deviation + penalty

        if self.enable_softplus_penalty:
            minus_two_log_likelihood += self.softplus_penalty(terms, o)

        return jnp.where(
            jnp.isfinite(minus_two_log_likelihood),
            minus_two_log_likelihood,
            jnp.inf,
        )
