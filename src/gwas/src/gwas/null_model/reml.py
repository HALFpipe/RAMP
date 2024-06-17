from dataclasses import dataclass

from jax import numpy as jnp
from jaxtyping import Array, Float

from .pml import OptimizeInput, ProfileMaximumLikelihood, logdet
from .pml import terms_count as terms_count


@dataclass
class RestrictedMaximumLikelihood(ProfileMaximumLikelihood):
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
