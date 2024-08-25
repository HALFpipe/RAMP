from chex import dataclass
from jax import numpy as jnp
from jaxtyping import Array, Float

from .pml import OptimizeInput, ProfileMaximumLikelihood
from .pml import terms_count as terms_count


@dataclass(frozen=True, eq=True)
class MaximumPenalizedLikelihood(ProfileMaximumLikelihood):
    """
    - Chung, Y., Rabe-Hesketh, S., Dorie, V., Gelman, A., & Liu, J. (2013).
      A nondegenerate penalized likelihood estimator for variance parameters in
      multilevel models.
      Psychometrika, 78, 685-709.
    - Chung, Y., Rabe-Hesketh, S., & Choi, I. H. (2013).
      Avoiding zero between-study variance estimates in random-effects meta-analysis.
      Statistics in medicine, 32(23), 4071-4089.
    - Chung, Y., Rabe-Hesketh, S., Gelman, A., Liu, J., & Dorie, V. (2012).
      Avoiding boundary estimates in linear mixed models through weakly informative
      priors.
    """

    def minus_two_log_likelihood(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> Float[Array, "..."]:
        penalty = -2 * jnp.log(terms).sum()
        return super().minus_two_log_likelihood(terms, o) + penalty
