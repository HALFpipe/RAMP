from dataclasses import dataclass
from functools import partial
from typing import override

import jax
from chex import register_dataclass_type_with_jax_tree_util
from jax import numpy as jnp
from jaxtyping import Array, Float

from .mlb import OptimizeInput
from .mlb import terms_count as terms_count
from .pml import ProfileMaximumLikelihood


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

    @override
    @partial(jax.jit, static_argnums=0)
    def minus_two_log_likelihood(
        self, terms: Float[Array, " terms_count"], o: OptimizeInput
    ) -> Float[Array, "..."]:
        base = super().minus_two_log_likelihood
        penalty = -2 * jnp.log(terms).sum()
        return base(terms, o) + penalty


register_dataclass_type_with_jax_tree_util(MaximumPenalizedLikelihood)
