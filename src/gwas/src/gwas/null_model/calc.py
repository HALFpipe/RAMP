from multiprocessing import cpu_count

from ..eig.base import Eigendecomposition
from ..pheno import VariableCollection
from .base import NullModelCollection
from .fastlmm import FaSTLMM, PenalizedFaSTLMM
from .ml import MaximumLikelihood
from .mpl import MaximumPenalizedLikelihood
from .pml import ProfileMaximumLikelihood
from .reml import RestrictedMaximumLikelihood

funcs = {
    "fastlmm": FaSTLMM.fit,
    "pfastlmm": PenalizedFaSTLMM.fit,
    "pml": ProfileMaximumLikelihood.fit,
    "mpl": MaximumPenalizedLikelihood.fit,
    "reml": RestrictedMaximumLikelihood.fit,
    "ml": MaximumLikelihood.fit,
}


def calc_null_model_collections(
    eigendecompositions: list[Eigendecomposition],
    variable_collections: list[VariableCollection],
    method: str | None = "fastlmm",
    num_threads: int = cpu_count(),
) -> list[NullModelCollection]:
    null_model_collections = [
        NullModelCollection.empty(eig, vc, method)
        for eig, vc in zip(eigendecompositions, variable_collections, strict=True)
    ]
    if method is not None:
        funcs[method](
            eigendecompositions,
            variable_collections,
            null_model_collections,
            num_threads=num_threads,
        )
    return null_model_collections
