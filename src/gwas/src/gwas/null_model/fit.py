from typing import TYPE_CHECKING

from ..eig.base import Eigendecomposition
from ..pheno import VariableCollection
from .fastlmm import FaSTLMM, PenalizedFaSTLMM
from .ml import MaximumLikelihood
from .mpl import MaximumPenalizedLikelihood
from .pml import ProfileMaximumLikelihood
from .reml import RestrictedMaximumLikelihood

if TYPE_CHECKING:
    from .base import NullModelCollection

funcs = {
    "fastlmm": FaSTLMM.fit,
    "pfastlmm": PenalizedFaSTLMM.fit,
    "pml": ProfileMaximumLikelihood.fit,
    "mpl": MaximumPenalizedLikelihood.fit,
    "reml": RestrictedMaximumLikelihood.fit,
    "ml": MaximumLikelihood.fit,
}


def fit(
    eig: Eigendecomposition,
    vc: VariableCollection,
    nm: "NullModelCollection",
    method: str,
    num_threads,
) -> None:
    func = funcs[method]
    func([eig], [vc], [nm], num_threads=num_threads)
