from dataclasses import dataclass
from itertools import chain
from typing import Mapping, Sequence, Type

from more_itertools import chunked
from tqdm.auto import tqdm

from ..eig.base import Eigendecomposition
from ..log import logger
from ..pheno import VariableCollection
from ..utils.multiprocessing import IterationOrder, make_pool_or_null_context
from .base import NullModelCollection, NullModelResult


def calc_null_model_collections(
    eigendecompositions: list[Eigendecomposition],
    variable_collections: list[VariableCollection],
    method: str | None = "fastlmm",
    num_threads: int = 1,
) -> list[NullModelCollection]:
    null_model_collections = [
        NullModelCollection.empty(eig, vc, method)
        for eig, vc in zip(eigendecompositions, variable_collections, strict=True)
    ]
    if method is not None:
        fit(
            method,
            eigendecompositions,
            variable_collections,
            null_model_collections,
            num_threads=num_threads,
        )
    return null_model_collections


@dataclass
class OptimizeJob:
    indices: tuple[int, Sequence[int]]

    eig: Eigendecomposition
    vc: VariableCollection

    method: str | None


def apply(optimize_job: OptimizeJob) -> list[tuple[tuple[int, int], NullModelResult]]:
    from .mlb import setup_jax

    setup_jax()

    from .fastlmm import FaSTLMM
    from .ml import MaximumLikelihood
    from .mpl import MaximumPenalizedLikelihood
    from .pml import ProfileMaximumLikelihood
    from .reml import RestrictedMaximumLikelihood

    ml_classes: Mapping[str, Type[ProfileMaximumLikelihood]] = {
        "fastlmm": FaSTLMM,
        "pml": ProfileMaximumLikelihood,
        "mpl": MaximumPenalizedLikelihood,
        "reml": RestrictedMaximumLikelihood,
        "ml": MaximumLikelihood,
    }

    if optimize_job.method is None:
        optimize_job.method = "fastlmm"
    ml_class = ml_classes[optimize_job.method]

    (variable_collection_index, phenotype_indices) = optimize_job.indices
    eig = optimize_job.eig
    vc = optimize_job.vc
    ml = ml_class.create(vc.sample_count, vc.covariate_count)

    o: list[tuple[tuple[int, int], NullModelResult]] = list()
    for phenotype_index in phenotype_indices:
        indices = (variable_collection_index, phenotype_index)
        o.append((indices, ml.get_null_model_result(vc, phenotype_index, eig)))

    return o


def fit(
    method: str | None,
    eigendecompositions: list[Eigendecomposition],
    variable_collections: list[VariableCollection],
    null_model_collections: list[NullModelCollection],
    num_threads: int,
    **kwargs,
) -> None:
    phenotype_count = sum(vc.phenotype_count for vc in variable_collections)
    chunksize, remainder = divmod(phenotype_count, num_threads * 4)
    if remainder:
        chunksize += 1

    optimize_jobs = [
        OptimizeJob(
            (collection_index, phenotype_indices),
            eig,
            vc,
            method,
        )
        for collection_index, (eig, vc) in enumerate(
            zip(
                eigendecompositions,
                variable_collections,
                strict=True,
            )
        )
        for phenotype_indices in chunked(range(vc.phenotype_count), chunksize)
    ]
    logger.debug(
        f"Running {len(optimize_jobs)} optimize jobs "
        f"for {phenotype_count} phenotypes"
    )

    pool, iterator = make_pool_or_null_context(
        optimize_jobs,
        apply,
        num_threads=num_threads,
        iteration_order=IterationOrder.UNORDERED,
    )
    with pool:
        for indices, null_model_result in tqdm(
            chain.from_iterable(iterator),
            desc="fitting null models",
            unit="phenotypes",
            total=phenotype_count,
        ):
            collection_index, phenotype_index = indices
            nm = null_model_collections[collection_index]
            nm.put(phenotype_index, null_model_result)
