from contextlib import nullcontext
from dataclasses import dataclass
from itertools import batched
from multiprocessing import current_process
from typing import ContextManager, Mapping, Sequence, Type

from tqdm.auto import tqdm
from upath import UPath

from ..eig.base import Eigendecomposition
from ..log import logger
from ..pheno import VariableCollection
from ..utils.multiprocessing import IterationOrder, make_pool_or_null_context
from .base import NullModelCollection


def calc_null_model_collections(
    eigendecompositions: list[Eigendecomposition],
    variable_collections: list[VariableCollection],
    method: str | None = "fastlmm",
    num_threads: int = 1,
    jax_trace_dir: UPath | None = None,
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
            jax_trace_dir=jax_trace_dir,
        )
    return null_model_collections


@dataclass
class OptimizeJob:
    phenotype_indices: Sequence[int]

    eig: Eigendecomposition
    vc: VariableCollection
    nm: NullModelCollection

    method: str | None
    jax_trace_dir: UPath | None


def apply(optimize_job: OptimizeJob) -> int:
    import jax
    from jax import numpy as jnp

    from .fastlmm import FaSTLMM
    from .ml import MaximumLikelihood
    from .mlb import OptimizeInput, setup_jax
    from .mpl import MaximumPenalizedLikelihood
    from .pml import ProfileMaximumLikelihood
    from .reml import RestrictedMaximumLikelihood

    setup_jax()

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

    eig = optimize_job.eig
    vc = optimize_job.vc
    ml = ml_class.create()

    covariates = vc.covariates.copy()
    phenotypes = vc.phenotypes[:, optimize_job.phenotype_indices]

    # Subtract column mean from covariates (except intercept)
    covariates[:, 1:] -= covariates[:, 1:].mean(axis=0)

    # Rotate covariates and phenotypes
    eigenvalues = jnp.asarray(eig.eigenvalues.copy())
    rotated_covariates = jnp.asarray(eig.eigenvectors.transpose() @ covariates)
    rotated_phenotypes = jnp.asarray(eig.eigenvectors.transpose() @ phenotypes)

    context: ContextManager = nullcontext()
    if optimize_job.jax_trace_dir is not None:
        name = current_process().name
        context = jax.profiler.trace(
            optimize_job.jax_trace_dir / name, create_perfetto_trace=True
        )

    with context:
        for i, phenotype_index in enumerate(optimize_job.phenotype_indices):
            rotated_phenotype = rotated_phenotypes[:, i, jnp.newaxis]
            o: OptimizeInput = (eigenvalues, rotated_covariates, rotated_phenotype)
            result = ml.get_null_model_result(o)
            optimize_job.nm.put(phenotype_index, result)

    return len(optimize_job.phenotype_indices)


def fit(
    method: str | None,
    eigendecompositions: list[Eigendecomposition],
    variable_collections: list[VariableCollection],
    null_model_collections: list[NullModelCollection],
    num_threads: int,
    jax_trace_dir: UPath | None = None,
) -> None:
    phenotype_count = sum(vc.phenotype_count for vc in variable_collections)
    chunksize, remainder = divmod(phenotype_count, num_threads * 4)
    if remainder:
        chunksize += 1

    optimize_jobs = [
        OptimizeJob(
            phenotype_indices,
            eig,
            vc,
            nm,
            method,
            jax_trace_dir,
        )
        for eig, vc, nm in zip(
            eigendecompositions,
            variable_collections,
            null_model_collections,
            strict=True,
        )
        for phenotype_indices in batched(range(vc.phenotype_count), chunksize)
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
    progress_bar = tqdm(
        desc="fitting null models",
        unit="phenotypes",
        total=phenotype_count,
    )
    with pool, progress_bar:
        for k in iterator:
            progress_bar.update(k)
