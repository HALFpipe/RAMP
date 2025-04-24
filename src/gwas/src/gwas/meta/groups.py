import pickle
from collections import defaultdict
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from typing import Mapping

from more_itertools import distribute
from tqdm.auto import tqdm

from ..utils.multiprocessing import make_pool_or_null_context
from .index import Index


def check_batch(
    index_sm_name: str, groups: list[Mapping[str, str | None]]
) -> tuple[int, dict[frozenset[str], set[frozenset[tuple[str, str | None]]]]]:
    index_sm = SharedMemory(name=index_sm_name)
    index: Index = pickle.loads(index_sm.buf)
    index_sm.close()

    groups_by_phenotypes: dict[
        frozenset[str], set[frozenset[tuple[str, str | None]]]
    ] = defaultdict(set)
    for group in groups:
        phenotypes = frozenset(index.get_phenotypes(**group))
        if len(phenotypes) < 5:
            continue
        groups_by_phenotypes[phenotypes].add(frozenset(group.items()))

    return len(groups), groups_by_phenotypes


def check_groups(
    index: Index,
    groups: list[Mapping[str, str | None]],
    num_threads: int = 1,
) -> dict[frozenset[str], set[frozenset[tuple[str, str | None]]]]:
    index_bytes = pickle.dumps(index)

    groups_by_phenotypes: dict[
        frozenset[str], set[frozenset[tuple[str, str | None]]]
    ] = defaultdict(set)

    # create a shared memory object to pass the task context to the worker processes
    # this is more efficient than pickling the task context for each phenotype, which
    # is the default behavior of the multiprocessing module
    sm: SharedMemory | None = None
    try:
        sm = SharedMemory(create=True, size=len(index_bytes))
        sm.buf[:] = index_bytes
        callable = partial(check_batch, sm.name)

        # create explicit batches to hide the overhead of unpickling the task context
        batches = list(map(list, distribute(num_threads * 4, groups)))

        pool, iterator = make_pool_or_null_context(batches, callable, num_threads)
        progress_bar = tqdm(total=len(groups), unit=" " + "groups", unit_scale=True)
        with pool, progress_bar:
            for k, batch_groups_by_phenotypes in iterator:
                for phenotypes, group_set in batch_groups_by_phenotypes.items():
                    groups_by_phenotypes[phenotypes].update(group_set)
                progress_bar.update(k)
    finally:
        if sm is not None:
            sm.close()
            sm.unlink()

    return groups_by_phenotypes
