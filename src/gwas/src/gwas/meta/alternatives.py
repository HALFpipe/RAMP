import pickle
from dataclasses import dataclass
from functools import partial
from multiprocessing.shared_memory import SharedMemory

from IPython.lib.pretty import pretty
from more_itertools import distribute
from tqdm.auto import tqdm

from ..log import logger
from ..utils.multiprocessing import make_pool_or_null_context
from .index import Index


@dataclass
class TaskContext:
    index: "Index"
    key: str
    alternative: str

    def check(self, phenotype: str) -> str | None:
        alternative_tags = self.index.tags_by_phenotypes[phenotype].copy()
        for ignore_key in self.index.ignore_keys:
            if ignore_key not in alternative_tags:
                continue
            del alternative_tags[ignore_key]
        alternative_tags[self.key] = self.alternative
        if self.index.get_phenotypes(**alternative_tags):
            # The alternative tag value already exists
            return None
        return phenotype


def check_alternative(
    task_context_sm_name: str, phenotypes: set[str]
) -> tuple[int, set[str]]:
    count = len(phenotypes)

    task_context_sm = SharedMemory(name=task_context_sm_name)
    c: TaskContext = pickle.loads(task_context_sm.buf)
    task_context_sm.close()

    phenotypes = set(filter(None, map(c.check, phenotypes)))
    return count, phenotypes


def alternative(
    index: Index,
    key: str,
    value: str,
    alternative: str,
    num_threads: int = 1,
) -> None:
    logger.debug(f"Ignore keys are {pretty(index.ignore_keys)}")
    phenotypes = index.phenotypes_by_tags[key][value]
    c = TaskContext(index, key, alternative)
    task_context_bytes = pickle.dumps(c)

    # create a shared memory object to pass the task context to the worker processes
    # this is more efficient than pickling the task context for each phenotype, which
    # is the default behavior of the multiprocessing module
    sm: SharedMemory | None = None
    try:
        sm = SharedMemory(create=True, size=len(task_context_bytes))
        sm.buf[:] = task_context_bytes
        callable = partial(check_alternative, sm.name)

        # create explicit batches to hide the overhead of unpickling the task context
        batches = list(map(set, distribute(num_threads * 4, phenotypes)))

        pool, iterator = make_pool_or_null_context(batches, callable, num_threads)
        progress_bar = tqdm(
            total=len(phenotypes),
            unit=" " + "phenotypes",
            unit_scale=True,
            position=1,
            leave=False,
        )
        with pool, progress_bar:
            for count, phenotypes in iterator:
                progress_bar.update(count)
                index.phenotypes_by_tags[key][alternative].update(phenotypes)

        index.alternatives[key][value].add(alternative)
    finally:
        if sm is not None:
            sm.close()
            sm.unlink()
