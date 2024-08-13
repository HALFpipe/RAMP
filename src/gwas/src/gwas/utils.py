import faulthandler
import multiprocessing as mp
import os
import re
import signal
from contextlib import nullcontext
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, auto
from logging import LogRecord
from multiprocessing import pool as mp_pool
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event, RLock
from pprint import pformat
from queue import Empty
from shutil import which
from threading import Thread
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    NamedTuple,
    Sequence,
    Sized,
    Type,
    TypeVar,
    get_args,
    get_origin,
    override,
)

import numpy as np
import pandas as pd
from numpy import typing as npt
from threadpoolctl import threadpool_info, threadpool_limits
from upath import UPath

from .log import logger, multiprocessing_context, worker_configurer

try:
    from pytest_cov.embed import cleanup_on_sigterm
except ImportError:
    pass
else:
    cleanup_on_sigterm()

global_lock: RLock = multiprocessing_context.RLock()


def parse_chromosome(chromosome: str) -> int | str:
    if chromosome == "X":
        return chromosome
    elif chromosome.isdigit():
        return int(chromosome)
    else:
        raise ValueError(f'Unknown chromosome "{chromosome}"')


def chromosome_to_int(chromosome: int | str) -> int:
    if chromosome == "X":
        return 23
    elif isinstance(chromosome, str) and chromosome.isdigit():
        return int(chromosome)
    elif isinstance(chromosome, int):
        return chromosome
    raise ValueError(f'Unknown chromsome "{chromosome}"')


def chromosome_from_int(chromosome_int: int) -> int | str:
    if chromosome_int == 23:
        return "X"
    else:
        return chromosome_int


def chromosomes_list() -> list[int | str]:
    return [*range(1, 22 + 1), "X"]


def chromosomes_set() -> set[int | str]:
    return set(chromosomes_list())


def unwrap_which(command: str) -> str:
    executable = which(command)
    if executable is None:
        raise ValueError(f"Could not find executable for {command}")
    return executable


def scale_rows(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return (a.transpose() * b.ravel()).transpose()


def to_str(x: Any) -> str:
    if isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.item()
    if np.issubdtype(type(x), np.floating):
        if np.isnan(x):
            return "NA"
        return np.format_float_scientific(x)
    return str(x)


def underscore(x: str) -> str:
    return re.sub(r"([a-z\d])([A-Z])", r"\1_\2", x).lower()


class InitArgs(NamedTuple):
    sched_affinity: set[int] | None
    num_threads: int | None
    logging_queue: Queue[LogRecord] | None
    log_level: int | str
    lock: RLock


def get_initargs(num_threads: int | None = None) -> InitArgs:
    from .log import logging_queue

    sched_affinity: set[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        sched_affinity = os.sched_getaffinity(0)

    initargs = InitArgs(
        sched_affinity,
        num_threads,
        logging_queue,
        logger.getEffectiveLevel(),
        global_lock,
    )
    logger.debug(f"Initializer arguments for child process are: {pformat(initargs)}")
    return initargs


num_threads_variables: Sequence[str] = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
]


def apply_num_threads(num_threads: int | None) -> None:
    faulthandler.register(signal.SIGUSR1)
    # Write a traceback to standard out every six hours
    faulthandler.dump_traceback_later(60 * 60 * 6, repeat=True)

    xla_flags = f'{os.getenv("XLA_FLAGS", "")} --xla_cpu_enable_fast_math=false'
    if num_threads is not None:
        threadpool_limits(limits=num_threads)
        for variable in num_threads_variables:
            os.environ[variable] = str(num_threads)
        xla_flags = (
            f"--xla_cpu_multi_thread_eigen={str(num_threads > 1).lower()} "
            f"intra_op_parallelism_threads={num_threads} "
            f"inter_op_parallelism_threads={num_threads} "
            f"{xla_flags}"
        )
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["XLA_FLAGS"] = xla_flags

    from chex import set_n_cpu_devices

    set_n_cpu_devices(1)

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_traceback_filtering", "off")


def initializer(
    sched_affinity: set[int] | None,
    num_threads: int | None,
    logging_queue: Queue[LogRecord] | None,
    log_level: int | str,
    lock: RLock,
) -> None:
    if sched_affinity is not None:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, sched_affinity)

    apply_num_threads(num_threads)

    if logging_queue is None:
        raise ValueError("Trying to initialize process without logging queue")

    worker_configurer(logging_queue, log_level)
    logger.debug(
        f'Configured process "{mp.current_process().name}" '
        f"with {pformat(threadpool_info())}"
    )

    global global_lock
    global_lock = lock


class Pool(mp_pool.Pool):
    _pool: list[mp.Process]

    def __init__(
        self,
        processes: int | None = None,
        maxtasksperchild: int | None = None,
        num_threads: int = 1,
    ) -> None:
        initargs = get_initargs(num_threads=num_threads)
        super().__init__(
            processes, initializer, initargs, maxtasksperchild, multiprocessing_context
        )

    @override
    def terminate(self) -> None:
        thread = Thread(target=super().terminate())
        thread.start()
        thread.join(timeout=10)
        if thread.is_alive():
            logger.debug("Pool did not terminate in time")
            for p in self._pool:
                soft_kill(p)


class Action(Enum):
    EVENT = auto()
    EXIT = auto()


V = TypeVar("V")


@dataclass
class SharedState:
    # Indicates that we should exit
    should_exit: Event = field(default_factory=multiprocessing_context.Event)
    # Passes exceptions
    exception_queue: Queue[Exception] = field(
        default_factory=multiprocessing_context.Queue
    )

    def get_name(self, value: Any) -> str:
        for dataclass_field in fields(self):
            field_value = getattr(self, dataclass_field.name)
            if value is field_value:
                return dataclass_field.name
            elif isinstance(field_value, Sequence) and value in field_value:
                index = field_value.index(value)
                return f"{dataclass_field.name}[{index}]"
        return repr(value)

    def get(self, queue: Queue[V]) -> V | Action:
        """Get an item from a queue while checking for exit.

        Args:
            queue (Queue[T]): The queue to get an item from.

        Returns:
            T | Action: The item from the queue or Action.EXIT if we should exit.
        """
        logger.debug(f'Waiting for queue "{self.get_name(queue)}"')
        while True:
            if self.should_exit.is_set():
                return Action.EXIT
            try:
                return queue.get(timeout=1)
            except Empty:
                pass

    def wait(self, event: Event) -> Action:
        logger.debug(f'Waiting for event "{self.get_name(event)}"')
        while True:
            if self.should_exit.is_set():
                return Action.EXIT
            if event.wait(timeout=1):
                return Action.EVENT


class Process(multiprocessing_context.Process):  # type: ignore
    def __init__(
        self,
        exception_queue: "mp.Queue[Exception] | None",
        num_threads: int | None,
        name: str | None = None,
    ) -> None:
        self.initargs: InitArgs = get_initargs(num_threads)
        self.exception_queue = exception_queue
        super().__init__(name=name)

    def set_num_threads(self, num_threads: int) -> None:
        self.initargs = get_initargs(num_threads=num_threads)

    def func(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        initializer(*self.initargs)
        logger.debug(f'Starting process "{self.name}"')

        try:
            self.func()
        except Exception as e:
            logger.exception(f'An error occurred in process "{self.name}"', exc_info=e)
            if self.exception_queue is not None:
                self.exception_queue.put_nowait(e)


def wait(running: list[Process]) -> None:
    for proc in running:
        proc.join(timeout=1)
        if proc.is_alive():
            break


def soft_kill(proc: mp.Process) -> None:
    try:
        proc.join(timeout=1)
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
        proc.join()
    except (ValueError, AttributeError, AssertionError) as e:
        logger.debug(f"Could not kill process {proc}", exc_info=e)
        pass


def soft_close(proc: Process) -> None:
    soft_kill(proc)
    try:
        proc.close()
    except ValueError:
        pass


def invert_pivot(pivot: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
    """Calculates the inverse of a pivot array. Taken from
    https://stackoverflow.com/a/25535723

    Args:
        pivot (npt.NDArray[np.integer]): The pivot array.

    Returns:
        npt.NDArray[np.integer]: The inverse.
    """
    inverse = np.empty_like(pivot)
    inverse[pivot] = np.arange(pivot.size)
    return inverse


T = TypeVar("T")


def parse_obj_as(cls: Type[T], data: Any) -> T:
    """Parses an object as the specified type. Inspired by the Pydantic function of the
    same name.

    Args:
        cls (Type[T]): The type to parse as.
        data (Any): The data to parse.

    Returns:
        T: The parsed object.
    """
    if is_dataclass(cls):
        return cls(
            **{f.name: parse_obj_as(f.type, data.get(f.name)) for f in fields(cls)}
        )  # type: ignore

    origin = get_origin(cls)
    if origin is list:
        (value_cls,) = get_args(cls)
        return [parse_obj_as(value_cls, element) for element in data]  # type: ignore
    elif origin is dict:
        (key_cls, value_cls) = get_args(cls)
        return {
            parse_obj_as(key_cls, key): parse_obj_as(value_cls, value)
            for key, value in data.items()
        }  # type: ignore

    return data  # type: ignore


def make_sample_boolean_vectors(
    base_samples: list[str],
    samples_iterable: Iterable[list[str]],
) -> list[npt.NDArray[np.bool_]]:
    return [
        np.fromiter((sample in samples for sample in base_samples), dtype=np.bool_)
        for samples in samples_iterable
    ]


class IterationOrder(Enum):
    ORDERED = auto()
    UNORDERED = auto()


S = TypeVar("S")


def make_pool_or_null_context(
    iterable: Iterable[T],
    callable: Callable[[T], S],
    num_threads: int = 1,
    size: int | None = None,
    chunksize: int | None = 1,
    iteration_order: IterationOrder = IterationOrder.UNORDERED,
) -> tuple[ContextManager[Any], Iterator[S]]:
    if num_threads < 2:
        return nullcontext(), map(callable, iterable)

    processes = max(1, num_threads)
    if isinstance(iterable, Sized):
        size = len(iterable)
        if size == 0:
            return nullcontext(), iter([])
    if size is not None:
        processes = min(processes, size)
        # Apply logic from pool.map (multiprocessing/pool.py#L481) here as well
        if chunksize is None:
            chunksize, extra = divmod(size, processes * 4)
            if extra:
                chunksize += 1
    if chunksize is None:
        chunksize = 1

    pool = Pool(processes=processes, num_threads=num_threads // processes)
    if iteration_order is IterationOrder.ORDERED:
        map_function = pool.imap
    elif iteration_order is IterationOrder.UNORDERED:
        map_function = pool.imap_unordered
    else:
        raise ValueError(f"Unknown iteration order {iteration_order}")
    output_iterator: Iterator[S] = map_function(callable, iterable, chunksize)
    cm: ContextManager[Any] = pool
    return cm, output_iterator


def greater_or_close(series: pd.Series, cutoff: float) -> npt.NDArray[np.bool_]:
    value = np.asarray(series.values)
    result = np.logical_or(
        np.logical_or(value >= cutoff, np.isclose(value, cutoff)), np.isnan(value)
    )
    return result


def make_variant_mask(
    allele_frequencies: pd.Series | pd.DataFrame,
    r_squared: pd.Series,
    minor_allele_frequency_cutoff: float,
    r_squared_cutoff: float,
    aggregate_func: str = "max",
) -> npt.NDArray[np.bool_]:
    allele_frequencies = allele_frequencies.copy()
    allele_frequencies = allele_frequencies.where(
        allele_frequencies.to_numpy() <= 0.5, 1 - allele_frequencies
    )
    if isinstance(allele_frequencies, pd.DataFrame):
        allele_frequencies = allele_frequencies.aggregate(aggregate_func, axis="columns")
    variant_mask = greater_or_close(
        allele_frequencies,
        minor_allele_frequency_cutoff,
    ) & greater_or_close(r_squared, r_squared_cutoff)
    return variant_mask


def get_lock_name(lock: RLock) -> str:
    assert hasattr(lock, "_semlock")
    return lock._semlock.name


def is_bfile(path: UPath) -> bool:
    return all(
        (path.parent / f"{path.name}{suffix}").is_file()
        for suffix in {".bed", ".bim", ".fam"}
    )


def is_pfile(path: UPath) -> bool:
    return all(
        (path.parent / f"{path.name}{suffix}").is_file()
        for suffix in {".pgen", ".pvar", ".psam"}
    )


def get_processes_and_num_threads(
    num_threads: int, count: int, capacity: int
) -> tuple[int, int]:
    processes = 2 ** int(np.log2(capacity))
    processes = min((processes, count, capacity, num_threads))
    num_threads_per_process = num_threads // processes

    logger.debug(
        f"Running in {processes} processes with "
        f"{num_threads_per_process} threads each given capacity {capacity}, "
        f"thread count {num_threads}, and task count {count}"
    )

    return processes, num_threads_per_process
