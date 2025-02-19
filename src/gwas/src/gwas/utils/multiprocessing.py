import multiprocessing as mp
import os
from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from logging import LogRecord
from multiprocessing import parent_process
from multiprocessing import pool as mp_pool
from multiprocessing.queues import SimpleQueue
from multiprocessing.synchronize import Event, RLock
from pprint import pformat
from threading import Thread
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Sequence,
    Sized,
    TypeVar,
    override,
)

import numpy as np

from ..log import logger, worker_configurer

S = TypeVar("S")
T = TypeVar("T")
V = TypeVar("V")

_global_lock: RLock | None = None
if parent_process() is None:
    from ..log import multiprocessing_context

    _global_lock = multiprocessing_context.RLock()


def get_global_lock() -> RLock:
    if _global_lock is None:
        raise ValueError("Global lock not set")
    return _global_lock


class InitArgs(NamedTuple):
    sched_affinity: set[int] | None
    num_threads: int | None
    logging_queue: SimpleQueue[LogRecord] | None
    log_level: int | str
    lock: RLock
    is_jax: bool


def get_initargs(num_threads: int | None = None, is_jax: bool = False) -> InitArgs:
    from ..log import logger, logging_queue

    sched_affinity: set[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        sched_affinity = os.sched_getaffinity(0)

    if num_threads is not None:
        num_threads = max(1, num_threads)
    initargs = InitArgs(
        sched_affinity,
        num_threads,
        logging_queue,
        logger.getEffectiveLevel(),
        get_global_lock(),
        is_jax,
    )
    logger.debug(f"Initializer arguments for child process are: {pformat(initargs)}")
    return initargs


def initializer(
    sched_affinity: set[int] | None,
    num_threads: int | None,
    logging_queue: SimpleQueue[LogRecord] | None,
    log_level: int | str,
    lock: RLock,
    is_jax: bool,
) -> None:
    if logging_queue is None:
        raise ValueError("Trying to initialize process without logging queue")

    worker_configurer(logging_queue, log_level)

    if sched_affinity is not None:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, sched_affinity)
    from .threads import apply_num_threads

    apply_num_threads(num_threads)

    global _global_lock
    _global_lock = lock

    logger.debug(f'Finished initializer for process "{mp.current_process().name}"')

    if is_jax:
        from .jax import setup_jax

        setup_jax()


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


class Pool(mp_pool.Pool):
    _pool: list[mp.Process]

    def __init__(
        self,
        processes: int | None = None,
        maxtasksperchild: int | None = None,
        num_threads: int = 1,
        is_jax: bool = False,
    ) -> None:
        initargs = get_initargs(num_threads=num_threads, is_jax=is_jax)
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
    Event = auto()
    Exit = auto()


@dataclass
class SharedState:
    # Indicates that we should exit
    should_exit: Event = field(default_factory=multiprocessing_context.Event)
    # Passes exceptions
    exception_queue: SimpleQueue[Exception] = field(
        default_factory=multiprocessing_context.SimpleQueue
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

    def get(self, queue: SimpleQueue[V]) -> V | Literal[Action.Exit]:
        """Get an item from a queue while checking for exit.

        Args:
            queue (SimpleQueue[T]): The queue to get an item from.

        Returns:
            T | Action: The item from the queue or Action.EXIT if we should exit.
        """
        logger.debug(f'Waiting for queue "{self.get_name(queue)}"')
        while True:
            if self.should_exit.is_set():
                return Action.Exit
            if queue.empty():
                self.should_exit.wait(timeout=1)
                continue
            return queue.get()

    def wait(self, event: Event) -> Action:
        logger.debug(f'Waiting for event "{self.get_name(event)}"')
        while True:
            if self.should_exit.is_set():
                return Action.Exit
            if event.wait(timeout=1):
                return Action.Event


class Process(multiprocessing_context.Process):  # type: ignore
    def __init__(
        self,
        exception_queue: SimpleQueue[Exception] | None,
        num_threads: int | None,
        name: str | None = None,
    ) -> None:
        self.initargs: InitArgs = get_initargs(num_threads)
        self.exception_queue = exception_queue

        super().__init__(name=name)

    def set_num_threads(self, num_threads: int) -> None:
        self.initargs = get_initargs(num_threads=num_threads)

    @abstractmethod
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
                self.exception_queue.put(e)


def wait(running: list[Process]) -> None:
    for proc in running:
        proc.join(timeout=1)
        if proc.is_alive():
            break


def soft_close(proc: Process) -> None:
    soft_kill(proc)
    try:
        proc.close()
    except ValueError:
        pass


class IterationOrder(Enum):
    ORDERED = auto()
    UNORDERED = auto()


def make_pool_or_null_context(
    iterable: Iterable[T],
    callable: Callable[[T], S],
    num_threads: int = 1,
    size: int | None = None,
    chunksize: int | None = 1,
    iteration_order: IterationOrder = IterationOrder.UNORDERED,
    is_jax: bool = False,
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

    pool = Pool(processes=processes, num_threads=num_threads // processes, is_jax=is_jax)
    if iteration_order is IterationOrder.ORDERED:
        map_function = pool.imap
    elif iteration_order is IterationOrder.UNORDERED:
        map_function = pool.imap_unordered
    else:
        raise ValueError(f"Unknown iteration order {iteration_order}")
    output_iterator: Iterator[S] = map_function(callable, iterable, chunksize)
    cm: ContextManager[Any] = pool
    return cm, output_iterator


def get_lock_name(lock: RLock) -> str:
    if not hasattr(lock, "_semlock"):
        raise ValueError(
            f"Cannot get name of lock {lock}, as it does not have a semlock attribute"
        )
    return lock._semlock.name


def get_processes_and_num_threads(
    num_threads: int, count: int, capacity: int
) -> tuple[int, int]:
    if capacity < 1:
        capacity = 1
    processes = 2 ** int(np.log2(capacity))
    processes = min((processes, count, capacity, num_threads))
    num_threads_per_process = num_threads // processes

    logger.debug(
        f"Running in {processes} processes with "
        f"{num_threads_per_process} threads each given capacity {capacity}, "
        f"thread count {num_threads}, and task count {count}"
    )

    return processes, num_threads_per_process
