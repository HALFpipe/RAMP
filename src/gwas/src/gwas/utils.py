# -*- coding: utf-8 -*-

from __future__ import annotations

import multiprocessing as mp
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, auto
from logging import LogRecord
from multiprocessing import pool as mp_pool
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event, RLock
from queue import Empty
from shutil import which
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Sequence,
    Sized,
    Type,
    TypeVar,
    get_args,
    get_origin,
)

import numpy as np
import pandas as pd
from numpy import typing as npt

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
        return np.format_float_scientific(x)
    return str(x)


def underscore(x: str) -> str:
    return re.sub(r"([a-z\d])([A-Z])", r"\1_\2", x).lower()


def get_initargs() -> tuple[set[int] | None, Queue[LogRecord] | None, int | str, RLock]:
    from .log import logging_thread

    logging_queue: Queue[LogRecord] | None = None
    if logging_thread is not None:
        logging_queue = logging_thread.logging_queue

    sched_affinity: set[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        sched_affinity = os.sched_getaffinity(0)

    initargs = (sched_affinity, logging_queue, logger.getEffectiveLevel(), global_lock)
    logger.debug(f"Initializer arguments for child process are: {initargs}")
    return initargs


def initializer(
    sched_affinity: set[int] | None,
    logging_queue: Queue[LogRecord] | None,
    log_level: int | str,
    lock: RLock,
) -> None:
    if sched_affinity is not None:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, sched_affinity)

    if logging_queue is not None:
        worker_configurer(logging_queue, log_level)
    logger.debug(f'Configured process "{mp.current_process().name}"')

    global global_lock
    global_lock = lock


class Pool(mp_pool.Pool):
    def __init__(
        self,
        processes: int | None = None,
        maxtasksperchild: int | None = None,
    ) -> None:
        initargs = get_initargs()
        super().__init__(
            processes, initializer, initargs, maxtasksperchild, multiprocessing_context
        )


class Action(Enum):
    EVENT = auto()
    EXIT = auto()


V = TypeVar("V")


@dataclass
class SharedState:
    # Indicates that we should exit.
    should_exit: Event = field(default_factory=multiprocessing_context.Event)
    # Passes exceptions.
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
        logger.debug(f'Waiting for queue "{self.get_name(event)}"')
        while True:
            if self.should_exit.is_set():
                return Action.EXIT
            if event.wait(timeout=1):
                return Action.EVENT


class Process(SpawnProcess):
    def __init__(
        self,
        exception_queue: mp.Queue[Exception] | None,
        name: str | None = None,
    ) -> None:
        self.initargs = get_initargs()
        self.exception_queue = exception_queue
        super().__init__(name=name)

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


def soft_close(proc: Process) -> None:
    try:
        proc.terminate()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
        proc.join()
    except (ValueError, AttributeError, AssertionError):
        pass
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
    chunksize: int | None = 1,
    iteration_order: IterationOrder = IterationOrder.UNORDERED,
) -> tuple[ContextManager[Any], Iterator[S]]:
    if num_threads < 2:
        return nullcontext(), map(callable, iterable)

    if isinstance(iterable, Sized):
        num_threads = min(len(iterable), num_threads)
        # Apply logic from pool.map (multiprocessing/pool.py#L481) here as well
        if chunksize is None:
            chunksize, extra = divmod(len(iterable), num_threads * 4)
            if extra:
                chunksize += 1
    if chunksize is None:
        chunksize = 1

    pool = Pool(processes=num_threads)
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
) -> npt.NDArray[np.bool_]:
    allele_frequencies = allele_frequencies.copy()
    allele_frequencies = allele_frequencies.where(
        allele_frequencies.to_numpy() <= 0.5, 1 - allele_frequencies
    )
    if isinstance(allele_frequencies, pd.DataFrame):
        allele_frequencies = allele_frequencies.max(axis="columns")
    variant_mask = greater_or_close(
        allele_frequencies,
        minor_allele_frequency_cutoff,
    ) & greater_or_close(r_squared, r_squared_cutoff)
    return variant_mask


def get_lock_name(lock: RLock) -> str:
    assert hasattr(lock, "_semlock")
    return lock._semlock.name
