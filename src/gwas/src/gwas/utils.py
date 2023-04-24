# -*- coding: utf-8 -*-

from __future__ import annotations

import multiprocessing as mp
import os
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from multiprocessing import get_context
from multiprocessing import pool as mp_pool
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from queue import Empty
from shutil import which
from typing import Any, TypeVar

import numpy as np
from numpy import typing as npt

from gwas.log import logger, setup_logging


def chromosome_to_int(chromosome: int | str) -> int:
    if chromosome == "X":
        return 23
    elif isinstance(chromosome, int):
        return chromosome
    raise ValueError(f'Unknown chromsome "{chromosome}"')


def chromosomes_set() -> set[int | str]:
    return set(range(1, 22 + 1)) | {"X"}


@dataclass
class MinorAlleleFrequencyCutoff:
    minor_allele_frequency_cutoff: float = 0.05

    def __call__(self, row) -> bool:
        mean = float(row.mean())
        minor_allele_frequency = mean / 2
        return not (
            (minor_allele_frequency < self.minor_allele_frequency_cutoff)
            or ((1 - minor_allele_frequency) < self.minor_allele_frequency_cutoff)
            or np.isclose(row.var(), 0)  # additional safety check
        )


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
        return np.format_float_scientific(x)  # type: ignore
    return str(x)


def underscore(x: str) -> str:
    return re.sub(r"([a-z\d])([A-Z])", r"\1_\2", x).lower()


def initializer(sched_affinity: set[int], logging_level: str | int) -> None:
    os.sched_setaffinity(0, sched_affinity)

    setup_logging(logging_level)


class Pool(mp_pool.Pool):
    def __init__(
        self,
        processes: int | None = None,
        maxtasksperchild: int | None = None,
        context: Any | None = None,
    ) -> None:
        initargs = (os.sched_getaffinity(0), logger.getEffectiveLevel())
        if context is None:
            context = get_context("spawn")
        super().__init__(processes, initializer, initargs, maxtasksperchild, context)


class Action(Enum):
    EVENT = auto()
    EXIT = auto()


T = TypeVar("T")


@dataclass
class SharedState:
    # Indicates that we should exit.
    should_exit: Event = field(default_factory=mp.Event)
    # Passes exceptions.
    exception_queue: Queue[Exception] = field(default_factory=mp.Queue)

    def get(self, queue: Queue[T]) -> T | Action:
        """Get an item from a queue while checking for exit.

        Args:
            queue (Queue[T]): The queue to get an item from.

        Returns:
            T | Action: The item from the queue or Action.EXIT if we should exit.
        """
        logger.debug("Waiting for queue %s", queue)
        while True:
            if self.should_exit.is_set():
                return Action.EXIT
            try:
                return queue.get(timeout=1)
            except Empty:
                pass

    def wait(self, event: Event) -> Action:
        logger.debug("Waiting for event %s", event)
        while True:
            if self.should_exit.is_set():
                return Action.EXIT
            if event.wait(timeout=1):
                return Action.EVENT


class Process(mp.Process):
    def __init__(
        self,
        exception_queue: mp.Queue[Exception] | None,
        *args,
        **kwargs,
    ) -> None:
        self.sched_affinity = os.sched_getaffinity(0)
        self.logging_level = logger.getEffectiveLevel()
        self.exception_queue = exception_queue
        super().__init__(*args, **kwargs)

    def func(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        initializer(self.sched_affinity, self.logging_level)

        try:
            self.func()
        except Exception as e:
            logger.exception("An error occurred in %s", self.name, exc_info=e)
            if self.exception_queue is not None:
                self.exception_queue.put_nowait(e)


def invert_pivot(pivot: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
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
