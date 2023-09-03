# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, auto
from multiprocessing import get_context
from multiprocessing import pool as mp_pool
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from queue import Empty
from shutil import which
from typing import Any, Iterable, Sequence, Type, TypeVar, get_args, get_origin

import numpy as np
import torch.multiprocessing as mp
from numpy import typing as npt

from gwas.log import logger, setup_logging

try:
    from pytest_cov.embed import cleanup_on_sigterm  # type: ignore
except ImportError:
    pass
else:
    cleanup_on_sigterm()


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


def chromosomes_set() -> set[int | str]:
    return set(range(1, 22 + 1)) | {"X"}


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


def get_initargs() -> tuple:
    return os.sched_getaffinity(0), logger.getEffectiveLevel()


def initializer(
    sched_affinity: set[int],
    logging_level: str | int,
) -> None:
    os.sched_setaffinity(0, sched_affinity)

    setup_logging(logging_level)


class Pool(mp_pool.Pool):
    def __init__(
        self,
        processes: int | None = None,
        maxtasksperchild: int | None = None,
        context: Any | None = None,
    ) -> None:
        initargs = get_initargs()
        if context is None:
            context = get_context("spawn")
        super().__init__(processes, initializer, initargs, maxtasksperchild, context)


class Action(Enum):
    EVENT = auto()
    EXIT = auto()


V = TypeVar("V")


@dataclass
class SharedState:
    # Indicates that we should exit.
    should_exit: Event = field(default_factory=mp.Event)
    # Passes exceptions.
    exception_queue: Queue[Exception] = field(default_factory=mp.Queue)

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


class Process(mp.Process):
    def __init__(
        self,
        exception_queue: mp.Queue[Exception] | None,
        *args,
        **kwargs,
    ) -> None:
        self.initargs = get_initargs()
        self.exception_queue = exception_queue
        super().__init__(*args, **kwargs)

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

    return data


def make_sample_boolean_vectors(
    base_samples: list[str],
    samples_iterable: Iterable[list[str]],
) -> list[npt.NDArray[np.bool_]]:
    return [
        np.fromiter((sample in samples for sample in base_samples), dtype=np.bool_)
        for samples in samples_iterable
    ]
