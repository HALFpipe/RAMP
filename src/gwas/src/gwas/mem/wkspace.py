# -*- coding: utf-8 -*-

import os
import pickle
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from itertools import pairwise
from mmap import MAP_SHARED, mmap
from multiprocessing import reduction as mp_reduction
from pprint import pformat
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Self, TypeVar, overload

import numpy as np
from numpy import typing as npt
from psutil import virtual_memory

from ..log import logger
from ..utils import get_lock_name, global_lock
from ._os import c_memfd_create

if TYPE_CHECKING:
    from .arr import ScalarType, SharedArray, SharedFloat64Array

DType = TypeVar("DType", bound=np.dtype[Any])


@dataclass(frozen=True)
class Allocation:
    start: int
    size: int

    shape: tuple[int, ...]
    dtype: np.dtype[Any]


@dataclass
class SharedWorkspace(AbstractContextManager["SharedWorkspace"]):
    fd: int
    size: int

    buf: mmap = field(init=False)

    def __post_init__(self) -> None:
        self.buf = mmap(self.fd, self.size, flags=MAP_SHARED)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
        self.unlink()

    @property
    def unallocated_size(self) -> int:
        end = max(a.start + a.size for a in self.allocations.values())
        return self.size - end

    @property
    def proportion_allocated(self) -> float:
        end = max(a.start + a.size for a in self.allocations.values())
        return end / self.size

    def get_array(self, name: str) -> "SharedArray[Any]":
        from .arr import SharedArray, SharedFloat64Array

        dtype = self.allocations[name].dtype
        if np.issubdtype(dtype, np.float64):
            return SharedFloat64Array(name, self)
        else:
            return SharedArray(name, self)

    @overload
    def alloc(
        self, name: str, *shape: int, dtype: type[np.float64] | None = None
    ) -> "SharedFloat64Array": ...

    @overload
    def alloc(
        self, name: str, *shape: int, dtype: "type[ScalarType] | np.dtype[ScalarType]"
    ) -> "SharedArray[ScalarType]": ...

    def alloc(
        self,
        name: str,
        *shape: int,
        dtype: "type[ScalarType] | np.dtype[ScalarType] | None" = None,
    ) -> "SharedArray[ScalarType]":
        """alloc.

        Parameters
        ----------
        self :
            self
        name : str
            name
        shape : int
            shape
        dtype : str
            dtype
        """
        if dtype is None:
            numpy_dtype: np.dtype[ScalarType | np.float64] = np.dtype(np.float64)
        else:
            numpy_dtype = np.dtype(dtype)

        with global_lock:
            logger.debug(
                f"Acquired {global_lock} with name {get_lock_name(global_lock)}"
            )
            allocations = self.allocations
            if name in allocations:
                raise ValueError(f'Allocation "{name}" already exists')

            # start at current end
            start = max(a.start + a.size for a in allocations.values())

            # calculate size in bytes
            itemsize = numpy_dtype.itemsize
            size = int(np.prod(shape) * itemsize)

            # check for overflow
            end = start + size - 1
            if end >= self.size:
                raise MemoryError(
                    f"No space left in shared memory buffer: {pformat(allocations)}"
                )

            allocations[name] = Allocation(start, size, shape, numpy_dtype)

            logger.debug(
                f'Created new allocation "{name}" '
                f"at {start} with size {size} "
                f"({end / self.size:.0%} of workspace used)"
            )
            self.allocations = allocations

        # initialize to zero
        array = self.get_array(name)
        array.to_numpy()[:] = 0

        return array

    def free(self, name: str) -> None:
        with global_lock:
            logger.debug(
                f"Acquired {global_lock} with name {get_lock_name(global_lock)}"
            )
            allocations = self.allocations
            del allocations[name]
            self.allocations = allocations
        logger.debug(f'Free allocation "{name}" ')

    def squash(self) -> None:
        """
        Squashes all allocations in the shared memory buffer to make them
        contiguous and to reclaim space. This method moves each allocation up to the
        end of the previous allocation.
        """
        data: npt.NDArray[np.uint8] = np.ndarray(
            (self.size,),
            buffer=self.buf,
            dtype=np.uint8,
        )

        with global_lock:
            allocations = self.allocations

            to_squash = list(allocations.keys())
            to_squash.sort(key=lambda t: allocations[t].start)

            for previous_name, name in pairwise(to_squash):
                a = allocations[previous_name]
                b = allocations[name]

                if a.start + a.size == b.start:
                    continue  # already contiguous

                # move memory up to start of previous allocation
                start = a.start + a.size
                data[start : start + b.size] = data[b.start : b.start + b.size]

                logger.debug(
                    f'Moved allocation "{name}" from {b.start} to {start} '
                    f'to be contiguous with preceding allocation "{previous_name}"'
                )
                allocations[name] = Allocation(start, b.size, b.shape, b.dtype)

            self.allocations = allocations

    @property
    def allocations(self) -> dict[str, Allocation]:
        a = pickle.loads(self.buf)
        assert isinstance(a, dict)
        return a

    @allocations.setter
    def allocations(self, allocations: dict[str, Allocation]) -> None:
        acquired_lock = global_lock.acquire(block=False)
        if acquired_lock is False:
            raise RuntimeError("Global lock was not acquired")
        logger.debug(f"Acquired {global_lock} with name {get_lock_name(global_lock)}")
        try:
            dict_size = allocations["index"].size
            dict_bytes = pickle.dumps(allocations)

            if len(dict_bytes) > dict_size:
                raise ValueError

            self.buf[: len(dict_bytes)] = dict_bytes
        finally:
            global_lock.release()

    def close(self) -> None:
        self.buf.close()

    def unlink(self) -> None:
        os.close(self.fd)
        # call([
        #     "bash",
        #     "-c",
        #     f"nohup rm -f /dev/shm/{self.name} >/dev/null 2>&1 &"
        # ])

    @classmethod
    def create(cls, size: int | None = None, dict_size: int = 2**20) -> Self:
        """Creates a shared workspace that is stored in an anonymous file,
        allocated via `memfd_create`. Adapted from
        https://github.com/ska-sa/katgpucbf/blob/main/src/katgpucbf/dsim/shared_array.py

        Args:
            size (int | None, optional): Size to allocate in bytes. Defaults to None.
            dict_size (int, optional): Size to reserve for the allocations dictionary.
                Defaults to 2**20.

        Returns:
            SharedWorkspace: _description_
        """
        # Create file.
        fd = c_memfd_create("shared-workspace")
        if fd == -1:
            raise OSError("Failed to create anonymous file")
        if size is None:
            size = int(virtual_memory().available)
        os.ftruncate(fd, size)

        # Now we actually instantiate the python wrapper.
        sw = cls(fd, size)

        logger.debug(
            "Created shared workspace "
            f"with a size of {size} bytes ({size / 1e9:f} gigabytes)"
        )

        # Initialize allocations dictionary.
        allocations: dict[str, Allocation] = dict(
            index=Allocation(
                start=0, size=dict_size, shape=tuple(), dtype=np.dtype(np.uint8)
            ),
        )
        sw.allocations = allocations
        return sw


def _reduce(
    a: SharedWorkspace,
) -> tuple[Callable[[Any, int], SharedWorkspace], tuple[Any, int]]:
    return _rebuild, (mp_reduction.DupFd(a.fd), a.size)


def _rebuild(dupfd: Any, size: int) -> SharedWorkspace:
    return SharedWorkspace(dupfd.detach(), size)


mp_reduction.register(SharedWorkspace, _reduce)
