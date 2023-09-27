# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import pickle
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from itertools import pairwise
from mmap import MAP_SHARED, mmap
from multiprocessing import reduction as mp_reduction
from typing import Callable

import numpy as np
from numpy import typing as npt
from psutil import virtual_memory

from ..log import logger
from ._os import c_memfd_create


@dataclass(frozen=True)
class Allocation:
    start: int
    size: int

    shape: tuple[int, ...]
    dtype: str


@dataclass
class SharedWorkspace(AbstractContextManager):
    fd: int
    size: int

    buf: mmap = field(init=False)

    def __post_init__(self) -> None:
        self.buf = mmap(self.fd, self.size, flags=MAP_SHARED)

    def __enter__(self) -> SharedWorkspace:
        return self

    def __exit__(self, *args) -> None:
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

    def get_array(self, name: str):
        from .arr import SharedArray

        return SharedArray(name, self)

    def alloc(self, name: str, *shape: int, dtype: str = "f8"):
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
        allocations = self.allocations
        if name in allocations:
            raise ValueError(f'Allocation "{name}" already exists')

        # start at current end
        start = max(a.start + a.size for a in allocations.values())

        # calculate size in bytes
        itemsize = np.dtype(dtype).itemsize
        size = int(np.prod(shape) * itemsize)

        # check for overflow
        end = start + size - 1
        if end >= self.size:
            raise MemoryError

        allocations[name] = Allocation(start, size, shape, dtype)

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

    def merge(self, *names: str):
        """Merge multiple allocations into a single contiguous allocation.

        Parameters
        ----------
        self : SharedWorkspace
            The shared workspace instance.
        *names : str
            The names of the allocations to merge.

        Returns
        -------
        SharedArray
            A new shared array representing the merged allocation.

        Raises
        ------
        ValueError
            If the allocations are not contiguous or have different dtypes or
            incompatible shapes.
        """
        allocations = self.allocations

        to_merge = [(name, a) for name, a in allocations.items() if name in names]
        to_merge.sort(key=lambda t: t[-1].start)

        # Check contiguous
        for (_, a), (_, b) in pairwise(to_merge):
            if a.start + a.size != b.start:
                raise ValueError(f'Allocations "{a}" and "{b}" are not contiguous')

        # Determine dtype
        dtype_set = set(a.dtype for _, a in to_merge)
        if len(dtype_set) != 1:
            raise ValueError("Allocations have different dtypes")
        (dtype,) = dtype_set

        # determine size
        start = min(a.start for _, a in to_merge)
        size = sum(a.size for _, a in to_merge)

        tile_shape_set = set(a.shape[:-1] for _, a in to_merge)
        if len(tile_shape_set) != 1:
            raise ValueError("Allocations have incompatible shapes")
        (tile_shape,) = tile_shape_set
        shape = tuple([*tile_shape, sum(a.shape[-1] for _, a in to_merge)])

        for name, _ in to_merge:
            del allocations[name]

        name, _ = to_merge[0]
        allocations[name] = Allocation(start, size, shape, dtype)
        logger.debug(f'Created merged allocation "{name}" at {start} with size {size}')
        self.allocations = allocations

        return self.get_array(name)

    def free(self, name) -> None:
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
        data: npt.NDArray = np.ndarray(
            (self.size,),
            buffer=self.buf,
            dtype=np.uint8,
        )

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
        return pickle.loads(self.buf)

    @allocations.setter
    def allocations(self, allocations: dict[str, Allocation]) -> None:
        dict_size = allocations["index"].size
        dict_bytes = pickle.dumps(allocations)

        if len(dict_bytes) > dict_size:
            raise ValueError

        self.buf[: len(dict_bytes)] = dict_bytes

    def close(self):
        self.buf.close()

    def unlink(self):
        os.close(self.fd)
        # call([
        #     "bash",
        #     "-c",
        #     f"nohup rm -f /dev/shm/{self.name} >/dev/null 2>&1 &"
        # ])

    @classmethod
    def create(
        cls, size: int | None = None, dict_size: int = 2**20
    ) -> SharedWorkspace:
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
            index=Allocation(start=0, size=dict_size, shape=tuple(), dtype=""),
        )
        sw.allocations = allocations
        return sw


def _reduce(a: SharedWorkspace) -> tuple[Callable, tuple]:
    return _rebuild, (mp_reduction.DupFd(a.fd), a.size)


def _rebuild(dupfd, size: int) -> SharedWorkspace:
    return SharedWorkspace(dupfd.detach(), size)


mp_reduction.register(SharedWorkspace, _reduce)
