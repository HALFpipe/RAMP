# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import pickle
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from itertools import pairwise
from multiprocessing.shared_memory import SharedMemory
from secrets import token_urlsafe

import numpy as np
from _posixshmem import shm_open
from numpy import typing as npt

from ..log import logger

# taken from cpython Lib/multiprocessing/shared_memory.py
flags: int = os.O_CREAT | os.O_EXCL | os.O_RDWR
mode: int = 0o600


@dataclass(frozen=True)
class Allocation:
    start: int
    size: int

    shape: tuple[int, ...]
    dtype: str


@dataclass
class SharedWorkspace(AbstractContextManager):
    name: str

    shm: SharedMemory = field(init=False)

    def __post_init__(self) -> None:
        self.shm = SharedMemory(name=self.name)

    def __enter__(self) -> SharedWorkspace:
        return self

    def __exit__(self, *args) -> None:
        self.close()
        self.unlink()

    @property
    def size(self) -> int:
        return self.shm.size

    @property
    def unallocated_size(self) -> int:
        end = max(a.start + a.size for a in self.allocations.values())
        return self.shm.size - end

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
            raise ValueError

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

        logger.info(
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
        allocations = self.allocations

        to_merge = [(name, a) for name, a in allocations.items() if name in names]
        to_merge.sort(key=lambda t: t[-1].start)

        # check contiguous
        for (_, a), (_, b) in pairwise(to_merge):
            if a.start + a.size != b.start:
                raise ValueError

        # determine dtype
        dtype_set = set(a.dtype for _, a in to_merge)
        if len(dtype_set) != 1:
            raise ValueError
        (dtype,) = dtype_set

        # determine size
        start = min(a.start for _, a in to_merge)
        size = sum(a.size for _, a in to_merge)

        tile_shape_set = set(a.shape[:-1] for _, a in to_merge)
        if len(tile_shape_set) != 1:
            raise ValueError
        (tile_shape,) = tile_shape_set
        shape = tuple([*tile_shape, sum(a.shape[-1] for _, a in to_merge)])

        for name, _ in to_merge:
            del allocations[name]

        name, _ = to_merge[0]
        allocations[name] = Allocation(start, size, shape, dtype)
        logger.info(f'Created merged allocation "{name}" at {start} with size {size}')
        self.allocations = allocations

        return self.get_array(name)

    def free(self, name) -> None:
        allocations = self.allocations
        del allocations[name]
        self.allocations = allocations
        logger.info(f'Free allocation "{name}" ')

    def squash(self) -> None:
        data: npt.NDArray = np.ndarray(
            (self.size,),
            buffer=self.shm.buf,
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

            logger.info(
                f'Moved allocation "{name}" from {b.start} to {start} '
                f'to be contiguous with preceding allocation "{previous_name}"'
            )
            allocations[name] = Allocation(start, b.size, b.shape, b.dtype)

        self.allocations = allocations

    @property
    def allocations(self) -> dict[str, Allocation]:
        return pickle.loads(self.shm.buf)

    @allocations.setter
    def allocations(self, allocations: dict[str, Allocation]) -> None:
        dict_size = allocations["index"].size
        dict_bytes = pickle.dumps(allocations)

        if len(dict_bytes) > dict_size:
            raise ValueError

        self.shm.buf[: len(dict_bytes)] = dict_bytes

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()

    @classmethod
    def create(
        cls, size: int | None = None, dict_size: int = 2**20
    ) -> SharedWorkspace:
        # Random name (hopefully unique).
        name = f"gwas-{token_urlsafe(8)}"

        # Create file.
        fd = shm_open(name, flags, mode)

        if size is None:
            # Increase size until we hit the limit.
            size = 0
            size_step: int = 2**29  # Half a gigabyte.
            while True:
                try:
                    os.posix_fallocate(fd, size, size_step)
                    size += size_step
                except OSError:
                    break
            if not size:
                raise RuntimeError
            # Decrease size by 10% to avoid out-of-memory crashes.
            size = int(np.round(size * 0.9))
            os.ftruncate(fd, size)
        else:
            os.posix_fallocate(fd, 0, size)

        os.close(fd)

        # Now we actually instantiate the python wrapper.
        sw = cls(name)
        size = sw.size

        logger.info(
            f'Created shared workspace "{name}" '
            f"with a size of {size} bytes ({size / 1e9:f} gigabytes)"
        )

        # Initialize allocations dictionary.
        allocations: dict[str, Allocation] = dict(
            index=Allocation(start=0, size=dict_size, shape=tuple(), dtype=""),
        )
        sw.allocations = allocations
        return sw
