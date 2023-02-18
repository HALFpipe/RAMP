# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
from dataclasses import dataclass
from itertools import pairwise
from multiprocessing import RLock
from multiprocessing.shared_memory import SharedMemory
from secrets import token_hex

import numpy as np
from numpy import typing as npt

from ..log import logger
from .lim import memory_limit


@dataclass(frozen=True)
class Allocation:
    start: int
    size: int

    shape: tuple[int, ...]
    dtype: str


class SharedWorkspace:
    def __init__(
        self,
        name: str,
    ) -> None:
        self.shm = SharedMemory(name=name)

        self.lock = RLock()

    @property
    def size(self) -> int:
        return self.shm.size

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
        with self.lock:
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
        with self.lock:
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
            logger.info(
                f'Created merged allocation "{name}" at {start} with size {size}'
            )
            self.allocations = allocations

        return self.get_array(name)

    def free(self, name) -> None:
        with self.lock:
            allocations = self.allocations
            del allocations[name]
            self.allocations = allocations

    def squash(self) -> None:
        data: npt.NDArray = np.ndarray(
            (self.size,),
            buffer=self.shm.buf,
            dtype=np.uint8,
        )

        with self.lock:
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
        if not self.lock.acquire(block=False):
            raise ValueError

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
        if size is None:
            available_memory = memory_limit()
            if not isinstance(available_memory, int):
                raise ValueError
            size = (available_memory // 5) * 4

        name = f"gwas-{token_hex(4)}"
        shm = SharedMemory(name=name, create=True, size=size)

        allocations: dict[str, Allocation] = dict(
            index=Allocation(start=0, size=dict_size, shape=tuple(), dtype=""),
        )
        dict_bytes = pickle.dumps(allocations)
        shm.buf[: len(dict_bytes)] = dict_bytes  # assume this is below limit

        shm.close()

        logger.info(
            f'Created shared workspace "{name}" '
            f"with a size of {size / 1e9:f} gigabytes"
        )

        return cls(name)
