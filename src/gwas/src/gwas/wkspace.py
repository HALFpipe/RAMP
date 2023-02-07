# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from itertools import pairwise
from multiprocessing import Condition, Lock
from multiprocessing.shared_memory import SharedMemory
import pickle
from secrets import token_hex

import numpy as np
from numpy import typing as npt

from .log import logger
from .mem import memory_limit


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

        self.lock = Lock()
        self.condition = Condition()

    def alloc(self, name: str, *shape: int, dtype: str = "f8") -> None:
        with self.lock:
            allocations = self._get_allocations()
            if name in allocations:
                raise ValueError

            # start at current end
            start = max(
                a.start + a.size
                for a in allocations.values()
            )

            # calculate size in bytes
            itemsize = np.dtype(dtype).itemsize
            size = int(np.prod(shape) * itemsize)

            # check for overflow
            end = start + size - 1
            if end >= self.shm.size:
                raise ValueError

            allocations[name] = Allocation(start, size, shape, dtype)

            logger.info(
                f"Created new allocation \"{name}\" "
                f"at {start} with size {size} "
                f"({end / self.shm.size:.0%} of workspace used)"
            )
            self._set_allocations(allocations)

        # initialize to zero
        self.get_array(name)[:] = 0

    def merge(self, *names: str) -> None:
        with self.lock:
            allocations = self._get_allocations()

            to_merge = [
                (name, a) for name, a in allocations.items()
                if name in names
            ]
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
                f"Created merged allocation \"{name}\" "
                f"at {start} with size {size} "
            )
            self._set_allocations(allocations)

    def free(self, name) -> None:
        with self.lock:
            allocations = self._get_allocations()
            del allocations[name]
            self._set_allocations(allocations)

    def _get_allocations(self) -> dict[str, Allocation]:
        return pickle.loads(
            self.shm.buf
        )

    def _set_allocations(self, allocations: dict[str, Allocation]) -> None:
        dict_size = allocations["index"].size
        dict_bytes = pickle.dumps(allocations)

        if len(dict_bytes) > dict_size:
            raise ValueError

        self.shm.buf[:len(dict_bytes)] = dict_bytes

    def get_array(
        self,
        name: str,
        include_trailing_free_memory: bool = False,
    ) -> npt.NDArray:
        allocations = self._get_allocations()
        a = allocations[name]

        shape = list(a.shape)
        dtype = np.dtype(a.dtype)

        is_at_end = all(b.start <= a.start for b in allocations.values())
        if is_at_end and include_trailing_free_memory:
            trailing_free_memory = self.shm.size - a.start - a.size

            tile_shape = shape[:-1]
            tile_size = int(np.prod(tile_shape)) * dtype.itemsize

            shape = [
                *tile_shape,
                shape[-1] + trailing_free_memory // tile_size
            ]

        buffer = self.shm.buf[a.start:]

        return np.ndarray(shape, buffer=buffer, dtype=dtype, order="F")

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()

    @classmethod
    def create(cls, dict_size: int = 2 ** 20):
        m = memory_limit()
        if not isinstance(m, int):
            raise ValueError
        n = (m // 5) * 4

        name = f"gwas-{token_hex(4)}"
        shm = SharedMemory(name=name, create=True, size=n)

        allocations: dict[str, Allocation] = dict(
            index=Allocation(start=0, size=dict_size, shape=tuple(), dtype=""),
        )
        dict_bytes = pickle.dumps(allocations)
        shm.buf[:len(dict_bytes)] = dict_bytes  # assume this is below limit

        shm.close()

        logger.info(
            f"Created shared workspace \"{name}\" "
            f"with a size of {n / 1e9:f} gigabytes"
        )

        return cls(name)
