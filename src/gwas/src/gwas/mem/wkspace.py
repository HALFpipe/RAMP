import os
import pickle
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from functools import cache
from itertools import pairwise
from mmap import MAP_SHARED, mmap
from multiprocessing import reduction as mp_reduction
from operator import attrgetter
from pprint import pformat
from time import sleep
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Self, TypeVar, overload

import numpy as np
from numpy import typing as npt
from psutil import virtual_memory

from ..log import logger
from ..utils.multiprocessing import get_global_lock, get_lock_name
from ._os import c_memfd_create

if TYPE_CHECKING:
    from .arr import ScalarType, SharedArray

DType = TypeVar("DType", bound=np.dtype[Any])


class Candidate(NamedTuple):
    start: int
    size: int


@dataclass(frozen=True, slots=True)
class Allocation:
    start: int
    size: int

    shape: tuple[int, ...]
    dtype: np.dtype[Any]

    @property
    def end(self) -> int:
        return self.start + self.size


def round_up(number_to_round: int, multiple: int) -> int:
    """
    Rounds up a number to the nearest multiple.
    Adapted from https://stackoverflow.com/a/3407254

    Args:
        number_to_round (int): The number to be rounded up.
        multiple (int): The multiple to round up to.

    Returns:
        int: The rounded up number.

    """
    if multiple == 0:
        return number_to_round
    remainder = number_to_round % multiple
    if remainder == 0:
        return number_to_round
    return number_to_round + multiple - remainder


shared_workspaces: dict[tuple[int, int], "SharedWorkspace"] = dict()


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

    @staticmethod
    def get_key(fd: int) -> tuple[int, int]:
        # A tuple which uniquely identifies a file descriptor
        # Adapted from https://github.com/pytorch/pytorch/blob/092349dcddb39b9844d730bb0da883921e45d3e7/torch/multiprocessing/reductions.py#L525
        stat = os.fstat(fd)
        return (stat.st_ino, stat.st_dev)

    @property
    def key(self) -> tuple[int, int]:
        return self.get_key(self.fd)

    @property
    def unallocated_size(self) -> int:
        end = max(a.end for a in self.allocations.values())
        return self.size - end

    @property
    def proportion_allocated(self) -> float:
        end = max(a.end for a in self.allocations.values())
        return end / self.size

    def get_array(self, name: str) -> "SharedArray[Any]":
        from .arr import SharedArray

        dtype = self.allocations[name].dtype
        if np.issubdtype(dtype, np.float64):
            return SharedArray(name, self)
        else:
            return SharedArray(name, self)

    def get_allocation_start(self, allocation_size: int, item_size: int) -> int:
        with get_global_lock():
            allocations = self.allocations
            allocations_list = sorted(allocations.values(), key=attrgetter("start"))

            candidates: list[Candidate] = []

            def add_candidate(free_start: int, free_end: int) -> None:
                # make sure the start is aligned
                free_start = round_up(free_start, item_size)
                free_size = free_end - free_start
                if free_size >= allocation_size:
                    candidates.append(Candidate(free_size, free_start))

            # Check if there is space at the end
            free_start = allocations_list[-1].end
            free_end = self.size
            add_candidate(free_start, free_end)

            # Check if there is space between allocations
            for a, b in pairwise(allocations_list):
                free_start = a.end
                free_end = b.start
                add_candidate(free_start, free_end)

            if len(candidates) == 0:
                raise MemoryError(
                    f"No space left in shared memory buffer: {pformat(allocations)}"
                )

        start = min(candidates)[1]
        logger.debug(
            f"Found space of {allocation_size} bytes at {start} "
            f"from the following candidates: {pformat(candidates)}"
        )
        return start

    @overload
    def alloc(
        self, name: str, *shape: int, dtype: type[np.float64] | None = None
    ) -> "SharedArray": ...

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

        global_lock = get_global_lock()
        with global_lock:
            logger.debug(
                f'Acquired lock "{global_lock}" with name '
                f'"{get_lock_name(global_lock)}" to allocate "{name}"'
            )

            allocations = self.allocations
            if name in allocations:
                raise ValueError(f'Allocation "{name}" already exists')

            # Calculate size in bytes
            itemsize = numpy_dtype.itemsize
            size = int(np.prod(shape) * itemsize)

            # Calculate start
            start = self.get_allocation_start(size, itemsize)
            allocations[name] = Allocation(start, size, shape, numpy_dtype)

            end = max(a.end for a in allocations.values())
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
        global_lock = get_global_lock()
        with global_lock:
            logger.debug(
                f'Acquired lock "{global_lock}" with name '
                f' "{get_lock_name(global_lock)}" to free "{name}"'
            )

            allocations = self.allocations

            if name in allocations:
                del allocations[name]
                self.allocations = allocations

        logger.debug(f'Freed allocation "{name}" ')

    def squash(self, names: set[str] | None = None) -> None:
        """
        Squashes allocations in the shared memory buffer to make them
        contiguous and to reclaim space. This method moves each allocation up to the
        end of the previous allocation.
        """
        data: npt.NDArray[np.uint8] = np.ndarray(
            (self.size,),
            buffer=self.buf,
            dtype=np.uint8,
        )

        with get_global_lock():
            allocations = self.allocations

            to_squash = list(allocations.keys())
            to_squash.sort(key=lambda name: allocations[name].start)

            for previous_name, name in pairwise(to_squash):
                if names is not None:
                    if name not in names:
                        continue

                a = allocations[previous_name]
                b = allocations[name]

                item_size = b.dtype.itemsize

                new_start = round_up(a.end, item_size)  # Align to item size
                if new_start == b.start:
                    continue  # already contiguous
                elif new_start > b.start:
                    raise ValueError(
                        f'Allocation "{name}" overlaps with preceding allocation '
                        f'"{previous_name}"'
                    )

                # Move memory up to start of previous allocation
                logger.debug(
                    f'Moving allocation "{name}" from {b.start} to {new_start} '
                    f'to be contiguous with preceding allocation "{previous_name}" '
                    f"({a.start}-{a.end})"
                )
                data[new_start : new_start + b.size] = data[b.start : b.start + b.size]

                allocations[name] = Allocation(new_start, b.size, b.shape, b.dtype)

            self.allocations = allocations

    @property
    def allocations(self) -> dict[str, Allocation]:
        exception: Exception | None = None
        for _ in range(10):
            try:
                allocations = pickle.loads(self.buf)
                if not isinstance(allocations, dict):
                    raise ValueError(f"Expected a dictionary, got {allocations}")
            except Exception as exc:
                exception = exc
                sleep(0.1)
            else:
                break
        else:
            if exception is not None:
                raise exception
        return allocations

    @allocations.setter
    def allocations(self, allocations: dict[str, Allocation]) -> None:
        global_lock = get_global_lock()
        with global_lock:
            logger.debug(
                f"Storing {len(allocations)} allocations after locking "
                f'with lock "{get_lock_name(global_lock)}"',
                stack_info=True,
            )

            dict_size = allocations["index"].size
            dict_bytes = pickle.dumps(allocations)

            if len(dict_bytes) > dict_size:
                raise ValueError

            self.buf[: len(dict_bytes)] = dict_bytes

    def close(self) -> None:
        del shared_workspaces[self.key]
        self.buf.close()
        os.close(self.fd)

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

        shared_workspaces[sw.key] = sw
        return sw

    @classmethod
    @cache
    def rebuild(cls, fd: int, size: int) -> Self:
        return cls(fd, size)


def _reduce(
    a: SharedWorkspace,
) -> tuple[Callable[[Any, int], SharedWorkspace], tuple[Any, int]]:
    return _rebuild, (mp_reduction.DupFd(a.fd), a.size)


def _rebuild(dupfd: Any, size: int) -> SharedWorkspace:
    fd = dupfd.detach()
    key = SharedWorkspace.get_key(fd)

    if key not in shared_workspaces:
        logger.debug(
            f"Rebuilding shared workspace from file descriptor {fd} and size {size}"
        )
        wkspace = SharedWorkspace(fd, size)
        shared_workspaces[key] = wkspace
    else:
        logger.debug(
            f"Reusing shared workspace from cache for key {key}, file descriptor {fd} "
            f"and size {size}"
        )
        os.close(fd)

    return shared_workspaces[key]


mp_reduction.register(SharedWorkspace, _reduce)
