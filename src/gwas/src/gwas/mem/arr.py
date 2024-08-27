from dataclasses import dataclass, fields
from itertools import pairwise
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Self,
    Sequence,
    Type,
    TypeVar,
    overload,
)

import numpy as np
import scipy
from numpy import typing as npt
from upath import UPath

from .._matrix_functions import dimatcopy, set_tril
from ..compression.arr.base import (
    CompressionMethod,
    FileArray,
    default_compression_method,
)
from ..log import logger
from ..utils.multiprocessing import get_global_lock
from ..utils.numpy import invert_pivot
from .wkspace import Allocation, SharedWorkspace

ScalarType = TypeVar("ScalarType", bound=np.generic)


@dataclass
class SharedArray(Generic[ScalarType]):
    name: str
    sw: SharedWorkspace

    compression_method: ClassVar[CompressionMethod] = default_compression_method

    @property
    def allocation(self) -> Allocation:
        return self.sw.allocations[self.name]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.allocation.shape

    @property
    def start(self) -> int:
        return self.allocation.start

    @property
    def end(self) -> int:
        return self.allocation.end

    @property
    def size(self) -> int:
        return self.allocation.size

    @property
    def dtype(self) -> np.dtype[ScalarType]:
        return np.dtype(self.allocation.dtype)

    def to_file_name(self) -> str:
        raise NotImplementedError

    def __setitem__(self, key: Any, value: npt.NDArray[ScalarType]) -> None:
        numpy_array = self.to_numpy()
        numpy_array[key] = value

    def to_numpy(
        self,
        shape: Sequence[int] | None = None,
        include_trailing_free_memory: bool = False,
    ) -> npt.NDArray[ScalarType]:
        """to_numpy.

        Parameters
        ----------
        self :
            self
        shape : tuple[int, ...] | None
            override allocation shape
        include_trailing_free_memory : bool
            include_trailing_free_memory

        Returns
        -------
        npt.NDArray

        """
        allocations = self.sw.allocations
        a = allocations[self.name]

        if shape is None:
            shape = tuple(a.shape)

        dtype = np.dtype(a.dtype)

        is_at_end = all(b.start <= a.start for b in allocations.values())
        if is_at_end and include_trailing_free_memory:
            trailing_free_memory = self.sw.size - a.start - a.size

            tile_shape = shape[:-1]
            tile_size = int(np.prod(tile_shape)) * dtype.itemsize

            shape = tuple([*tile_shape, shape[-1] + trailing_free_memory // tile_size])

        if a.start % dtype.itemsize != 0:
            raise ValueError("Array is not aligned")

        return np.ndarray(
            shape,
            buffer=self.sw.buf,
            offset=a.start,
            dtype=dtype,
            order="F",
        )

    def to_metadata(self) -> dict[str, Any]:
        ignore: set[str] = {field.name for field in fields(SharedArray)}
        data = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in ignore
        }
        return data

    @staticmethod
    def get_prefix(**kwargs: str | int | None) -> str:
        return "arr"

    @classmethod
    def get_name(
        cls,
        sw: SharedWorkspace,
        *,
        prefix: str | None = None,
        name: str | None = None,
        **kwargs: str | int | None,
    ) -> str:
        if prefix is None:
            prefix = cls.get_prefix(**kwargs)

        with get_global_lock():
            allocations = sw.allocations
            if name is not None:
                if name in allocations:
                    raise ValueError(f"Allocation {name} already exists")
                return name
            i = 0
            while True:
                name = f"{prefix}-{i}"
                if name not in allocations:
                    return name
                i += 1

    @classmethod
    def from_numpy(
        cls, array: npt.NDArray[ScalarType], sw: SharedWorkspace, **kwargs
    ) -> Self:
        if array.dtype == np.object_:
            raise ValueError("Cannot create SharedArray from object array")

        with get_global_lock():
            name = cls.get_name(sw, **kwargs)
            sa = sw.alloc(name, *array.shape, dtype=array.dtype)
            a = sa.to_numpy()

        a[:] = array

        cls_names = {field.name for field in fields(cls)}
        cls_kwargs = {name: v for name, v in kwargs.items() if name in cls_names}
        if "name" in cls_kwargs:
            del cls_kwargs["name"]
        return cls(name, sw, **cls_kwargs)

    @classmethod
    def from_file(
        cls,
        file_path: UPath,
        sw: SharedWorkspace,
        dtype: Type[ScalarType],
        num_threads: int = 1,
    ) -> Self:
        reader = FileArray.from_file(file_path, dtype, num_threads)
        shape = reader.shape

        if reader.extra_metadata is not None:
            kwargs = reader.extra_metadata
        else:
            kwargs = dict()

        with get_global_lock():
            name = cls.get_name(sw, **kwargs)
            sw.alloc(name, *shape, dtype=dtype)
            array = cls(name, sw, **kwargs)
            a = array.to_numpy()

        s = slice(None)
        with reader:
            reader.read((s, s), a)
        return array

    def to_file(self, file_path: UPath, num_threads: int = 1) -> UPath:
        if file_path.is_dir():
            file_path = file_path / self.to_file_name()

        array = self.to_numpy()
        writer = FileArray.create(
            file_path,
            array.shape,
            array.dtype.type,
            self.compression_method,
            extra_metadata=self.to_metadata(),
            num_threads=num_threads,
        )

        with writer:
            writer[:, :] = array

        return writer.file_path

    def free(self) -> None:
        self.sw.free(self.name)

    def compress(self, indices: npt.NDArray[np.integer], axis: int = 0) -> None:
        """Compress rows or columns

        Parameters
        ----------
        self :
            self
        indices : npt.NDArray[np.integer]
            indices

        Returns
        -------
        None

        """
        a = self.to_numpy()

        shape = list(a.shape)

        (k,) = indices.shape
        shape[axis] = k

        compressed = self.to_numpy(shape=shape)

        a.take(indices, axis=axis, out=compressed, mode="clip")

        self.resize(*shape)

    def resize(self, *shape: int) -> None:
        """resize array

        Parameters
        ----------
        self :
            self
        shape : tuple[int, ...]
            new shape, needs to be smaller than old one

        Returns
        -------
        None

        """
        with get_global_lock():
            allocations = self.sw.allocations
            a = allocations[self.name]

            # calculate size in bytes
            itemsize = np.dtype(a.dtype).itemsize
            size = int(np.prod(shape) * itemsize)

            allocations[self.name] = Allocation(a.start, size, shape, a.dtype)
            self.sw.allocations = allocations

    @classmethod
    def merge(cls, *arrays: Self) -> Self:
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
        with get_global_lock():
            names = [a.name for a in arrays]
            sw = arrays[0].sw

            allocations = sw.allocations
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
            logger.debug(
                f'Created merged allocation "{name}" at {start} with size {size}'
            )
            sw.allocations = allocations

        return cls(name, sw)

    def transpose(
        self: "SharedArray[np.float64]", shape: tuple[int, ...] | None = None
    ) -> None:
        """Transpose the matrix in place

        Parameters
        ----------
        self :
            self
        shape : tuple[int, ...] | None
            shape

        Returns
        -------
        None

        """
        array = self.to_numpy(shape=shape)
        dimatcopy(array)

        if shape is not None:
            # no need to change the array shape, as
            # we only transposed a submatrix
            return

        # update array shape
        with get_global_lock():
            allocations = self.sw.allocations
            a = allocations[self.name]

            allocations[self.name] = Allocation(a.start, a.size, a.shape[::-1], a.dtype)

            self.sw.allocations = allocations

    @overload
    def triangularize(
        self: "SharedArray[np.float64]", pivoting: Literal[True] = True
    ) -> npt.NDArray[np.uint32]: ...

    @overload
    def triangularize(
        self: "SharedArray[np.float64]", pivoting: Literal[False]
    ) -> None: ...

    def triangularize(
        self: "SharedArray[np.float64]", pivoting: bool = True
    ) -> npt.NDArray[np.uint32] | None:
        """Triangularize to upper triangular matrix via the LAPACK routine GEQRF or
        GEQP3, which is what scipy.linalg.qr uses internally.

        Raises:
            RuntimeError: If the LAPACK routine fails.
        """
        a: npt.NDArray[np.float64] = self.to_numpy()
        jpvt: npt.NDArray[np.uint32] | None = None

        if pivoting:
            # Retrieve function
            func = scipy.linalg.get_lapack_funcs("geqp3", (a,))

            # Calculate lwork
            _, _, _, lwork, _ = func(a, lwork=-1, overwrite_a=True)
            lwork = int(lwork.item())

            # Run the computation
            _, jpvt, _, _, info = func(
                a,
                lwork=lwork,
                overwrite_a=True,
            )
            if info != 0:
                raise RuntimeError

            jpvt = np.asarray(jpvt, dtype=np.uint32)

            # Make pivot indices 0-based
            jpvt -= 1
        else:
            # Retrieve function.
            func = scipy.linalg.get_lapack_funcs("geqrf", (a,))

            # Calculate lwork.
            _, _, lwork, _ = func(a, lwork=-1, overwrite_a=True)
            lwork = int(lwork.item())

            # Direct computation for better precision as per
            # https://doi.org/10.1145/1996092.1996103
            _, _, _, info = func(
                a,
                lwork=lwork,
                overwrite_a=True,
            )
            if info != 0:
                raise RuntimeError

        # Set lower triangular part to zero.
        set_tril(a)

        return jpvt

    def apply_inverse_pivot(self, jpvt: npt.NDArray[np.uint32]) -> None:
        # Apply the inverse pivot to the columns
        matrix = self.to_numpy()
        matrix[:] = matrix[:, invert_pivot(jpvt)]
