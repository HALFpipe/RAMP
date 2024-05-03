# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass, fields
from itertools import pairwise
from pathlib import Path
from typing import Any, Generic, Literal, Self, TypeVar, overload

import numpy as np
import scipy
from numpy import typing as npt

from .._matrix_functions import dimatcopy, set_tril
from ..compression.pipe import CompressedTextReader
from ..log import logger
from ..utils import invert_pivot
from .wkspace import Allocation, SharedWorkspace

ScalarType = TypeVar("ScalarType", bound=np.generic)


@dataclass
class SharedArray(Generic[ScalarType]):
    name: str
    sw: SharedWorkspace

    @property
    def shape(self) -> tuple[int, ...]:
        a = self.sw.allocations[self.name]
        return tuple(a.shape)

    @property
    def dtype(self) -> np.dtype[ScalarType]:
        a = self.sw.allocations[self.name]
        return np.dtype(a.dtype)

    def to_file_name(self) -> str:
        raise NotImplementedError

    def __setitem__(self, key: Any, value: npt.NDArray[ScalarType]) -> None:
        numpy_array = self.to_numpy()
        numpy_array[key] = value

    def to_numpy(
        self,
        shape: tuple[int, ...] | None = None,
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

    @classmethod
    def parse_header(cls, header: str | None) -> dict[str, Any]:
        if header is None:
            return dict()
        else:
            header = header.removeprefix("#")  # remove comment prefix
            header = header.strip()  # remove extra whitespace
            parsed_header = json.loads(header)
            assert isinstance(parsed_header, dict)
            return parsed_header

    def to_header(self) -> str | None:
        ignore = [field.name for field in fields(SharedArray)]

        data = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in ignore
        }

        return json.dumps(data)

    @staticmethod
    def get_prefix(**kwargs: str | int | None) -> str:
        return "arr"

    @classmethod
    def get_name(
        cls, sw: SharedWorkspace, prefix: str | None = None, **kwargs: str | int | None
    ) -> str:
        if prefix is None:
            prefix = cls.get_prefix(**kwargs)
        allocations = sw.allocations

        i = 0
        while True:
            name = f"{prefix}-{i}"
            if name not in allocations:
                return name
            i += 1

    @classmethod
    def from_numpy(
        cls,
        array: npt.NDArray[ScalarType],
        sw: SharedWorkspace,
        **kwargs: Any,
    ) -> Self:
        name = cls.get_name(sw, **kwargs)
        sa = sw.alloc(name, *array.shape, dtype=array.dtype)

        sa.to_numpy()[:] = array

        cls_names = {field.name for field in fields(cls)}
        cls_kwargs = {name: v for name, v in kwargs.items() if name in cls_names}
        return cls(name, sw, **cls_kwargs)

    @classmethod
    def read_file_metadata(
        cls,
        file_path: Path | str,
    ) -> tuple[int, dict[str, Any]]:
        file_path = Path(file_path)

        header: str | None = None
        first_line: str | None = None
        with CompressedTextReader(file_path) as file_handle:
            for line in file_handle:
                if line.startswith("#"):
                    header = line
                    continue

                first_line = line
                break

        if first_line is None:
            raise ValueError
        tokens = first_line.split()

        n = len(tokens)
        kwargs = cls.parse_header(header)

        return n, kwargs

    @classmethod
    def from_file(
        cls,
        file_path: Path | str,
        sw: SharedWorkspace,
        dtype: type[ScalarType] | np.dtype[ScalarType],
    ) -> Self:
        file_path = Path(file_path)
        n, kwargs = cls.read_file_metadata(file_path)

        name = cls.get_name(sw, **kwargs)

        sw.alloc(name, n, 1, dtype=dtype)
        array = cls(name, sw, **kwargs)
        a = array.to_numpy(include_trailing_free_memory=True)

        m: int = 0
        with CompressedTextReader(file_path) as file_handle:
            for line in file_handle:
                if line.startswith("#"):
                    continue
                tokens = line.split()
                a[:, m] = tokens
                m += 1

        array.resize(n, m)

        return array

    def to_file(
        self,
        file_path: Path | str,
    ) -> Path:
        file_path = Path(file_path)

        if file_path.is_dir():
            file_path = file_path / self.to_file_name()

        header = self.to_header()
        if header is None:
            header = ""

        # get shortest representation of the floating point numbers
        # that does not have any loss of precision
        a = np.vectorize(np.format_float_scientific)(self.to_numpy().transpose())
        np.savetxt(file_path, a, fmt="%s", header=header)

        return file_path

    def free(self) -> None:
        self.sw.free(self.name)

    def compress(self, indices: npt.NDArray[np.uint32]) -> None:
        """Compress rows

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

        (_, m) = a.shape
        (k,) = indices.shape
        shape = (k, m)

        compressed = self.to_numpy(shape=shape)

        a.take(indices, axis=0, out=compressed, mode="clip")

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
        # with self.sw.lock:
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
        logger.debug(f'Created merged allocation "{name}" at {start} with size {size}')
        sw.allocations = allocations

        return cls(name, sw)


class SharedFloat64Array(SharedArray[np.float64]):
    def transpose(self, shape: tuple[int, ...] | None = None) -> None:
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
        # with self.sw.lock:
        allocations = self.sw.allocations
        a = allocations[self.name]

        allocations[self.name] = Allocation(a.start, a.size, a.shape[::-1], a.dtype)

        self.sw.allocations = allocations

    @overload
    def triangularize(
        self, pivoting: Literal[True] = True
    ) -> npt.NDArray[np.uint32]: ...

    @overload
    def triangularize(self, pivoting: Literal[False]) -> None: ...

    def triangularize(self, pivoting: bool = True) -> npt.NDArray[np.uint32] | None:
        """Triangularize to upper triangular matrix via the LAPACK routine GEQRF or
        GEQP3, which is what scipy.linalg.qr uses internally.

        Raises:
            RuntimeError: If the LAPACK routine fails.
        """
        a = self.to_numpy()
        jpvt: npt.NDArray[np.uint32] | None = None

        if pivoting:
            # Retrieve function
            func = scipy.linalg.get_lapack_funcs("geqp3", (a,))

            # Calculate lwork
            _, _, _, lwork, _ = func(a, lwork=-1, overwrite_a=True)
            lwork = int(lwork)

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
