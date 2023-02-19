# -*- coding: utf-8 -*-
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import scipy
from numpy import typing as npt

from .._matrix_functions import dimatcopy, set_tril
from ..vcf import CompressedTextFile
from .wkspace import Allocation, SharedWorkspace


@dataclass
class SharedArray:
    name: str
    sw: SharedWorkspace

    @property
    def shape(self) -> tuple[int, ...]:
        a = self.sw.allocations[self.name]
        return tuple(a.shape)

    def to_file_name(self) -> str:
        raise NotImplementedError

    def to_numpy(
        self,
        shape: tuple[int, ...] | None = None,
        include_trailing_free_memory: bool = False,
    ) -> npt.NDArray:
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
            trailing_free_memory = self.sw.shm.size - a.start - a.size

            tile_shape = shape[:-1]
            tile_size = int(np.prod(tile_shape)) * dtype.itemsize

            shape = tuple([*tile_shape, shape[-1] + trailing_free_memory // tile_size])

        buffer = self.sw.shm.buf[a.start :]

        return np.ndarray(shape, buffer=buffer, dtype=dtype, order="F")

    @classmethod
    def parse_header(cls, header: str | None) -> dict:
        if header is None:
            return dict()
        else:
            header = header.removeprefix("#")  # remove comment prefix
            header = header.strip()  # remove extra whitespace
            return json.loads(header)

    def to_header(self) -> str | None:
        ignore = [field.name for field in fields(SharedArray)]

        data = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in ignore
        }

        return json.dumps(data)

    @staticmethod
    def get_prefix(**kwargs) -> str:
        return "arr"

    @classmethod
    def get_name(cls, sw: SharedWorkspace, **kwargs) -> str:
        prefix = cls.get_prefix(**kwargs)
        allocations = sw.allocations

        i = 0
        while True:
            name = f"{prefix}-{i}"
            if name not in allocations:
                return name
            i += 1

    @classmethod
    def from_file(
        cls,
        file_path: Path | str,
        sw: SharedWorkspace,
        name: str | None = None,
    ):
        file_path = Path(file_path)

        header: str | None = None
        first_line: str | None = None
        with CompressedTextFile(file_path) as file_handle:
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
        name = cls.get_name(sw, **kwargs)

        sw.alloc(name, n, 1)
        array = cls(name, sw, **kwargs)
        a = array.to_numpy(include_trailing_free_memory=True)

        m: int = 0
        with CompressedTextFile(file_path) as file_handle:
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
        np.savetxt(file_path, self.to_numpy().transpose(), header=header)

        return file_path

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
        dimatcopy(self.to_numpy(shape=shape))

        if shape is not None:
            # no need to change the array shape, as
            # we only transposed a submatrix
            return

        # update array shape
        with self.sw.lock:
            allocations = self.sw.allocations
            a = allocations[self.name]

            allocations[self.name] = Allocation(a.start, a.size, a.shape[::-1], a.dtype)

            self.sw.allocations = allocations

    def compress(self, indices: npt.NDArray[np.integer]) -> None:
        """compress rows

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
        with self.sw.lock:
            allocations = self.sw.allocations
            a = allocations[self.name]

            # calculate size in bytes
            itemsize = np.dtype(a.dtype).itemsize
            size = int(np.prod(shape) * itemsize)

            allocations[self.name] = Allocation(a.start, size, shape, a.dtype)
            self.sw.allocations = allocations

    def triangularize(self):
        """Triangularize to upper triangular matrix via GEQRF, which
        is what scipy.linalg.qr uses internally.

        Parameters
        ----------
        self :
            self
        """
        a = self.to_numpy()

        # retrieve function
        func = scipy.linalg.get_lapack_funcs("geqrf", (a,))

        # calculate lwork
        _, _, lwork, _ = func(a, lwork=-1, overwrite_a=True)
        lwork = int(lwork)

        # direct computation for better precision
        # https://doi.org/10.1145/1996092.1996103
        _, _, _, info = func(
            a,
            lwork=lwork,
            overwrite_a=True,
        )
        if info != 0:
            raise RuntimeError

        # remove lower triangle
        set_tril(a)
