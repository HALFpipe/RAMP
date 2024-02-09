# -*- coding: utf-8 -*-

import pickle
from dataclasses import dataclass
from math import prod
from multiprocessing import cpu_count
from types import TracebackType
from typing import Any, Self, Sequence, Type

import blosc2
import numpy as np
from numpy import typing as npt

from ...log import logger
from ..pipe import CompressedBytesWriter
from .base import Blosc2CompressionMethod, FileArray, T


@dataclass
class Blosc2FileArray(FileArray[T]):
    chunk_shape: tuple[int, ...] | None = None

    def __setitem__(self, key: tuple[slice, ...], value: npt.NDArray[T]) -> None:
        if isinstance(self.compression_method, Blosc2CompressionMethod):
            array = self.get_blosc2_ndarray()
            array[key] = value
        else:
            raise NotImplementedError

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def reduce_shape(
        self,
        shape: tuple[int, ...],
        max_size: int,
    ) -> tuple[int, ...]:
        max_size //= np.dtype(self.dtype).itemsize
        axis_count = len(shape)
        reduced_shape: Sequence[int | None] = [None] * axis_count
        while any(s is None for s in reduced_shape):
            # Divide the maximum size by the shape of the known axes
            available_size = max_size / np.prod(
                [s for s in reduced_shape if s is not None]
            )
            # Use the n-th root to determine the ideal shape of the unknown axes
            unknown_count = sum(s is None for s in reduced_shape)
            edge_size = int(np.floor(np.power(available_size, 1 / unknown_count)))
            if any(
                s < edge_size
                for c, s in zip(reduced_shape, shape, strict=True)
                if c is None
            ):
                reduced_shape = [
                    s if s < edge_size else r
                    for r, s in zip(reduced_shape, shape, strict=True)
                ]
                continue
            reduced_shape = [edge_size if r is None else r for r in reduced_shape]

        if all(isinstance(s, int) for s in reduced_shape):
            shape = tuple(s for s in reduced_shape if s is not None)
            if prod(shape) > max_size:
                raise RuntimeError(
                    "Shape reduction failed to produce size within bounds"
                )
            return shape
        else:
            raise RuntimeError("Empty dimensions remain")

    def get_blosc2_ndarray(self) -> blosc2.NDArray:
        if not isinstance(self.compression_method, Blosc2CompressionMethod):
            raise RuntimeError(
                "Tried to get Blosc2 array with non-Blosc2 compression method"
            )

        blosc2.set_nthreads(cpu_count())

        file_path = self.file_path
        if not str(file_path).endswith(self.compression_method.suffix):
            file_path = (
                file_path.parent / f"{file_path.name}{self.compression_method.suffix}"
            )
        self.file_paths.add(file_path)

        if not file_path.is_file():
            kwargs: dict[str, Any] = dict()
            # Chunk size of 1024 megabytes
            max_chunk_size = 2**30
            if self.chunk_shape is not None:
                chunk_shape = self.reduce_shape(self.chunk_shape, max_chunk_size)
            else:
                chunk_shape = self.reduce_shape(self.shape, max_chunk_size)
            kwargs["chunks"] = chunk_shape
            # Block size of 32 megabytes
            max_block_size = 2**25
            block_shape = self.reduce_shape(chunk_shape, max_block_size)
            kwargs["blocks"] = block_shape
            logger.debug(f"Creating Blosc2 array with kwargs {kwargs}")
            array = blosc2.empty(
                shape=self.shape,
                urlpath=str(file_path),
                dtype=self.dtype,  # type: ignore
                cparams=dict(
                    codec=self.compression_method.codec,
                    clevel=self.compression_method.clevel,
                    filters=list(self.compression_method.filters),
                    nthreads=self.num_threads,
                    use_dict=self.compression_method.use_dict,
                ),
                dparams=dict(
                    nthreads=self.num_threads,
                ),
                **kwargs,
            )
            axis_metadata_path = (
                file_path.parent / f"{file_path.stem}.axis-metadata.pkl.zst"
            )
            with CompressedBytesWriter(axis_metadata_path) as file_handle:
                pickle.dump(self.axis_metadata, file_handle)
        else:
            array = blosc2.open(
                urlpath=str(file_path),
                cparams=dict(
                    nthreads=self.num_threads,
                ),
                dparams=dict(
                    nthreads=self.num_threads,
                ),
            )

        logger.debug(f"Using Blosc2 array {array.info}")

        if array.shape != self.shape or array.dtype != self.dtype:
            raise ValueError(
                f'Existing array at "{self.file_path}" with shape {array.shape} and '
                f"dtype {array.dtype} does not match required shape {self.shape} or "
                f"dtype {self.dtype}. Please delete the file or use a different "
                "output directory"
            )

        return array
