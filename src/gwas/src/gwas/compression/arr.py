# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, ClassVar, Generic, Mapping, Sequence, Type, TypeVar

import blosc2
import numpy as np

from ..log import logger


@dataclass(frozen=True, kw_only=True)
class CompressionMethod:
    suffix: ClassVar[str] = ""


@dataclass(frozen=True, kw_only=True)
class Blosc2CompressionMethod(CompressionMethod):
    codec: blosc2.Codec
    clevel: int
    filters: tuple[blosc2.Filter, ...] = field(default_factory=tuple)
    use_dict: bool = False

    suffix: ClassVar[str] = ".b2array"


compression_methods: Mapping[str, CompressionMethod] = dict(
    blosc2_zstd=Blosc2CompressionMethod(
        codec=blosc2.Codec.ZSTD,
        clevel=9,
    ),
    blosc2_zstd_shuffle=Blosc2CompressionMethod(
        codec=blosc2.Codec.ZSTD,
        clevel=9,
        filters=(blosc2.Filter.SHUFFLE,),
    ),
    blosc2_zstd_bitshuffle=Blosc2CompressionMethod(
        codec=blosc2.Codec.ZSTD,
        clevel=9,
        filters=(blosc2.Filter.BITSHUFFLE,),
    ),
    blosc2_zstd_bitshuffle_use_dict=Blosc2CompressionMethod(
        codec=blosc2.Codec.ZSTD,
        clevel=9,
        filters=(blosc2.Filter.BITSHUFFLE,),
        use_dict=True,
    ),
    blosc2_zstd_bytedelta=Blosc2CompressionMethod(
        codec=blosc2.Codec.ZSTD,
        clevel=9,
        filters=(
            blosc2.Filter.SHUFFLE,
            blosc2.Filter.BYTEDELTA,
        ),
    ),
)


T = TypeVar("T", bound=np.generic)


@dataclass
class ArrayProxy(Generic[T]):
    file_path: Path

    shape: tuple[int, ...]
    dtype: Type[T]

    compression_method: CompressionMethod = compression_methods["blosc2_zstd_bytedelta"]
    num_threads: int = cpu_count()

    chunk_shape: tuple[int, ...] | None = None

    def __setitem__(self, key: tuple[slice, ...], value: np.ndarray) -> None:
        if isinstance(self.compression_method, Blosc2CompressionMethod):
            array = self.get_blosc2_ndarray()
        else:
            raise ValueError

        array[key] = value

    def reduce_shape(
        self,
        shape: tuple[int, ...],
        max_size: int,
    ) -> tuple[int, ...]:
        max_size //= self.dtype().itemsize
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
            if any(s < edge_size for c, s in zip(reduced_shape, shape) if c is None):
                reduced_shape = [
                    s if s < edge_size else r for r, s in zip(reduced_shape, shape)
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

        if not self.file_path.is_file():
            kwargs: dict[str, Any] = dict()
            # Chunk size of 512 megabytes
            max_chunk_size = 2**29
            if self.chunk_shape is not None:
                chunk_shape = self.reduce_shape(self.chunk_shape, max_chunk_size)
            else:
                chunk_shape = self.reduce_shape(self.shape, max_chunk_size)
            kwargs["chunks"] = chunk_shape
            # Block size of 16 megabytes
            max_block_size = 2**24
            block_shape = self.reduce_shape(chunk_shape, max_block_size)
            kwargs["blocks"] = block_shape
            logger.debug(f"Creating Blosc2 array with kwargs {kwargs}")
            array = blosc2.empty(
                shape=self.shape,
                urlpath=str(self.file_path),
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
        else:
            array = blosc2.open(
                urlpath=str(self.file_path),
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

    def __post_init__(self) -> None:
        if not str(self.file_path).endswith(self.compression_method.suffix):
            raise ValueError(
                f"File path must have `{self.compression_method.suffix}` suffix"
            )
