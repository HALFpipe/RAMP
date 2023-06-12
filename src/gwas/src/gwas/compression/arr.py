# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, ClassVar, Generic, Mapping, Sequence, Type, TypeVar

import blosc2
import numcodecs
import numpy as np
import zarr
from numcodecs.abc import Codec as ZarrCodec

from ..log import logger


@dataclass(frozen=True, kw_only=True)
class CompressionMethod:
    suffix: ClassVar[str] = ""


@dataclass(frozen=True, kw_only=True)
class ZarrCompressionMethod(CompressionMethod):
    codec: ZarrCodec
    filters: tuple[ZarrCodec, ...] = field(default_factory=tuple)

    suffix: ClassVar[str] = ".zarr"


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
    zarr_zstd=ZarrCompressionMethod(
        codec=numcodecs.Zstd(level=11),
    ),
    zarr_zstd_shuffle=ZarrCompressionMethod(
        codec=numcodecs.Blosc(
            cname="zstd",
            clevel=9,
            shuffle=numcodecs.Blosc.SHUFFLE,
        ),
    ),
    zarr_zstd_bitshuffle=ZarrCompressionMethod(
        codec=numcodecs.Blosc(
            cname="zstd",
            clevel=9,
            shuffle=numcodecs.Blosc.BITSHUFFLE,
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
        if isinstance(self.compression_method, ZarrCompressionMethod):
            array = self.get_zarr()
        elif isinstance(self.compression_method, Blosc2CompressionMethod):
            array = self.get_blosc2_ndarray()
        else:
            raise ValueError

        array[key] = value

    def get_zarr(self) -> zarr.Array:
        if not isinstance(self.compression_method, ZarrCompressionMethod):
            raise RuntimeError

        chunks = self.chunk_shape if self.chunk_shape is not None else True

        array = zarr.open_array(
            store=str(self.file_path),
            mode="a",
            shape=self.shape,
            dtype=self.dtype,
            chunks=chunks,  # type: ignore
            compressor=self.compression_method.codec,  # type: ignore
            filters=list(self.compression_method.filters),
        )

        if array.shape != self.shape:
            raise RuntimeError(f"Array shape {array.shape} does not match {self.shape}")

        return array

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
                raise RuntimeError("This should not happen")
            return shape
        else:
            raise RuntimeError("Empty dimensions remain")

    def get_blosc2_ndarray(self) -> blosc2.NDArray:
        if not isinstance(self.compression_method, Blosc2CompressionMethod):
            raise RuntimeError

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

        if array.shape != self.shape:
            raise ValueError
        if array.dtype != self.dtype:
            raise ValueError

        return array

    def __post_init__(self) -> None:
        if not str(self.file_path).endswith(self.compression_method.suffix):
            raise ValueError(
                f"File path must have `{self.compression_method.suffix}` suffix"
            )
