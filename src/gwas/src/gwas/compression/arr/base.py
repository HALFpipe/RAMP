# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Generic, Mapping, Type, TypeVar

import blosc2
import numpy as np
import pandas as pd


@dataclass(frozen=True, kw_only=True)
class CompressionMethod:
    suffix: str = ""


@dataclass(frozen=True, kw_only=True)
class Blosc2CompressionMethod(CompressionMethod):
    codec: blosc2.Codec
    clevel: int
    filters: tuple[blosc2.Filter, ...] = field(default_factory=tuple)
    use_dict: bool = False

    suffix: str = ".b2array"


@dataclass(frozen=True, kw_only=True)
class TextCompressionMethod(CompressionMethod):
    suffix: str


@dataclass(frozen=True, kw_only=True)
class ZstdTextCompressionMethod(TextCompressionMethod):
    level: int


compression_methods: Mapping[str, CompressionMethod] = dict(
    zstd_text=ZstdTextCompressionMethod(suffix=".txt.zst", level=11),
    zstd_ultra_text=ZstdTextCompressionMethod(suffix=".txt.zst", level=22),
    gzip_text=TextCompressionMethod(suffix=".txt.gz"),
    xzip_text=TextCompressionMethod(suffix=".txt.xz"),
    bzip2_text=TextCompressionMethod(suffix=".txt.bz2"),
    lz4_text=TextCompressionMethod(suffix=".txt.lz4"),
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
class FileArray(Generic[T], AbstractContextManager):
    file_path: Path

    shape: tuple[int, ...]
    dtype: Type[T]
    compression_method: CompressionMethod

    axis_metadata: list[pd.DataFrame | pd.Series | None] = field(init=False)

    num_threads: int = cpu_count()

    file_paths: set[Path] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.axis_metadata = [None] * len(self.shape)

    def set_axis_metadata(self, axis: int, metadata: pd.DataFrame | pd.Series) -> None:
        self.axis_metadata[axis] = metadata

    @abstractmethod
    def __setitem__(self, key: tuple[slice, ...], value: np.ndarray) -> None:
        raise NotImplementedError

    @classmethod
    def create(
        cls,
        file_path: Path,
        shape: tuple[int, ...],
        dtype: Type[T],
        compression_method: CompressionMethod,
        **kwargs: Any,
    ) -> FileArray[T]:
        if isinstance(compression_method, Blosc2CompressionMethod):
            from .bin import Blosc2FileArray

            return Blosc2FileArray(
                file_path, shape, dtype, compression_method, **kwargs
            )
        elif isinstance(compression_method, TextCompressionMethod):
            from .text import TextFileArray

            return TextFileArray(file_path, shape, dtype, compression_method, **kwargs)
        else:
            raise NotImplementedError
