# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, ClassVar, Generic, Mapping, Type, TypeVar

import blosc2
import numpy as np
import pandas as pd
from numpy import typing as npt


@dataclass(frozen=True, kw_only=True)
class Blosc2CompressionMethod:
    codec: blosc2.Codec
    clevel: int
    filters: tuple[blosc2.Filter, ...] = field(default_factory=tuple)
    use_dict: bool = False

    suffix: ClassVar[str] = ".b2array"

    @property
    def cparams(self) -> dict[str, Any]:
        return dict(
            codec=self.codec,
            clevel=self.clevel,
            filters=list(self.filters),
            use_dict=self.use_dict,
        )


@dataclass(frozen=True, kw_only=True)
class TextCompressionMethod:
    suffix: str


@dataclass(frozen=True, kw_only=True)
class ZstdTextCompressionMethod(TextCompressionMethod):
    level: int


CompressionMethod = Blosc2CompressionMethod | TextCompressionMethod

blosc2_zstd_bitshuffle = Blosc2CompressionMethod(
    codec=blosc2.Codec.ZSTD,
    clevel=9,
    filters=(blosc2.Filter.BITSHUFFLE,),
)
default_compression_method = blosc2_zstd_bitshuffle
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
    blosc2_zstd_bitshuffle=blosc2_zstd_bitshuffle,
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


ScalarType = TypeVar("ScalarType", bound=np.generic)


@dataclass
class FileArray(Generic[ScalarType], AbstractContextManager["FileArray[ScalarType]"]):
    file_path: Path

    shape: tuple[int, ...]
    dtype: Type[ScalarType]
    compression_method: CompressionMethod

    axis_metadata: list[pd.DataFrame | pd.Series | None] = field(init=False)

    num_threads: int = cpu_count()

    file_paths: set[Path] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.axis_metadata = [None] * len(self.shape)

    def set_axis_metadata(self, axis: int, metadata: pd.DataFrame | pd.Series) -> None:
        self.axis_metadata[axis] = metadata

    @abstractmethod
    def __setitem__(
        self, key: tuple[slice, ...], value: npt.NDArray[ScalarType]
    ) -> None:
        raise NotImplementedError

    @classmethod
    def create(
        cls,
        file_path: Path,
        shape: tuple[int, ...],
        dtype: Type[ScalarType],
        compression_method: CompressionMethod,
        **kwargs: Any,
    ) -> FileArray[ScalarType]:
        if isinstance(compression_method, Blosc2CompressionMethod):
            from .bin import Blosc2FileArray

            return Blosc2FileArray(file_path, shape, dtype, compression_method, **kwargs)
        elif isinstance(compression_method, TextCompressionMethod):
            from .text import TextFileArray

            return TextFileArray(file_path, shape, dtype, compression_method, **kwargs)
        else:
            raise NotImplementedError
