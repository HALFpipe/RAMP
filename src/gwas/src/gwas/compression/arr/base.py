from abc import abstractmethod
from contextlib import AbstractContextManager
from dataclasses import KW_ONLY, dataclass, field
from math import prod
from pprint import pformat
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Mapping,
    Type,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd
from numpy import typing as npt
from upath import UPath

from ...log import logger


@dataclass(frozen=True, kw_only=True)
class CompressionMethod:
    suffix: str


@dataclass(frozen=True, kw_only=True)
class TextCompressionMethod(CompressionMethod):
    type: ClassVar[str] = "text"


@dataclass(frozen=True, kw_only=True)
class Blosc2CompressionMethod(CompressionMethod):
    type: ClassVar[str] = "blosc2"
    suffix: str = ".b2array"


@dataclass(frozen=True, kw_only=True)
class ParquetCompressionMethod(CompressionMethod):
    type: ClassVar[str] = "parquet"
    suffix: str = ".parquet"


@dataclass(frozen=True, kw_only=True)
class ZstdTextCompressionMethod(TextCompressionMethod):
    level: int


compression_methods: Mapping[str, CompressionMethod] = dict(
    zstd_text=ZstdTextCompressionMethod(suffix=".txt.zst", level=11),
    zstd_high_text=ZstdTextCompressionMethod(suffix=".txt.zst", level=19),
    zstd_ultra_text=ZstdTextCompressionMethod(suffix=".txt.zst", level=22),
    gzip_text=TextCompressionMethod(suffix=".txt.gz"),
    xzip_text=TextCompressionMethod(suffix=".txt.xz"),
    bzip2_text=TextCompressionMethod(suffix=".txt.bz2"),
    lz4_text=TextCompressionMethod(suffix=".txt.lz4"),
    text=TextCompressionMethod(suffix=".txt"),
    blosc2=Blosc2CompressionMethod(),
    parquet=ParquetCompressionMethod(),
)
default_compression_method_name: str = "zstd_text"
default_compression_method: CompressionMethod = compression_methods[
    default_compression_method_name
]


def compression_method_from_file(file_path: UPath) -> CompressionMethod:
    for compression_method in compression_methods.values():
        if str(file_path).endswith(compression_method.suffix):
            return compression_method
    for compression_method in compression_methods.values():
        with_suffix = file_path.parent / f"{file_path.name}{compression_method.suffix}"
        if with_suffix.is_file():
            return compression_method
    return default_compression_method


ScalarType = TypeVar("ScalarType", bound=np.generic)


def ensure_suffix(file_path: UPath, compression_method: CompressionMethod) -> UPath:
    if not str(file_path).endswith(compression_method.suffix):
        name = f"{file_path.name}{compression_method.suffix}"
        file_path = file_path.parent / name
    return file_path


@dataclass(kw_only=True)
class FileArray(Generic[ScalarType]):
    _: KW_ONLY
    file_path: UPath

    shape: tuple[int, int]
    dtype: Type[ScalarType]
    compression_method: CompressionMethod

    column_names: list[str] | None = None
    row_metadata: pd.DataFrame | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    file_paths: set[UPath] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.file_paths.add(self.file_path)
        self.file_path = ensure_suffix(self.file_path, self.compression_method)
        self.file_paths.add(self.file_path)

    @property
    def itemsize(self) -> int:
        return np.dtype(self.dtype).itemsize

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def metadata_column_count(self) -> int:
        if self.row_metadata is None:
            return 0
        return len(self.row_metadata.columns)

    @classmethod
    def create(
        cls,
        file_path: UPath,
        shape: tuple[int, ...],
        dtype: Type[ScalarType],
        compression_method: CompressionMethod,
        num_threads: int,
        **kwargs: Any,
    ) -> "FileArrayWriter[ScalarType]":
        kwargs = dict(
            file_path=file_path,
            shape=shape,
            dtype=dtype,
            compression_method=compression_method,
            num_threads=num_threads,
            **kwargs,
        )
        logger.debug(f"Creating file array writer with {pformat(kwargs)}")
        if isinstance(compression_method, TextCompressionMethod):
            from .text import TextFileArrayWriter

            return TextFileArrayWriter(**kwargs)
        elif isinstance(compression_method, Blosc2CompressionMethod):
            from .blosc2 import Blosc2FileArrayWriter

            return Blosc2FileArrayWriter(**kwargs)
        elif isinstance(compression_method, ParquetCompressionMethod):
            from .parquet import ParquetFileArrayWriter

            return ParquetFileArrayWriter(**kwargs)
        else:
            raise NotImplementedError

    @classmethod
    def from_file(
        cls, file_path: UPath, dtype: Type[ScalarType], num_threads: int
    ) -> "FileArrayReader":
        compression_method = compression_method_from_file(file_path)
        if not file_path.is_file():
            if not file_path.name.endswith(compression_method.suffix):
                file_path = (
                    file_path.parent / f"{file_path.name}{compression_method.suffix}"
                )
        if isinstance(compression_method, TextCompressionMethod):
            from .text import TextFileArrayReader

            return TextFileArrayReader.from_file(file_path, dtype, num_threads)
        elif isinstance(compression_method, Blosc2CompressionMethod):
            from .blosc2 import Blosc2FileArrayReader

            return Blosc2FileArrayReader.from_file(file_path, dtype, num_threads)
        elif isinstance(compression_method, ParquetCompressionMethod):
            from .parquet import ParquetFileArrayReader

            return ParquetFileArrayReader.from_file(file_path, dtype, num_threads)
        else:
            raise NotImplementedError


@dataclass(kw_only=True)
class FileArrayWriter(
    FileArray[ScalarType], AbstractContextManager["FileArrayWriter[ScalarType]"]
):
    num_threads: int

    @overload
    def set_axis_metadata(self, axis: Literal[0], metadata: pd.DataFrame) -> None: ...
    @overload
    def set_axis_metadata(self, axis: Literal[1], metadata: list[str]) -> None: ...
    def set_axis_metadata(self, axis, metadata):
        row_count, column_count = self.shape
        if axis == 0:
            self.row_metadata = metadata
            metadata_row_count, _ = metadata.shape
            if metadata_row_count != row_count:
                raise ValueError(
                    "Row metadata does not match shape: "
                    f"{metadata_row_count} != {row_count}"
                )
        elif axis == 1:
            self.column_names = metadata
            if len(metadata) != column_count:
                raise ValueError(
                    "Column names do not match shape: "
                    f"{len(metadata)} != {column_count}"
                )

    @abstractmethod
    def __setitem__(
        self, key: tuple[slice, slice], value: npt.NDArray[ScalarType]
    ) -> None:
        raise NotImplementedError


@dataclass(kw_only=True)
class FileArrayReader(
    FileArray[ScalarType], AbstractContextManager["FileArrayReader[ScalarType]"]
):
    @abstractmethod
    def read_indices(
        self,
        row_indices: npt.NDArray[np.uint32],
        column_indices: npt.NDArray[np.uint32],
        array: npt.NDArray[ScalarType],
    ) -> None:
        raise NotImplementedError

    def read(
        self,
        key: tuple[slice | npt.NDArray[np.uint32], slice | npt.NDArray[np.uint32]],
        array: npt.NDArray[ScalarType] | None = None,
    ) -> npt.NDArray[ScalarType]:
        row_indices, column_indices = (
            np.arange(*k.indices(shape), dtype=np.uint32) if isinstance(k, slice) else k
            for k, shape in zip(key, self.shape, strict=False)
        )
        if array is None:
            array = np.empty((len(row_indices), len(column_indices)), dtype=self.dtype)
        self.read_indices(row_indices, column_indices, array)
        return array

    def __getitem__(
        self, key: tuple[slice | npt.NDArray[np.uint32], slice | npt.NDArray[np.uint32]]
    ) -> npt.NDArray[ScalarType]:
        return self.read(key)
