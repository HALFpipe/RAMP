# -*- coding: utf-8 -*-

from abc import abstractmethod
from contextlib import AbstractContextManager
from dataclasses import KW_ONLY, dataclass, field
from math import prod
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Generic, Mapping, MutableSequence, Type, TypeAlias, TypeVar

import numpy as np
import pandas as pd
from numpy import typing as npt


@dataclass(frozen=True, kw_only=True)
class TextCompressionMethod:
    suffix: str


@dataclass(frozen=True, kw_only=True)
class ZstdTextCompressionMethod(TextCompressionMethod):
    level: int


CompressionMethod: TypeAlias = TextCompressionMethod

zstd_high_text = ZstdTextCompressionMethod(suffix=".txt.zst", level=19)
default_compression_method = zstd_high_text
compression_methods: Mapping[str, CompressionMethod] = dict(
    zstd_text=ZstdTextCompressionMethod(suffix=".txt.zst", level=11),
    zstd_high_text=zstd_high_text,
    zstd_ultra_text=ZstdTextCompressionMethod(suffix=".txt.zst", level=22),
    gzip_text=TextCompressionMethod(suffix=".txt.gz"),
    xzip_text=TextCompressionMethod(suffix=".txt.xz"),
    bzip2_text=TextCompressionMethod(suffix=".txt.bz2"),
    lz4_text=TextCompressionMethod(suffix=".txt.lz4"),
    text=TextCompressionMethod(suffix=".txt"),
)


def compression_method_from_file(file_path: Path) -> CompressionMethod:
    for compression_method in compression_methods.values():
        if str(file_path).endswith(compression_method.suffix):
            return compression_method
    return default_compression_method


AxisMetadata: TypeAlias = pd.DataFrame | pd.Series | None
ScalarType = TypeVar("ScalarType", bound=np.generic)


def ensure_suffix(file_path: Path, compression_method: CompressionMethod) -> Path:
    if not str(file_path).endswith(compression_method.suffix):
        name = f"{file_path.name}{compression_method.suffix}"
        file_path = file_path.parent / name
    return file_path


@dataclass
class FileArray(Generic[ScalarType], AbstractContextManager["FileArray[ScalarType]"]):
    _: KW_ONLY
    file_path: Path

    shape: tuple[int, ...]
    dtype: Type[ScalarType]
    compression_method: CompressionMethod

    axis_metadata: MutableSequence[AxisMetadata] = field(init=False)
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    file_paths: set[Path] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.axis_metadata = [None] * len(self.shape)
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
        row_metadata = self.axis_metadata[0]
        if row_metadata is None:
            return 0
        elif isinstance(row_metadata, pd.Series):
            return 1
        else:
            return len(row_metadata.columns)

    @classmethod
    def create(
        cls,
        file_path: Path,
        shape: tuple[int, ...],
        dtype: Type[ScalarType],
        compression_method: CompressionMethod,
        **kwargs: Any,
    ) -> "FileArrayWriter[ScalarType]":
        if isinstance(compression_method, TextCompressionMethod):
            from .text import TextFileArrayWriter

            return TextFileArrayWriter(
                file_path=file_path,
                shape=shape,
                dtype=dtype,
                compression_method=compression_method,
                **kwargs,
            )
        else:
            raise NotImplementedError

    @classmethod
    def from_file(cls, file_path: Path, dtype: Type[ScalarType]) -> "FileArrayReader":
        compression_method = compression_method_from_file(file_path)
        if isinstance(compression_method, TextCompressionMethod):
            from .text import TextFileArrayReader

            return TextFileArrayReader.from_file(file_path, dtype)
        else:
            raise NotImplementedError


@dataclass
class FileArrayWriter(FileArray[ScalarType]):
    num_threads: int = cpu_count()

    def set_axis_metadata(self, axis: int, metadata: pd.DataFrame | pd.Series) -> None:
        self.axis_metadata[axis] = metadata

    @abstractmethod
    def __setitem__(
        self, key: tuple[slice, slice], value: npt.NDArray[ScalarType]
    ) -> None:
        raise NotImplementedError


@dataclass
class FileArrayReader(FileArray[ScalarType]):
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
