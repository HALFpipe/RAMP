# -*- coding: utf-8 -*-

import json
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Iterator, Type, override

import numpy as np
import pandas as pd
from numpy import typing as npt

from gwas.compression.arr._read_float import read_float
from gwas.compression.arr._read_str import (
    read_str,
)

from ...log import logger
from ...utils import to_str
from ..pipe import CompressedTextReader, CompressedTextWriter
from ._write_float import write_float
from .base import (
    FileArrayReader,
    FileArrayWriter,
    ScalarType,
    TextCompressionMethod,
    ZstdTextCompressionMethod,
    compression_method_from_file,
)

delimiter: str = "\t"
extra_metadata_prefix: str = "# "


def make_args_tuple(*args: Any) -> tuple[Any, ...]:
    return args


def read_header(
    file_path: Path,
) -> tuple[pd.Series | pd.DataFrame, pd.Series, dict[str, Any] | None, int, int]:
    reader = CompressedTextReader(file_path)
    extra_metadata: dict[str, Any] | None = None
    header_line: str | None = None
    header_length: int = 0

    with reader as file_handle:
        for line in file_handle:
            header_length += len(line)
            if line.startswith(extra_metadata_prefix):
                line = line.removeprefix(extra_metadata_prefix)
                extra_metadata = json.loads(line)
                continue
            header_line = line
            break
    if header_line is None:
        raise ValueError("No header line found in text file")
    column_names = pd.Series(header_line.strip().split(delimiter), dtype=str)
    column_count = len(column_names)

    metadata_column_indices: npt.NDArray[np.uint32] = np.asarray([0], dtype=np.uint32)
    if extra_metadata is not None and "metadata_columns" in extra_metadata:
        metadata_column_indices = np.asarray(
            extra_metadata.pop("metadata_column_indices"), dtype=np.uint32
        )
    if metadata_column_indices != np.arange(len(metadata_column_indices)):
        raise ValueError("Metadata columns must be contiguous and start at 0")
    metadata_column_names = column_names.iloc[metadata_column_indices]
    column_names = column_names.drop(metadata_column_indices, inplace=False)

    metadata_tuples: list[tuple[Any, ...]] = list()
    with reader as file_handle:
        read_str(
            metadata_tuples,
            make_args_tuple,
            file_handle.fileno(),
            header_length,
            column_count,
            metadata_column_indices,
        )

    row_metadata = pd.DataFrame(metadata_tuples, columns=metadata_column_names)
    if len(row_metadata.columns) == 1:
        row_metadata = row_metadata.iloc[:, 0]

    return row_metadata, column_names, extra_metadata, header_length, column_count


@dataclass
class TextFileArrayReader(FileArrayReader[ScalarType]):
    _: KW_ONLY
    compression_method: TextCompressionMethod
    header_length: int
    column_count: int

    @override
    @classmethod
    def from_file(cls, file_path: Path, dtype: Type[ScalarType]) -> FileArrayReader:
        row_metadata, column_metadata, extra_metadata, header_length, column_count = (
            read_header(file_path)
        )
        shape = (row_metadata.shape[0], column_metadata.size)
        compression_method = compression_method_from_file(file_path)
        reader = TextFileArrayReader(
            file_path=file_path,
            shape=shape,
            dtype=dtype,
            compression_method=compression_method,
            extra_metadata=extra_metadata,
            header_length=header_length,
            column_count=column_count,
        )
        reader.axis_metadata = [row_metadata, column_metadata]
        return reader

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def read_indices(
        self,
        row_indices: npt.NDArray[np.uint32],
        column_indices: npt.NDArray[np.uint32],
        array: npt.NDArray[ScalarType],
    ) -> None:
        reader = CompressedTextReader(self.file_path)

        column_indices = column_indices.copy()
        column_indices += self.metadata_column_count

        if isinstance(self.dtype, np.float64):
            with reader as file_handle:
                read_float(
                    array,
                    file_handle.fileno(),
                    self.header_length,
                    self.column_count,
                    column_indices,
                    row_indices,
                    ring_buffer_size=reader.pipesize,
                )
        else:
            raise NotImplementedError


def unpack_key(
    shape: tuple[int, ...], row_index: int, key: tuple[slice, slice]
) -> tuple[int, int, int, int]:
    if len(key) != len(shape):
        raise ValueError("Key must have same number of dimensions as array")

    row_slice, col_slice = key
    row_start, row_step, row_stop = row_slice.start, row_slice.step, row_slice.stop
    col_start, col_step, col_stop = col_slice.start, col_slice.step, col_slice.stop

    row_start = 0 if row_start is None else row_start
    col_start = 0 if col_start is None else col_start

    row_stop = shape[0] if row_stop is None else row_stop
    col_stop = shape[1] if col_stop is None else col_stop

    if row_step is not None and row_step != 1:
        raise ValueError("Can only write text file sequentially, row step must be 1")
    if col_step is not None and col_step != 1:
        raise ValueError("Can only write text file sequentially, column step must be 1")
    if row_start != row_index:
        raise ValueError(
            "Can only write text file sequentially, row start must be "
            f"{row_index} (got {row_start})"
        )
    if col_start != 0:
        raise ValueError(
            "Can only write text file sequentially, column start must be 0 "
            f"(got {col_start})"
        )
    if col_stop != shape[1]:
        raise ValueError(
            "Can only write text file sequentially, column stop must be "
            f"{shape[1]} (got {col_start})"
        )

    return row_start, row_stop, col_start, col_stop


@dataclass
class TextFileArrayWriter(FileArrayWriter[ScalarType]):
    compressed_text_writer: CompressedTextWriter | None = None
    file_handle: IO[str] | None = None
    row_index: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.shape) != 2:
            raise ValueError("TextFileArrayWriter only supports two-dimensional arrays")

    @override
    def set_axis_metadata(self, axis: int, metadata: pd.DataFrame | pd.Series) -> None:
        if axis == 1:
            if not isinstance(metadata, pd.Series):
                raise ValueError("Column metadata must be a Series")
        super().set_axis_metadata(axis, metadata)

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.compressed_text_writer is not None:
            self.compressed_text_writer.close()
            self.file_handle = None
            self.compressed_text_writer = None

    def open(self, col_start: int, col_stop: int) -> None:
        compression_level: int | None = None
        if isinstance(self.compression_method, ZstdTextCompressionMethod):
            compression_level = self.compression_method.level
        self.compressed_text_writer = CompressedTextWriter(
            self.file_path,
            num_threads=self.num_threads,
            compression_level=compression_level,
        )
        self.file_handle = self.compressed_text_writer.open()
        self.write_header(col_start, col_stop)

    def write_header(self, column_start: int, column_stop: int) -> None:
        if self.compressed_text_writer is None or self.file_handle is None:
            raise RuntimeError("File is not open for writing")

        row_metadata, column_metadata = self.axis_metadata
        if column_metadata is None:
            logger.debug(
                "Not writing header for file "
                f'"{str(self.compressed_text_writer.file_path)}" '
                "because column metadata is not available"
            )
            return

        if self.extra_metadata is not None:
            self.file_handle.write(
                f"{extra_metadata_prefix}{json.dumps(self.extra_metadata)}\n"
            )

        header = list()
        if isinstance(row_metadata, pd.DataFrame):
            header.extend(row_metadata.columns.astype(str))
        elif isinstance(row_metadata, pd.Series):
            name = row_metadata.name
            if name is None:
                name = ""
            header.append(name)
        header.extend(column_metadata.iloc[column_start:column_stop].astype(str))
        self.file_handle.write(delimiter.join(header) + "\n")

    def generate_row_prefixes(self, row_start: int, row_stop: int) -> Iterator[bytes]:
        row_metadata, _ = self.axis_metadata
        if row_metadata is None:
            for _ in range(row_start, row_stop):
                yield b""
        else:
            for row_index in range(row_start, row_stop):
                metadata = row_metadata.iloc[row_index]

                if isinstance(metadata, pd.Series):
                    yield delimiter.join(map(to_str, metadata)).encode("utf-8")
                else:
                    yield to_str(metadata).encode("utf-8")

                self.row_index += 1

    def write_values(
        self,
        value: npt.NDArray[ScalarType],
        row_start: int,
        row_stop: int,
    ) -> None:
        if self.file_handle is None:
            raise RuntimeError("File is not open for writing")

        row_count = row_stop - row_start
        write_float(
            self.generate_row_prefixes(row_start, row_stop),
            value[:row_count, :],
            self.file_handle.fileno(),
        )

    def __setitem__(
        self, key: tuple[slice, slice], value: npt.NDArray[ScalarType]
    ) -> None:
        row_start, row_stop, col_start, col_stop = unpack_key(
            self.shape, self.row_index, key
        )

        # Validate key
        row_size = row_stop - row_start
        col_size = col_stop - col_start
        if value.shape != (row_size, col_size):
            raise ValueError(
                f"Value shape {value.shape} does not match key shape "
                f"{(row_size, col_size)}"
            )

        # Open file handle
        if self.compressed_text_writer is None:
            self.open(col_start, col_stop)

        if self.file_handle is None:
            raise RuntimeError("File is not open for writing")

        # Write data
        self.write_values(value, row_start, row_stop)
