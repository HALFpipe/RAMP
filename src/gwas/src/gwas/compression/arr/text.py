# -*- coding: utf-8 -*-

from dataclasses import dataclass
from itertools import chain
from types import TracebackType
from typing import IO, Iterator, Self, Type

import pandas as pd
from numpy import typing as npt

from ...log import logger
from ...utils import to_str
from ..pipe import CompressedTextWriter
from .base import FileArray, ScalarType, TextCompressionMethod, ZstdTextCompressionMethod


@dataclass
class TextFileArray(FileArray[ScalarType]):
    compression_method: TextCompressionMethod

    compressed_text_writer: CompressedTextWriter | None = None
    file_handle: IO[str] | None = None

    current_part_index: int = 0
    has_multiple_parts: bool | None = None

    current_row_index: int = 0
    current_column_index: int = 0

    delimiter: str = "\t"

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.shape) != 2:
            raise ValueError("TextFileArray only supports two-dimensional arrays")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.compressed_text_writer is not None:
            self.compressed_text_writer.close()

    def write_header(self, column_start: int, column_stop: int) -> None:
        if self.compressed_text_writer is None or self.file_handle is None:
            raise RuntimeError("File is not open for writing")

        row_metadata, column_metadata = self.axis_metadata
        if column_metadata is not None:
            header = list()
            if isinstance(row_metadata, pd.DataFrame):
                header.extend(row_metadata.columns.astype(str))
            elif isinstance(row_metadata, pd.Series):
                name = row_metadata.name
                if name is None:
                    name = ""
                header.append(name)
            header.extend(column_metadata.iloc[column_start:column_stop].astype(str))
            self.file_handle.write(self.delimiter.join(header) + "\n")
        else:
            logger.debug(
                "Not writing header for file "
                f'"{str(self.compressed_text_writer.file_path)}" '
                "because column metadata is not available"
            )

    def write_values(
        self,
        value: npt.NDArray[ScalarType],
        row_start: int,
        row_stop: int,
    ) -> None:
        if self.file_handle is None:
            raise RuntimeError("File is not open for writing")

        row_metadata, _ = self.axis_metadata
        for row_index in range(row_start, row_stop):
            row_value_iterators: list[Iterator[str]] = list()
            if row_metadata is not None:
                metadata = row_metadata.iloc[row_index]
                if isinstance(metadata, pd.Series):
                    row_value_iterators.append(map(to_str, metadata.values))
                elif metadata is not None:
                    row_value_iterators.append(iter([metadata]))

            row_value_iterators.append(
                map(
                    to_str,
                    value[
                        row_index - row_start,  # Rows go from 0 of values
                        :,
                    ],
                )
            )

            self.file_handle.write(
                self.delimiter.join(chain.from_iterable(row_value_iterators))
            )
            self.file_handle.write("\n")
            self.current_row_index += 1

    def unpack_key(self, key: tuple[slice, ...]) -> tuple[int, int, int, int]:
        if len(key) != len(self.shape):
            raise ValueError("Key must have same number of dimensions as array")

        row_slice, col_slice = key
        row_start, row_step, row_stop = row_slice.start, row_slice.step, row_slice.stop
        col_start, col_step, col_stop = col_slice.start, col_slice.step, col_slice.stop

        if col_start is None:
            col_start = 0
        if col_stop is None:
            col_stop = self.shape[1]
        if row_start is None:
            row_start = 0
        if row_stop is None:
            row_stop = self.shape[0]

        self.validate_key(row_start, row_step, col_start, col_step, col_stop)
        return row_start, row_stop, col_start, col_stop

    def validate_key(
        self, row_start: int, row_step: int, col_start: int, col_step: int, col_stop: int
    ) -> None:
        if row_step is not None and row_step != 1:
            raise ValueError("Can only write text file sequentially, row step must be 1")
        if col_step is not None and col_step != 1:
            raise ValueError(
                "Can only write text file sequentially, column step must be 1"
            )
        if row_start != self.current_row_index:
            raise ValueError(
                "Can only write text file sequentially, row start must be "
                f"{self.current_row_index} (got {row_start})"
            )
        if col_start != self.current_column_index:
            raise ValueError(
                "Can only write text file sequentially, column start must be "
                f"{self.current_column_index} (got {col_start})"
            )
        if self.has_multiple_parts is None:
            self.has_multiple_parts = col_stop != self.shape[1]
        if not self.has_multiple_parts:
            if col_stop != self.shape[1]:
                raise ValueError(
                    "Cannot change a single part TextFileArray to a multi-part one"
                )

    def __setitem__(
        self, key: tuple[slice, ...], value: npt.NDArray[ScalarType]
    ) -> None:
        row_start, row_stop, col_start, col_stop = self.unpack_key(key)

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
            file_path = self.file_path
            if self.has_multiple_parts:
                file_path = (
                    file_path.parent
                    / f"{file_path.name}.part-{self.current_part_index + 1:d}"
                )
            if not str(file_path).endswith(self.compression_method.suffix):
                file_path = (
                    file_path.parent
                    / f"{file_path.name}{self.compression_method.suffix}"
                )
            self.file_paths.add(file_path)
            compression_level: int | None = None
            if isinstance(self.compression_method, ZstdTextCompressionMethod):
                compression_level = self.compression_method.level
            self.compressed_text_writer = CompressedTextWriter(
                file_path,
                num_threads=self.num_threads,
                compression_level=compression_level,
            )
            self.file_handle = self.compressed_text_writer.open()
            self.write_header(col_start, col_stop)

        if self.file_handle is None:
            raise RuntimeError("File is not open for writing")

        # Write data
        self.write_values(value, row_start, row_stop)

        if row_stop == self.shape[0]:
            # We have finished writing all rows
            self.compressed_text_writer.close()
            self.file_handle = None
            self.compressed_text_writer = None
        if self.has_multiple_parts:
            # Prepare for next part
            self.current_part_index += 1
            self.current_row_index = 0
            self.current_column_index = col_stop
