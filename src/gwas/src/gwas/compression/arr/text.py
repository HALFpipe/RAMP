# -*- coding: utf-8 -*-

from dataclasses import dataclass
from types import TracebackType
from typing import IO, Self, Type

import pandas as pd
from numpy import typing as npt

from ...log import logger
from ...utils import to_str
from ..pipe import CompressedTextWriter
from .base import FileArray, T


@dataclass
class TextFileArray(FileArray[T]):
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
            if row_metadata is not None:
                header.extend(row_metadata.columns.astype(str))
            header.extend(column_metadata.iloc[column_start:column_stop].astype(str))
            self.file_handle.write(self.delimiter.join(header) + "\n")
        else:
            logger.warning(
                "Not writing header for file "
                f'"{str(self.compressed_text_writer.file_path)}" '
                "because column metadata is not available"
            )

    def write_values(
        self,
        value: npt.NDArray[T],
        row_start: int,
        row_stop: int,
        column_start: int,
        column_stop: int,
    ):
        if self.file_handle is None:
            raise RuntimeError("File is not open for writing")
        row_metadata, _ = self.axis_metadata
        for row_index in range(row_start, row_stop):
            row_values: list[str] = list()
            if row_metadata is not None:
                metadata = row_metadata.iloc[row_index]
                if isinstance(metadata, pd.Series):
                    row_values.extend(map(to_str, metadata.values))
                elif metadata is not None:
                    row_values.append(metadata)

            row_values.extend(
                map(
                    to_str,
                    value[
                        row_index - row_start,  # Rows go from 0 of values
                        :,
                    ],
                )
            )

            self.file_handle.write(self.delimiter.join(row_values) + "\n")
            self.current_row_index += 1

    def __setitem__(self, key: tuple[slice, ...], value: npt.NDArray[T]) -> None:
        if len(key) != len(self.shape):
            raise ValueError("Key must have same number of dimensions as array")

        # Unpack key
        row_slice, column_slice = key
        row_start, row_step, row_stop = row_slice.start, row_slice.step, row_slice.stop
        column_start, column_step, column_stop = (
            column_slice.start,
            column_slice.step,
            column_slice.stop,
        )

        # Validate key
        if value.shape != (row_stop - row_start, column_stop - column_start):
            raise ValueError(
                f"Value shape {value.shape} does not match key shape "
                f"{(row_stop - row_start, column_stop - column_start)}"
            )
        if row_step is not None and row_step != 1:
            raise ValueError(
                "Can only write text file sequentially, row step must be 1"
            )
        if column_step is not None and column_step != 1:
            raise ValueError(
                "Can only write text file sequentially, column step must be 1"
            )
        if row_start != self.current_row_index:
            raise ValueError(
                "Can only write text file sequentially, row start must be "
                f"{self.current_row_index} (got {row_start})"
            )
        if column_start != self.current_column_index:
            raise ValueError(
                "Can only write text file sequentially, column start must be "
                f"{self.current_column_index} (got {column_start})"
            )
        if self.has_multiple_parts is None:
            self.has_multiple_parts = column_stop != self.shape[1]
        if not self.has_multiple_parts:
            if column_stop != self.shape[1]:
                raise ValueError(
                    "Cannot change a single part TextFileArray to a multi-part one"
                )

        # Open file handle
        if self.compressed_text_writer is None:
            file_path = self.file_path
            if self.has_multiple_parts:
                file_path = (
                    file_path.parent
                    / f"{file_path.name}.part-{self.current_part_index + 1:d}"
                )
            file_path = (
                file_path.parent
                / f"{file_path.name}.txt{self.compression_method.suffix}"
            )
            self.file_paths.add(file_path)
            self.compressed_text_writer = CompressedTextWriter(
                file_path, num_threads=self.num_threads
            )
            self.file_handle = self.compressed_text_writer.open()
            self.write_header(column_start, column_stop)

        if self.file_handle is None:
            raise RuntimeError("File is not open for writing")

        # Write data
        self.write_values(value, row_start, row_stop, column_start, column_stop)

        # Prepare for next part
        if self.has_multiple_parts:
            if row_stop == self.shape[0]:
                if column_stop == self.shape[1]:
                    self.current_part_index += 1
                    self.current_row_index = 0
                    self.current_column_index = column_stop
                    self.compressed_text_writer.close()
                    self.file_handle = None
                    self.compressed_text_writer = None
