import json
from dataclasses import KW_ONLY, dataclass
from types import TracebackType
from typing import IO, Any, Iterator, Self, Type, override

import numpy as np
import pandas as pd
from numpy import typing as npt
from upath import UPath

from ...log import logger
from ...utils.numpy import to_str
from ..pipe import CompressedTextReader, CompressedTextWriter
from ._read_float import read_float
from ._read_str import (
    read_str,
)
from ._write_float import write_float
from .base import (
    FileArrayReader,
    FileArrayWriter,
    ScalarType,
    TextCompressionMethod,
    ZstdTextCompressionMethod,
    compression_method_from_file,
    compression_methods,
)

delimiter: str = "\t"
header_prefix: str = "# "
extra_metadata_prefix: str = "## "


def make_args_tuple(*args: Any) -> tuple[Any, ...]:
    return args


def read_row_metadata(
    reader: CompressedTextReader,
    header_length: int,
    file_column_count: int,
    metadata_column_indices: Any,
) -> list[tuple[Any, ...]]:
    if not isinstance(metadata_column_indices, list):
        raise ValueError("Metadata column indices must be a list")
    if metadata_column_indices != list(range(len(metadata_column_indices))):
        raise ValueError("Metadata columns must be contiguous and start at 0")

    metadata_tuples: list[tuple[Any, ...]] = list()
    with reader as file_handle:
        read_str(
            metadata_tuples,
            make_args_tuple,
            file_handle.fileno(),
            header_length,
            file_column_count,
            np.asarray(metadata_column_indices, dtype=np.uint32),
        )

    return metadata_tuples


def get_header_data(
    reader: CompressedTextReader,
) -> tuple[str | None, dict[str, Any], str | None, int]:
    line: str | None = None
    extra_metadata: dict[str, Any] = dict()
    header_line: str | None = None
    header_length: int = 0
    with reader as file_handle:
        for line in file_handle:
            if line.startswith(extra_metadata_prefix):
                header_length += len(line)
                line = line.removeprefix(extra_metadata_prefix)
                extra_metadata = json.loads(line)
                continue
            if line.startswith(header_prefix):
                header_length += len(line)
                header_line = line.removeprefix(header_prefix)
                continue
            break
    return line, extra_metadata, header_line, header_length


def read_header(
    file_path: UPath,
) -> tuple[pd.DataFrame | None, list[str] | None, dict[str, Any], int, int, int]:
    reader = CompressedTextReader(file_path)
    line, extra_metadata, header_line, header_length = get_header_data(reader)
    if line is None:
        raise ValueError("No line found in text file")

    row_metadata: pd.DataFrame | None = None
    row_count: int | None = None
    column_count: int | None = None

    file_column_count = len(line.strip().split(delimiter))

    if header_line is not None:
        column_names = header_line.strip().split(delimiter)
    else:
        column_names = None

    metadata_column_indices: list[int] | None = None
    if "metadata_column_indices" in extra_metadata:
        metadata_column_indices = extra_metadata.pop("metadata_column_indices")
        if not isinstance(metadata_column_indices, list):
            raise ValueError("`metadata_column_indices` must be a list")

        metadata_tuples = read_row_metadata(
            reader,
            header_length,
            file_column_count,
            metadata_column_indices,
        )

        if column_names is None:
            raise ValueError(
                "Column names must be provided if `metadata_column_indices` is present"
            )

        row_metadata_columns = [
            c for i, c in enumerate(column_names) if i in metadata_column_indices
        ]
        row_metadata = pd.DataFrame(metadata_tuples, columns=row_metadata_columns)
        row_count, _ = row_metadata.shape

        column_names = [
            c for i, c in enumerate(column_names) if i not in metadata_column_indices
        ]
        column_count = len(column_names)

    if column_count is None:
        column_count = file_column_count

    if row_count is None:
        with reader as file_handle:
            file_handle.read(header_length)
            row_count = sum(1 for _ in file_handle)

    return (
        row_metadata,
        column_names,
        extra_metadata,
        header_length,
        row_count,
        column_count,
    )


@dataclass(kw_only=True)
class TextFileArrayReader(FileArrayReader[ScalarType]):
    _: KW_ONLY
    compression_method: TextCompressionMethod
    header_length: int
    column_count: int

    @override
    @classmethod
    def from_file(
        cls, file_path: UPath, dtype: Type[ScalarType], num_threads: int
    ) -> FileArrayReader:
        candidates = [
            "",
            *(
                c.suffix
                for c in compression_methods.values()
                if isinstance(c, TextCompressionMethod)
            ),
        ]

        _file_path: UPath | None = None
        for candidate in candidates:
            _file_path = file_path.parent / f"{file_path.name}{candidate}"
            if _file_path.is_file():
                break
        if _file_path is None:
            raise FileNotFoundError(f"File not found: {file_path}")

        (
            row_metadata,
            column_names,
            extra_metadata,
            header_length,
            row_count,
            column_count,
        ) = read_header(_file_path)
        shape = (row_count, column_count)
        compression_method = compression_method_from_file(_file_path)
        if not isinstance(compression_method, TextCompressionMethod):
            raise ValueError("File is not a text file")
        reader = cls(
            file_path=_file_path,
            shape=shape,
            dtype=dtype,
            compression_method=compression_method,
            extra_metadata=extra_metadata,
            header_length=header_length,
            column_count=column_count,
            row_metadata=row_metadata,
            column_names=column_names,
        )
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

        if np.issubdtype(self.dtype, np.float64):
            with reader as file_handle:
                read_float(
                    array,  # type: ignore
                    file_handle.fileno(),
                    self.header_length,
                    self.column_count + self.metadata_column_count,
                    column_indices,
                    row_indices,
                )
        else:
            raise NotImplementedError


def unpack_key(
    shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    row_index: int,
    key: tuple[slice, slice],
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
        raise ValueError("Can only write file sequentially, row step must be 1")
    if col_step is not None and col_step != 1:
        raise ValueError("Can only write file sequentially, column step must be 1")
    if row_start != row_index:
        raise ValueError(
            "Can only write file sequentially, row start must be "
            f"{row_index} (got {row_start}): {key=} {shape=}"
        )
    if col_start != 0:
        raise ValueError(
            "Can only write file sequentially, column start must be 0 "
            f"(got {col_start}): {key=} {shape=}"
        )
    if col_stop != shape[1]:
        raise ValueError(
            "Can only write file sequentially, column stop must be "
            f"{shape[1]} (got {col_start}): {key=} {shape=}"
        )

    # Validate key
    row_size = row_stop - row_start
    col_size = col_stop - col_start
    if value_shape != (row_size, col_size):
        raise ValueError(
            f"Value shape {value_shape} does not match key shape "
            f"{(row_size, col_size)}: {key}"
        )
    if col_start != 0:
        raise ValueError("Cannot write partial rows to file: column start must be 0")
    if col_stop != shape[1]:
        raise ValueError(
            "Cannot write partial rows to file: "
            f"column stop must be {shape[1]}: {key}"
        )

    return row_start, row_stop, col_start, col_stop


@dataclass(kw_only=True)
class TextFileArrayWriter(FileArrayWriter[ScalarType]):
    compressed_text_writer: CompressedTextWriter | None = None
    file_handle: IO[str] | None = None
    row_index: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.shape) != 2:
            raise ValueError("TextFileArrayWriter only supports two-dimensional arrays")

    def __enter__(self) -> Self:
        compression_level: int | None = None
        if isinstance(self.compression_method, ZstdTextCompressionMethod):
            compression_level = self.compression_method.level
        self.compressed_text_writer = CompressedTextWriter(
            self.file_path,
            num_threads=self.num_threads,
            compression_level=compression_level,
        )
        self.file_handle = self.compressed_text_writer.open()
        self.write_header()
        return self

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

    def write_header(self) -> None:
        if self.compressed_text_writer is None or self.file_handle is None:
            raise RuntimeError("File is not open for writing")

        row_metadata, column_names = self.row_metadata, self.column_names

        header: list[str] = list()
        if column_names is not None:
            if isinstance(row_metadata, pd.DataFrame):
                header.extend(row_metadata.columns.astype(str))
            elif isinstance(row_metadata, pd.Series):
                name = row_metadata.name
                if name is None:
                    name = ""
                header.append(name)
            if header:
                self.extra_metadata["metadata_column_indices"] = list(range(len(header)))
            header.extend(column_names)
        else:
            logger.debug(
                "Not writing header for file "
                f'"{str(self.compressed_text_writer.file_path)}" '
                "because column metadata is not available"
            )
        if self.extra_metadata:
            self.file_handle.write(
                f"{extra_metadata_prefix}{json.dumps(self.extra_metadata)}\n"
            )
        if header:
            self.file_handle.write(f"{header_prefix}{delimiter.join(header)}\n")
        self.file_handle.flush()

    def generate_row_prefixes(
        self, row_start: int, row_stop: int
    ) -> Iterator[bytes | None]:
        if self.row_metadata is None:
            for _ in range(row_start, row_stop):
                yield None
        else:
            for row_index in range(row_start, row_stop):
                metadata = self.row_metadata.iloc[row_index]

                if isinstance(metadata, pd.Series):
                    yield delimiter.join(map(to_str, metadata)).encode("utf-8")
                else:
                    yield to_str(metadata).encode("utf-8")

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
        row_start, row_stop, _, _ = unpack_key(
            self.shape, value.shape, self.row_index, key
        )

        if self.file_handle is None:
            raise RuntimeError("File is not open for writing")

        # Write data
        self.write_values(value, row_start, row_stop)

        row_size = row_stop - row_start
        self.row_index += row_size
