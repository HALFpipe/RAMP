import json
from dataclasses import dataclass, field
from functools import cached_property
from types import TracebackType
from typing import Any, Self, Type, override

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy import typing as npt
from pyarrow import parquet as pq
from pyarrow.pandas_compat import dataframe_to_types
from upath import UPath

from .base import FileArrayReader, FileArrayWriter, ParquetCompressionMethod, ScalarType
from .text import unpack_key


@dataclass(kw_only=True)
class ParquetFileArrayWriter(FileArrayWriter[ScalarType]):
    parquet_writer: pq.ParquetWriter | None = None
    row_index: int = 0

    def __post_init__(self) -> None:
        pa.set_cpu_count(self.num_threads)
        return super().__post_init__()

    @cached_property
    def row_metadata_fields(self) -> list[pa.Field]:
        fields: list[pa.Field] = list()
        row_metadata = self.row_metadata
        if row_metadata is not None:
            names, types, _ = dataframe_to_types(row_metadata, preserve_index=False)
            for name, type in zip(names, types, strict=True):
                fields.append(pa.field(name, type))
            self.extra_metadata["metadata_column_indices"] = list(range(len(fields)))
        return fields

    @cached_property
    def arrow_type(self) -> pa.DataType:
        return pa.from_numpy_dtype(np.dtype(self.dtype))

    @cached_property
    def schema(self) -> pa.Schema:
        fields: list[pa.Field] = self.row_metadata_fields.copy()

        _, column_count = self.shape
        column_names: list[str] | None = None
        if self.column_names is not None:
            column_names = self.column_names
        for i in range(column_count):
            if column_names is not None:
                name = column_names[i]
            else:
                name = f"column-{i}"
            fields.append(pa.field(name, self.arrow_type))

        metadata: dict[bytes | str, bytes | str] = dict()
        for key, value in self.extra_metadata.items():
            metadata[key.encode()] = json.dumps(value).encode()

        return pa.schema(fields=fields, metadata=metadata)

    def __enter__(self) -> Self:
        self.parquet_writer = pq.ParquetWriter(
            self.file_path,
            schema=self.schema,
            use_dictionary=False,
            compression="zstd",
            compression_level=9,
        )
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.parquet_writer is not None:
            self.parquet_writer.close()
        self.parquet_writer = None

    def __setitem__(
        self, key: tuple[slice, slice], value: npt.NDArray[ScalarType]
    ) -> None:
        row_start, row_stop, _, _ = unpack_key(
            self.shape, value.shape, self.row_index, key
        )

        if self.parquet_writer is None:
            raise ValueError("Writer is not open")

        arrays: list[pa.Array] = list()

        row_metadata = self.row_metadata
        if row_metadata is not None:
            row_metadata = row_metadata.iloc[row_start:row_stop]
            for field, (_, series) in zip(
                self.row_metadata_fields, row_metadata.items(), strict=True
            ):
                array = pa.Array.from_pandas(series, type=field.type)
                arrays.append(array)

        for column_index in range(value.shape[1]):
            column = value[:, column_index]
            array = pa.Array.from_buffers(
                self.arrow_type,
                column.size,
                # None will produce a null buffer pointer for the bitmap of null values
                [
                    None,  # type: ignore
                    pa.py_buffer(column.data),
                ],
            )
            arrays.append(array)

        record_batch = pa.RecordBatch.from_arrays(arrays, schema=self.schema)
        self.parquet_writer.write_batch(record_batch)

        row_size = row_stop - row_start
        self.row_index += row_size


@dataclass(kw_only=True)
class ParquetFileArrayReader(FileArrayReader[ScalarType]):
    column_names: list[str] = field()
    num_threads: int = field()

    def __post_init__(self) -> None:
        pa.set_cpu_count(self.num_threads)
        return super().__post_init__()

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def read_indices(
        self,
        row_indices: npt.NDArray[np.uint32],
        column_indices: npt.NDArray[np.uint32],
        array: npt.NDArray[ScalarType],
    ) -> None:
        use_threads = self.num_threads > 1

        columns = [
            c for i, c in enumerate(self.column_names) if np.isin(i, column_indices)
        ]

        row_count, _ = self.shape
        row_mask = np.zeros(row_count, dtype=bool)
        row_mask[row_indices] = True

        record_batch_start = 0
        row_start = 0

        with pq.ParquetFile(self.file_path) as parquet_file:
            iterator = parquet_file.iter_batches(
                columns=columns, use_threads=use_threads
            )
            for record_batch in iterator:
                record_batch_end = record_batch_start + record_batch.num_rows

                mask = row_mask[record_batch_start:record_batch_end]
                row_end = row_start + np.count_nonzero(mask)
                if np.any(mask):
                    tensor = record_batch.to_tensor().to_numpy()
                    array[row_start:row_end, :] = tensor[mask, :]

                record_batch_start = record_batch_end
                row_start = row_end

    @override
    @classmethod
    def from_file(
        cls, file_path: UPath, dtype: Type[ScalarType], num_threads: int
    ) -> FileArrayReader:
        use_threads = num_threads > 1
        with pq.ParquetFile(file_path) as parquet_file:
            try:
                schema = parquet_file.schema.to_arrow_schema()
                metadata = schema.metadata

                fields = [
                    pa.field(name, type)
                    for name, type in zip(schema.names, schema.types, strict=True)
                ]

                metadata_column_indices: list[int] = list()
                if b"metadata_column_indices" in metadata:
                    metadata_column_indices = json.loads(
                        metadata.pop(b"metadata_column_indices")
                    )

                row_metadata_fields = [fields[i] for i in metadata_column_indices]
                row_metadata: pd.DataFrame | None = None
                if row_metadata_fields:
                    row_metadata_columns = [
                        fields[i].name for i in metadata_column_indices
                    ]
                    row_metadata = parquet_file.read(
                        columns=row_metadata_columns, use_threads=use_threads
                    ).to_pandas()

                column_names: list[str] = [
                    field.name
                    for i, field in enumerate(fields)
                    if i not in metadata_column_indices
                ]
                column_count = len(column_names)
                row_count = parquet_file.metadata.num_rows
                shape = (row_count, column_count)

                extra_metadata: dict[str, Any] = {
                    key.decode(): json.loads(value.decode())
                    for key, value in metadata.items()
                }

                return cls(
                    file_path=file_path,
                    dtype=dtype,
                    shape=shape,
                    compression_method=ParquetCompressionMethod(),
                    row_metadata=row_metadata,
                    column_names=column_names,
                    extra_metadata=extra_metadata,
                    num_threads=num_threads,
                )
            finally:
                parquet_file.close()
