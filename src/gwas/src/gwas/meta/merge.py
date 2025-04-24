from argparse import Namespace
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from operator import attrgetter, itemgetter, methodcaller
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq
from tqdm.auto import tqdm
from upath import UPath

from ..log import logger
from .base import marker_key
from .index import Index

index_columns = ["marker_name", "allele1", "allele2"]


def merge(arguments: Namespace) -> None:
    pa.set_cpu_count(arguments.num_threads)
    pa.set_io_thread_count(arguments.num_threads)

    input_directory = UPath(arguments.input_directory)
    output_directory = UPath(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_directory.glob("*population*.parquet"))
    paths = [path for path in paths if not path.stem.endswith("_metal")]

    index = Index()
    for path in paths:
        index.put(path.stem)

    group_by = arguments.group_by
    groups = index.get_tag_groups(group_by)

    for group in tqdm(groups, unit=" " + "groups"):
        phenotypes = index.get_phenotypes(**group)
        group_paths = sorted(
            input_directory / f"{phenotype}.parquet" for phenotype in phenotypes
        )
        logger.debug(f"Processing group {group} with {len(group_paths)} phenotypes")

        tags = [(key, value) for key, value in group.items() if value is not None]
        tags.append(("suffix", "metal"))
        stem = index.format(tags)
        merge_group(group_paths, output_directory / f"{stem}.parquet", arguments)


def merge_group(  # noqa: C901
    paths: Sequence[UPath], output_path: UPath, arguments: Namespace
) -> None:
    with ExitStack() as stack:
        thread_pool_executor = stack.enter_context(
            ThreadPoolExecutor(max_workers=arguments.num_threads)
        )

        parquet_files: list[pq.ParquetFile] = list()
        for path in tqdm(paths, unit=" " + "files"):
            parquet_files.append(stack.enter_context(pq.ParquetFile(path)))

        fields = get_fields(parquet_files)
        names: list[list[str]] = [list(map(attrgetter("name"), f)) for f in fields]

        schema = pa.schema(
            fields=[
                pa.field("marker_name", pa.string()),
                pa.field("allele1", pa.dictionary(pa.uint32(), pa.string())),
                pa.field("allele2", pa.dictionary(pa.uint32(), pa.string())),
                *chain.from_iterable(fields),
            ]
        )

        parquet_writer = stack.enter_context(
            pq.ParquetWriter(
                output_path, schema=schema, compression="zstd", compression_level=9
            )
        )

        marker_key_indices: dict[str, set[int]] = defaultdict(set)
        marker_keys: list[str] = list()

        batch_size: int = arguments.batch_size
        reader_factory = partial(
            TableReader, thread_pool_executor=thread_pool_executor, batch_size=batch_size
        )
        readers = list(
            tqdm(
                map(reader_factory, parquet_files, names),
                unit=" " + "phenotypes",
                total=len(parquet_files),
            )
        )

        def update_marker_keys(phenotype_index: int) -> None:
            reader = readers[phenotype_index]
            if reader.marker_key is None:
                return
            if reader.marker_key not in marker_key_indices:
                marker_keys.append(reader.marker_key)
            marker_key_indices[reader.marker_key].add(phenotype_index)

        for phenotype_index in tqdm(range(len(readers)), unit=" " + "phenotypes"):
            update_marker_keys(phenotype_index)

        progress_bar = stack.enter_context(tqdm(unit=" " + "markers", unit_scale=True))
        future: Future[None] | None = None
        while marker_keys:
            pylist: list[dict[str, str]] = list()
            for row_index in range(batch_size):
                marker_keys.sort()
                if not marker_keys:
                    break
                marker_key = marker_keys.pop(0)

                phenotype_indices = marker_key_indices.pop(marker_key)

                reader = readers[next(iter(phenotype_indices))]
                pylist.append(reader.w.pylist[reader.record_index])

                for phenotype_index in phenotype_indices:
                    reader = readers[phenotype_index]
                    reader.mark(row_index)
                    update_marker_keys(phenotype_index)
                progress_bar.update(1)

            tables: list[pa.Table] = list()
            tables.append(pa.Table.from_pylist(pylist).select(index_columns))
            tables.extend(thread_pool_executor.map(methodcaller("get_table"), readers))
            if future is not None:
                future = future.result()
            future = thread_pool_executor.submit(
                write_tables, parquet_writer, schema, tables
            )
            # write_tables(parquet_writer, schema, tables)


def write_tables(
    parquet_writer: pq.ParquetWriter,
    schema: pa.Schema,
    tables: list[pa.Table],
) -> None:
    arrays: list[pa.Array] = list()
    chunked_arrays: Iterator[pa.ChunkedArray] = chain.from_iterable(
        map(attrgetter("columns"), tables)
    )
    arrays.extend(chain.from_iterable(map(attrgetter("chunks"), chunked_arrays)))

    if len(arrays) != len(schema):
        raise ValueError

    logger.debug("Create batch")
    record_batch = pa.record_batch(arrays, schema=schema)

    logger.debug("Start writing batch")
    parquet_writer.write_batch(record_batch)
    logger.debug("Finished writing batch")


def make_field(name: str, type: pa.DataType) -> pa.Field:
    if name.endswith("_direction"):
        return pa.field(name, pa.dictionary(pa.uint32(), pa.string()))
    return pa.field(name, type)


def get_fields(parquet_files: list[pq.ParquetFile]) -> list[list[pa.Field]]:
    seen: set[str] = {*index_columns}
    fields: list[list[pa.Field]] = []
    for parquet_file in parquet_files:
        schema = parquet_file.schema.to_arrow_schema()
        fields.append(
            [
                make_field(name, type)
                for name, type in zip(schema.names, schema.types, strict=True)
                if name not in seen
            ]
        )
        seen.update(schema.names)
    return fields


@dataclass(frozen=True, slots=True)
class Wrapper:
    record_batch: pa.RecordBatch
    pylist: list[dict[str, str]]
    marker_keys: list[str]


@dataclass
class TableReader:
    parquet_file: pq.ParquetFile
    names: list[str]
    thread_pool_executor: ThreadPoolExecutor
    batch_size: int = 1000

    iterator: Iterator[pa.RecordBatch] = field(init=False)

    _w: Wrapper | None = None
    _w_future: Future[Wrapper | None] | None = None

    record_index: int = field(init=False)
    record_indices: list[int] = field(default_factory=list)

    row_start: int = 0
    row_indices: list[int] = field(default_factory=list)

    row_batches: list[pa.RecordBatch] = field(default_factory=list)

    @property
    def w(self) -> Wrapper:
        if self._w is None:
            if self._w_future is not None:
                self._w = self._w_future.result()
                self._w_future = None
        if self._w is None:
            raise ValueError
        return self._w

    @property
    def marker_key(self) -> str | None:
        w = self.w
        if w.record_batch is None:
            return None
        return w.marker_keys[self.record_index]

    @property
    def size(self) -> int:
        return sum(row_batch.num_rows for row_batch in self.row_batches)

    def __post_init__(self) -> None:
        columns: list[str] = [*index_columns, *self.names]
        self.iterator = self.parquet_file.iter_batches(
            batch_size=self.batch_size, columns=columns
        )
        self.next()

    def next(self) -> None:
        self._w = None
        self._w_future = self.thread_pool_executor.submit(self._next)

    def _next(self) -> Wrapper | None:
        record_batch = next(self.iterator, None)
        self.record_index = 0

        if record_batch is None:
            return None

        pylist = record_batch.select(index_columns).to_pylist()
        marker_keys = list(map(marker_key, map(itemgetter("marker_name"), pylist)))

        return Wrapper(record_batch, pylist, marker_keys)

    def take(self, w: Wrapper) -> None:
        if self.row_indices:
            row_start = self.row_start
            row_end = self.row_indices[-1]
            self.row_start = row_end + 1
            size = (row_end + 1) - row_start
        else:
            size = self.batch_size - self.size

        batch_indices: np.ma.MaskedArray = np.ma.masked_all(size, dtype=np.uint32)
        row_indices = [row_index - row_start for row_index in self.row_indices]
        if not all(0 <= row_index < size for row_index in row_indices):
            raise ValueError
        batch_indices[row_indices] = self.record_indices

        record_indices_array: Any = pa.Array.from_pandas(
            batch_indices.data, mask=batch_indices.mask
        )
        self.row_batches.append(w.record_batch.take(record_indices_array))

        self.row_indices.clear()
        self.record_indices.clear()

    def mark(self, row_index: int) -> None:
        self.row_indices.append(row_index)
        self.record_indices.append(self.record_index)

        self.record_index += 1

        w = self.w
        if self.record_index == w.record_batch.num_rows:
            self.take(w)
            self.next()

    def get_table(self) -> pa.Table:
        while self.size < self.batch_size:
            self.take(self.w)

        table = (
            pa.Table.from_batches(self.row_batches).select(self.names).combine_chunks()
        )

        if table.num_rows != self.batch_size:
            raise ValueError

        self.row_start = 0
        self.row_batches.clear()

        return table
