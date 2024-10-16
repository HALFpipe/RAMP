from argparse import Namespace
from functools import partial

import numpy as np
import pandas as pd
from IPython.lib.pretty import pretty
from pandas.api.types import is_object_dtype
from tqdm.auto import tqdm
from upath import UPath

from ..compression.arr.base import (
    CompressionMethod,
    FileArray,
    FileArrayReader,
    FileArrayWriter,
    compression_methods,
)
from ..log import logger
from ..utils.multiprocessing import (
    get_processes_and_num_threads,
    make_pool_or_null_context,
)
from ..vcf.base import VCFFile, base_allele_frequency_columns

dtype = np.float64


def convert(arguments: Namespace, size: int) -> None:
    num_threads = arguments.num_threads

    compression_method = compression_methods[arguments.compression_method]
    suffixes_to_convert = {
        compression_method.suffix
        for name, compression_method in compression_methods.items()
        if name != arguments.compression_method
    }

    paths: list[UPath] = list()
    for path in arguments.path:
        path = UPath(path)
        if path.is_file():
            paths.append(path)
            continue

        for s in suffixes_to_convert:
            for p in path.rglob(f"*{s}"):
                name = p.name.removesuffix(s)
                if name.endswith("score") or name.endswith("covariance"):
                    paths.append(p)

    paths = [path for path in paths if path.is_file()]
    paths.sort()

    logger.info(f"Converting {len(paths)} files: {pretty(paths)}")

    processes, num_threads_per_process = get_processes_and_num_threads(
        num_threads, len(paths), num_threads
    )

    size_per_process = size // processes // dtype().itemsize
    logger.debug(
        f"Using {processes} processes with {num_threads_per_process} threads "
        f"and {size_per_process} bytes of memory per process"
    )

    callable = partial(
        convert_file,
        compression_method=compression_method,
        num_threads=num_threads_per_process,
        size=size_per_process,
        debug=arguments.debug,
    )
    pool, iterator = make_pool_or_null_context(paths, callable, processes)
    with pool:
        for _ in tqdm(iterator, total=len(paths), unit="files"):
            pass


def axis_metadata_path(path: UPath) -> UPath:
    return path.parent / f"{path.stem}.axis-metadata.pkl.zst"


def update_row_metadata_dtypes(
    name: str, row_metadata: pd.DataFrame | pd.Series
) -> pd.DataFrame:
    if name.endswith("score"):
        if isinstance(row_metadata, pd.Series):
            raise ValueError("Score file must have a data frame of row metadata")
        VCFFile.update_data_frame_types(row_metadata)
        row_metadata["is_imputed"] = row_metadata["is_imputed"].astype(bool)
        float_columns: set[str] = {"r_squared"}
        for column in row_metadata.columns:
            if any(column.endswith(s) for s in base_allele_frequency_columns):
                float_columns.add(column)
        for column in float_columns:
            row_metadata[column] = row_metadata[column].astype(np.float64)
        data_frame = row_metadata
    elif name.endswith("covariance"):
        if isinstance(row_metadata, pd.Series):
            data_frame = pd.DataFrame(dict(variable=row_metadata), dtype="string")
    else:
        raise ValueError(f"Unknown file type: {name}")
    return data_frame


def convert_file(
    path: UPath,
    compression_method: CompressionMethod,
    num_threads: int,
    size: int,
    debug: bool,
) -> None:
    converted_path: UPath | None = None
    try:
        logger.debug(f'Reading metadata for "{path}"')
        reader = FileArray.from_file(path, dtype, num_threads=num_threads)
        row_count, column_count = reader.shape

        row_chunk_size = size // column_count
        logger.debug(
            f'Converting "{path}" with shape {reader.shape} '
            f"with {row_chunk_size} rows per chunk"
        )

        name = path.name.removesuffix(reader.compression_method.suffix)
        converted_path = path.parent / f"{name}{compression_method.suffix}"
        if converted_path.is_file():
            logger.warning(
                f'Not converting "{path}" because "{converted_path}" already exists'
            )
        else:
            writer = FileArray.create(
                converted_path,
                (row_count, column_count),
                np.float64,
                compression_method=compression_method,
                num_threads=num_threads,
            )

            row_metadata = reader.row_metadata
            if row_metadata is not None:
                row_metadata = update_row_metadata_dtypes(name, row_metadata)
                for d in row_metadata.dtypes:
                    if is_object_dtype(d):
                        raise ValueError("Refusing to convert object dtype")

                writer.set_axis_metadata(0, row_metadata)
            if reader.column_names is not None:
                writer.set_axis_metadata(1, reader.column_names)

            copy(reader, writer, row_chunk_size)

        converted_reader = FileArray.from_file(
            converted_path, dtype, num_threads=num_threads
        )

        row_chunk_size = row_chunk_size // 2
        logger.debug(
            f'Verifying "{converted_path}" with shape {converted_reader.shape} '
            f"with {row_chunk_size} rows per chunk"
        )
        verify(reader, converted_reader, row_chunk_size)
    except Exception as exception:
        logger.error(f'Error converting "{path}": {exception}', exc_info=True)
        if converted_path is not None:
            logger.info(f'Deleting "{converted_path}" after error', exc_info=True)
            converted_path.unlink(missing_ok=True)
        if debug:
            raise exception


def copy(
    reader: FileArrayReader,
    writer: FileArrayWriter,
    row_chunk_size: int,
) -> None:
    row_count, _ = reader.shape
    with reader, writer:
        for row_start in tqdm(
            range(0, row_count, row_chunk_size),
            desc="converting",
            unit="chunks",
            leave=False,
            position=1,
        ):
            row_end = min(row_start + row_chunk_size, row_count)
            logger.debug(f"Reading chunk from {row_start} to {row_end} of {row_count}")
            row_chunk = reader[row_start:row_end, :]
            logger.debug(f"Writing chunk with shape {row_chunk.shape} at {row_start}")
            writer[row_start:row_end, :] = np.asfortranarray(row_chunk)


def verify(
    reader: FileArrayReader,
    converted_reader: FileArrayReader,
    row_chunk_size: int,
) -> None:
    row_count, _ = reader.shape
    with reader, converted_reader:
        for row_start in tqdm(
            range(0, row_count, row_chunk_size),
            desc="verifying",
            unit="chunks",
            leave=False,
            position=1,
        ):
            row_end = min(row_start + row_chunk_size, row_count)
            logger.debug(f"Verifying chunk from {row_start} to {row_end} of {row_count}")
            row_chunk = reader[row_start:row_end, :]
            converted_row_chunk = converted_reader[row_start:row_end, :]

            if not np.array_equal(row_chunk, converted_row_chunk, equal_nan=True):
                raise ValueError(f"Failed to verify chunk {row_start}:{row_end}")
