import pickle
from argparse import Namespace
from functools import partial

import blosc2
import numpy as np
from tqdm.auto import tqdm
from upath import UPath

from ..compression.arr.base import (
    CompressionMethod,
    FileArray,
    compression_methods,
    default_compression_method,
)
from ..compression.pipe import CompressedBytesReader
from ..log import logger
from ..utils import get_processes_and_num_threads, make_pool_or_null_context

suffix_to_convert = ".b2array"


def convert(arguments: Namespace) -> None:
    num_threads = arguments.num_threads

    compression_method: CompressionMethod = default_compression_method
    if arguments.compression_method is not None:
        compression_method = compression_methods[arguments.compression_method]

    path = UPath(arguments.path)
    if path.is_file():
        paths = [path]
    else:
        paths = list(path.rglob(f"*{suffix_to_convert}"))

    paths = [path for path in paths if path.is_file()]

    processes, num_threads_per_process = get_processes_and_num_threads(
        num_threads, len(paths), num_threads
    )

    callable = partial(
        convert_file,
        compression_method=compression_method,
        num_threads=num_threads_per_process,
    )
    pool, iterator = make_pool_or_null_context(paths, callable, processes)
    with pool:
        for _ in tqdm(iterator, total=len(paths), unit="files"):
            pass


def axis_metadata_path(path: UPath) -> UPath:
    return path.parent / f"{path.stem}.axis-metadata.pkl.zst"


def convert_file(
    path: UPath, compression_method: CompressionMethod, num_threads: int
) -> None:
    if path.name.endswith(suffix_to_convert):
        array = blosc2.open(
            urlpath=str(path),
            cparams=dict(nthreads=num_threads),
            dparams=dict(nthreads=num_threads),
        )
        try:
            vlmeta = array.schunk.vlmeta
            axis_metadata_bytes = vlmeta.get_vlmeta("axis_metadata")
            row_metadata, column_metadata = pickle.loads(axis_metadata_bytes)
        except KeyError:
            if axis_metadata_path(path).is_file():
                with CompressedBytesReader(axis_metadata_path(path)) as file_handle:
                    row_metadata, column_metadata = pickle.load(file_handle)
            else:
                row_metadata, column_metadata = None, None

        row_chunk_size, _ = array.chunks
        row_count, column_count = array.shape

        name = path.name.removesuffix(suffix_to_convert)
        stat_file_array_path = path.parent / f"{name}{compression_method.suffix}"
        if stat_file_array_path.is_file():
            logger.warning(
                f'Skipping "{path}" because "{stat_file_array_path}" already exists'
            )
            return
        stat_file_array = FileArray.create(
            stat_file_array_path,
            (row_count, column_count),
            np.float64,
            compression_method=compression_method,
            num_threads=num_threads,
        )
        stat_file_array.set_axis_metadata(0, row_metadata)
        stat_file_array.set_axis_metadata(1, column_metadata)

        with stat_file_array:
            for row_start in tqdm(
                range(0, row_count, row_chunk_size), unit="chunks", leave=False
            ):
                row_end = min(row_start + row_chunk_size, row_count)
                row_chunk = array[row_start:row_end, :]
                stat_file_array[row_start:row_end, :] = np.asfortranarray(row_chunk)
